"""
rules.py — Rules engine for AFWIP.

This is the *verb* layer. Where state.py stores facts and board.py answers
geometry questions, rules.py resolves the actual game: it rolls dice, validates
legality, mutates a `GameState`, and computes Victory Points.

Layering:
    constants / tokens / cards / board   (definitions, pure functions)
        -> state.py                      (mutable facts)
            -> rules.py                  (this file: legality + resolution)
                -> env (Gymnasium/PettingZoo, downstream)

Everything a caller does goes through a `RulesEngine` bound to one GameState and
one RNG (inject a seeded `random.Random` for reproducible play/training).

Scope & fidelity
----------------
Fully modeled:
  - D4 rolls with advantage/disadvantage (one-for-one cancellation, no stacking)
  - Bid for Initiative, Cyber-Rate raise, Play Intel
  - Squadron activation + token generation, incl. Contingency Location roll
  - Move / Acquire / Shoot (air & surface), Winchester, naval salvos
  - Missile Defense (ADA auto-disadvantage; naval declared MD consuming a salvo)
  - Base damage (squadron cards + airbase VP boxes) with destruction cascade
  - Turn alternation, pass-to-end-ATO, end-of-ATO cleanup
  - Cyber-rate instant win and end-of-campaign VP scoring for all missions

Structural (framework, not every card's bespoke text):
  - Enabler play handles the mechanically-general cases automatically —
    token generation, enduring roll-modifier effects (the EW/munitions cards),
    and play-logging that scoring depends on. The ~90 unique one-off effects are
    left as explicit extension points rather than hard-coded here; see
    `ENDURING_ROLL_MODIFIERS` and `play_enabler`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from afwip.core.constants import (
    Side,
    Phase,
    BandID,
    TokenType,
    TokenOrigin,
    TokenScoreType,
    PostureType,
    MissionType,
    IntelTrack,
    EnablerClass,
    RollMode,
    DIE_SIDES,
    WINCHESTER_ROLL_STANDARD,
    DAMAGE_TO_DESTROY_SQUADRON,
    INFANTRY_DAMAGE_BOXES,
    AIRBASE_BONUS_DAMAGE_BOXES,
    CYBER_ACCESS_VALUES,
    MAX_CYBER_RATE,
    CYBER_RATE_WIN,
)
from afwip.core.tokens import TOKEN_REGISTRY
from afwip.core.cards import SQUADRON_REGISTRY, ENABLER_REGISTRY, POSTURE_REGISTRY, MISSION_REGISTRY
from afwip.core.campaigns import CampaignProfile, get_campaign
from afwip.core import board
from afwip.core.state import (
    GameState,
    PlayerState,
    SquadronState,
    TokenInstance,
    CaptureRecord,
    CardZone,
)
from afwip.core.enablers import (
    EnablerPlay, EnablerResult, ENABLER_HANDLERS, RECOVER_AIRCRAFT_CARDS,
    AERIAL_REFUEL_CARDS, FLYING_CREW_CHIEF_CARDS, RAPID_RESUPPLY_CARDS,
    CANCEL_BASE_DAMAGE_CARDS, SUBMARINE_STRIKE_CARDS,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class IllegalAction(Exception):
    """Raised when an attempted action violates the rules."""


# ---------------------------------------------------------------------------
# Roll contexts and enduring roll-modifier table
# ---------------------------------------------------------------------------

class RollContext(Enum):
    """Which kind of roll is being made — drives advantage/disadvantage lookup."""
    ACQUIRE = "ACQUIRE"
    AIR_ATTACK = "AIR_ATTACK"
    SURF_ATTACK = "SURF_ATTACK"    # attack against an enemy surface combatant
    BASE_ATTACK = "BASE_ATTACK"    # attack against an enemy airbase / CL
    MISSILE_DEFENSE = "MISSILE_DEFENSE"


_ATTACK_CONTEXTS = frozenset(
    {RollContext.AIR_ATTACK, RollContext.SURF_ATTACK, RollContext.BASE_ATTACK}
)


@dataclass(frozen=True)
class RollModifier:
    """
    An enduring effect that biases a roll. `applies` decides whether it fires for
    a given (rolling_side, context, token) and yields `mode` when it does.

    owner_side   : side that must have the enduring card active
    rolling_side : side actually making the roll that gets modified
    """
    card_id: int
    rolling_side: Side
    contexts: frozenset
    mode: RollMode
    token_type: Optional[TokenType] = None  # restrict to a specific token

    def applies(self, rolling_side: Side, context: RollContext,
                token: Optional[TokenInstance]) -> bool:
        if rolling_side != self.rolling_side:
            return False
        if context not in self.contexts:
            return False
        if self.token_type is not None:
            if token is None or token.token_type != self.token_type:
                return False
        return True


# Enduring enabler cards whose whole effect is an advantage/disadvantage bias.
# Keyed by the card that must be active; the modifier says whose rolls change.
ENDURING_ROLL_MODIFIERS: dict[int, RollModifier] = {
    # US — Improved Munitions: US air-to-air at advantage.
    19: RollModifier(19, Side.US, frozenset({RollContext.AIR_ATTACK}), RollMode.ADVANTAGE),
    # US — EW Spoofing: PRC acquisition at disadvantage.
    26: RollModifier(26, Side.PRC, frozenset({RollContext.ACQUIRE}), RollMode.DISADVANTAGE),
    # US — Defensive EW: PRC attack rolls at disadvantage.
    27: RollModifier(27, Side.PRC, _ATTACK_CONTEXTS, RollMode.DISADVANTAGE),
    # US — Offensive EW: US attack rolls at advantage.
    28: RollModifier(28, Side.US, _ATTACK_CONTEXTS, RollMode.ADVANTAGE),
    # US — Space-Based EW: PRC base attacks at disadvantage.
    35: RollModifier(35, Side.PRC, frozenset({RollContext.BASE_ATTACK}), RollMode.DISADVANTAGE),
    # PRC — Badger Surge: H-6K rolls at advantage.
    71: RollModifier(71, Side.PRC, _ATTACK_CONTEXTS, RollMode.ADVANTAGE, TokenType.H_6K),
    # PRC — Space-Based EW: US attack rolls at disadvantage.
    90: RollModifier(90, Side.US, _ATTACK_CONTEXTS, RollMode.DISADVANTAGE),
}


# ---------------------------------------------------------------------------
# Roll / result records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RollResult:
    dice: tuple[int, ...]
    mode: RollMode
    natural: int   # chosen die value before bonus
    value: int     # natural + bonus

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Roll({self.dice}, {self.mode.value}, ={self.value})"


@dataclass
class AcquireResult:
    acquirer_uid: int
    target_uid: int
    rolls: list[RollResult]
    success: bool


@dataclass
class ShootResult:
    attacker_uid: int
    context: RollContext
    hit_roll: Optional[RollResult]
    hit: bool
    damage: int
    destroyed_token_uids: list[int] = field(default_factory=list)
    destroyed_squadron_ids: list[int] = field(default_factory=list)
    base_vp_damage: int = 0
    attacker_winchester: bool = False
    missile_defense_uid: Optional[int] = None
    infantry_destroyed: bool = False   # Infantry Battalion marker hit at the base
    # Set when the strike was rolled with defer_allocation=True: the damage has
    # NOT been applied yet — the attacker must call resolve_base_allocation.
    pending_allocation: Optional["PendingBaseAllocation"] = None


# ---------------------------------------------------------------------------
# Dice
# ---------------------------------------------------------------------------

def combine_modes(modes: list[RollMode]) -> RollMode:
    """
    Fold multiple advantage/disadvantage sources into one. They cancel
    one-for-one and never stack, so only the sign of the net count matters.
    """
    net = 0
    for m in modes:
        if m == RollMode.ADVANTAGE:
            net += 1
        elif m == RollMode.DISADVANTAGE:
            net -= 1
    if net > 0:
        return RollMode.ADVANTAGE
    if net < 0:
        return RollMode.DISADVANTAGE
    return RollMode.NORMAL


@dataclass
class LegalAction:
    """
    One executable action available to a player, used by the CLI harness and
    (later) the environment's action masking. `apply` it via
    `RulesEngine.apply_action`.

    kind ∈ {"pass", "activate", "move", "acquire", "shoot_air",
            "shoot_surface", "play_enabler"}
    """
    kind: str
    label: str = ""
    card_id: Optional[int] = None
    token_uid: Optional[int] = None
    target_uid: Optional[int] = None
    dest_band: Optional[BandID] = None
    target_band: Optional[BandID] = None
    play: Optional[EnablerPlay] = None


@dataclass
class _SquadronDamageUndo:
    """Reversal record for damage applied to one squadron during a base attack."""
    defender_side: Side
    card_id: int
    prev_damage: int
    prev_zone: CardZone
    destroyed: bool = False
    revived_tokens: list[TokenInstance] = field(default_factory=list)
    token_captures: list[CaptureRecord] = field(default_factory=list)
    card_capture: Optional[CaptureRecord] = None
    # Never-generated tokens scored on destruction (no instance to revive) —
    # tracked so a cancel-response restores the squadron's tokens_lost.
    phantom_lost: int = 0


@dataclass
class _AttackUndo:
    """
    Everything needed to reverse the most recent attack, so a response card
    (Air Launched Decoy, Decoy Warheads, Red Horse, Resilient Bases) can cancel
    its hit / damage. Populated during shoot/strike resolution.
    """
    attacker_side: Side
    revived_tokens: list[TokenInstance] = field(default_factory=list)
    token_captures: list[Optional[CaptureRecord]] = field(default_factory=list)
    squad_undos: list[_SquadronDamageUndo] = field(default_factory=list)
    base_vp: Optional[tuple[Side, int]] = None
    # Infantry Battalion damage to restore: (side, band, damage_before).
    infantry_damage: Optional[tuple[Side, BandID, int]] = None


@dataclass
class _AttackReapply:
    """
    Redo record for an attack that a cancel-base-damage card (Red Horse /
    Resilient Bases) reversed. Captures the POST-attack state so that if the
    cancel-card is itself cancelled (e.g. by Anti-Access/Area Denial), the
    damage can be re-applied exactly. Squadron rows carry the post-attack field
    values to restore; `dead_tokens` are re-marked destroyed; `captures` go back
    to the attacker.
    """
    attacker_side: Side
    dead_tokens: list[TokenInstance] = field(default_factory=list)
    captures: list[CaptureRecord] = field(default_factory=list)
    # (defender_side, card_id, damage, zone, grounded_token_uids, tokens_lost)
    squads: list[tuple] = field(default_factory=list)
    base_vp: Optional[tuple[Side, int, int]] = None          # (side, damage, scored)
    infantry: Optional[tuple[Side, BandID, int]] = None       # (side, band, damage)


@dataclass
class _CardRecoveryUndo:
    """
    Undo record for a card recovery (Rapid Resupply) so the recovery can be
    reversed if the card is cancelled. For an enabler: its prior zone/enduring.
    For a squadron: its full prior state plus the captures removed from the
    opponent (re-added on reversal).
    """
    side: Side
    card_id: int
    is_squadron: bool
    prev_zone: "CardZone"
    prev_enduring: bool = False
    prev_damage: int = 0
    prev_activated: bool = False
    prev_grounded: list[int] = field(default_factory=list)
    prev_location: Optional[BandID] = None
    prev_tokens_lost: int = 0
    prev_recovered: bool = False
    removed_captures: list[CaptureRecord] = field(default_factory=list)


@dataclass
class PendingBaseAllocation:
    """
    A base strike whose damage roll succeeded with `defer_allocation=True`: the
    amount is known but not yet applied. The attacker distributes it with
    `RulesEngine.resolve_base_allocation`; until then no other turn action is
    accepted. Lets a driver (the RL env) turn the rulebook's free damage
    distribution into explicit per-point decisions instead of a callback.
    """
    attacker_side: Side
    target_side: Side
    target_band: BandID
    amount: int
    targets: list[tuple]          # (kind, id): "squadron"/"token"/"infantry"/"vp"
    result: ShootResult
    undo: _AttackUndo


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RulesEngine:
    """
    Stateful rules resolver bound to one GameState.

    Typical lifecycle::

        engine = RulesEngine(GameState.new_game(campaign=1), rng=random.Random(0))
        engine.setup_missions(us_mission=51, prc_mission=105)
        engine.select_posture(Side.US, 49, squadron_ids=[10], enabler_ids=[])
        engine.select_posture(Side.PRC, 103, squadron_ids=[60], enabler_ids=[])
        engine.bid_for_initiative()
        engine.play_intel()
        engine.begin_player_turns()
        ... issue actions ...
    """

    def __init__(self, state: GameState, rng: Optional[random.Random] = None):
        self.state = state
        self.rng = rng or random.Random()
        # Enabler-resolution scratch state.
        self._resolving_card_id: Optional[int] = None       # card currently in play_enabler
        self._last_attack_undo: Optional[_AttackUndo] = None  # for cancel-attack responses
        self._last_played_card: Optional[tuple[Side, int, EnablerResult]] = None  # for cancel-card
        # Redo/undo records so a cancel-card can itself be cancelled (e.g.
        # Anti-Access/Area Denial voiding a US Red Horse or Rapid Resupply):
        # re-apply an attack a cancel-base-damage card reversed, and undo a card
        # recovery Rapid Resupply performed.
        self._last_attack_reapply: Optional[_AttackReapply] = None
        self._last_card_recovery: Optional[_CardRecoveryUndo] = None
        # Enemy uids the most recent cancel un-acquired (so a cyber-cancel handler
        # can record them on its result for a cancel-of-a-cancel to restore).
        self._reversed_acquired: list[int] = []
        self._initiative_by_ato: dict[int, Side] = {}       # for campaign-specific initiative
        self._md_cancelled_next: bool = False               # Decoy Warheads / ALD: void next declared MD
        # Two-phase drivers (RL env): rolled-but-unresolved intel reveals and
        # a deferred base-strike damage distribution.
        self._pending_intel: dict[Side, int] = {}
        self._pending_allocation: Optional[PendingBaseAllocation] = None
        # Most recent Play Intel dice, keyed by the VIEWER who rolled (for
        # display: the physical game shows the D4 that sets the reveal count).
        self.last_intel_rolls: dict[Side, RollResult] = {}
        # Setup dice (initiative bid, cyber raise, Play Intel) recorded as
        # structured data for the UI to render as clearly-labeled events — these
        # rolls happen automatically (no decision node), so they otherwise show
        # up only as an unlabeled dice blob. The view layer formats the message;
        # the engine stores facts. `_last_cyber_attempt` carries the winner's
        # raise so the bid can log it (enabler-driven raises are not setup rolls).
        self.setup_roll_log: list[dict] = []
        self._last_cyber_attempt: Optional[dict] = None

    # -- dice ---------------------------------------------------------------

    def _d4(self) -> int:
        return self.rng.randint(1, DIE_SIDES)

    def roll(self, mode: RollMode = RollMode.NORMAL, bonus: int = 0,
             note: Optional[str] = None) -> RollResult:
        # `note` is a display-only hint ("hit" / "damage" / "acquire") for the
        # event log; the engine ignores it — the view-layer dice tap reads it so
        # the UI can label which die is which. Kept out of the RollResult.
        if mode == RollMode.NORMAL:
            dice = (self._d4(),)
            natural = dice[0]
        else:
            dice = (self._d4(), self._d4())
            natural = max(dice) if mode == RollMode.ADVANTAGE else min(dice)
        return RollResult(dice=dice, mode=mode, natural=natural, value=natural + bonus)

    @staticmethod
    def _roll_result_data(r: RollResult) -> dict:
        """A RollResult as the UI's die-face payload (matches the dice-tap shape)."""
        return {"natural": r.natural, "value": r.value, "bonus": r.value - r.natural,
                "mode": r.mode.name, "dice": list(r.dice)}

    # -- roll-mode aggregation ---------------------------------------------

    def _enduring_modes(self, rolling_side: Side, context: RollContext,
                        token: Optional[TokenInstance]) -> list[RollMode]:
        """Advantage/disadvantage contributions from active enduring enablers."""
        modes: list[RollMode] = []
        for player in (self.state.us, self.state.prc):
            for enabler in player.enduring_effects():
                mod = ENDURING_ROLL_MODIFIERS.get(enabler.card_id)
                if mod is not None and mod.applies(rolling_side, context, token):
                    modes.append(mod.mode)
        return modes

    def _posture_modes(self, rolling_side: Side, context: RollContext,
                       target_side: Optional[Side], target_band: Optional[BandID]) -> list[RollMode]:
        """Posture-driven biases (currently the US ACE base-defense bonus)."""
        modes: list[RollMode] = []
        if context == RollContext.BASE_ATTACK and target_side == Side.US:
            us = self.state.us
            if us.posture_type == PostureType.ACE:
                # ACE: all strikes against US airbases roll at disadvantage.
                modes.append(RollMode.DISADVANTAGE)
        return modes

    def _ec130_modes(self, rolling_side: Side, attacker: Optional[TokenInstance]) -> list[RollMode]:
        """
        EC-130 Compass Call: any PRC attack made from the same band as a live
        EC-130 token is at disadvantage.
        """
        if attacker is None or rolling_side != Side.PRC:
            return []
        for t in self.state.us.living_tokens():
            if t.token_type == TokenType.EC_130 and t.location == attacker.location:
                return [RollMode.DISADVANTAGE]
        return []

    def attack_mode(self, attacker: TokenInstance, context: RollContext,
                    target_side: Optional[Side] = None,
                    target_band: Optional[BandID] = None,
                    extra: Optional[list[RollMode]] = None) -> RollMode:
        """Net advantage/disadvantage for an attack (to-hit) roll from `attacker`."""
        modes = self._enduring_modes(attacker.side, context, attacker)
        modes += self._posture_modes(attacker.side, context, target_side, target_band)
        modes += self._ec130_modes(attacker.side, attacker)
        if extra:
            modes += extra
        return combine_modes(modes)

    def damage_mode(self, attacker: Optional[TokenInstance], context: RollContext,
                    side: Optional[Side] = None,
                    extra: Optional[list[RollMode]] = None) -> RollMode:
        """
        Net advantage/disadvantage for a DAMAGE roll.

        User ruling 2026-07-31: advantage/disadvantage applies to the ATTACK
        (to-hit) roll, NOT the damage roll — with ONE exception, Missile Defense,
        which the Player Guide keeps on both rolls ("Attacks subject to missile
        defense must roll at disadvantage for both the hit roll and the damage
        roll"). The MD modes are passed in via `extra`, so the damage roll picks
        up disadvantage from a covering ADA / declared naval defender but nothing
        else. Enabler biases (e.g. Badger Surge), EC-130 jamming and posture
        (ACE) are deliberately NOT folded in here — they shape the to-hit roll
        alone. (`attacker`/`context`/`side` are retained for a stable signature.)
        """
        return combine_modes(list(extra) if extra else [])

    # =====================================================================
    # Phase 1 — setup
    # =====================================================================

    @property
    def campaign(self) -> CampaignProfile:
        """The pre-defined campaign's rules for the current game."""
        return get_campaign(self.state.campaign)

    def setup_missions(self, us_mission: int, prc_mission: int) -> None:
        """Assign both Mission Cards, enforcing the campaign's allowed missions."""
        camp = self.campaign
        for side, mid in ((Side.US, us_mission), (Side.PRC, prc_mission)):
            profile = MISSION_REGISTRY[mid]
            if profile.side != side:
                raise IllegalAction(f"Mission {mid} is not a {side.value} card")
            if not camp.mission_allowed(profile.mission_type):
                raise IllegalAction(
                    f"Mission {profile.name} not permitted in Campaign {camp.number} ({camp.name})"
                )
        self.state.us.mission_card_id = us_mission
        self.state.prc.mission_card_id = prc_mission

    def _posture_card_limits(self, profile) -> tuple[int, int]:
        """
        (squadron_limit, enabler_limit) for a posture. The base counts come from
        the card; posture bonus fields (free ADA / bomber slots, PLARF enabler)
        are added as an allowance — a slight over-approximation of the exact
        "these specific cards don't count" wording, but never under-permits.
        """
        squad_limit = profile.squadrons + profile.ada_bonus_squadron + profile.bomber_bonus
        enabler_limit = profile.enablers + (profile.plarf_bonus or 0)
        return squad_limit, enabler_limit

    def _validate_posture_choice(self, side: Side, posture_card_id: int):
        """Shared posture legality: ownership, campaign allowance, no-repeat."""
        camp = self.campaign
        player = self.state.player(side)
        profile = POSTURE_REGISTRY[posture_card_id]
        if profile.side != side:
            raise IllegalAction(f"Posture {posture_card_id} is not a {side.value} card")
        if not camp.posture_allowed(profile.posture_type):
            raise IllegalAction(
                f"Posture {profile.name} not permitted in Campaign {camp.number} ({camp.name})"
            )
        if profile.posture_type != PostureType.STANDARD and profile.posture_type in player.postures_used:
            raise IllegalAction(f"Posture {profile.name} already used this campaign")
        return profile

    def eligible_enabler_pool(self, side: Side, profile) -> list[int]:
        """
        Campaign-legal Enabler Cards still in play for `side`, honoring the
        posture's PLAAF-only restriction. Every ATO cycle drafts from this
        pool (ruling 2026-07-17): played single-use cards (REMOVED) never
        return; unplayed single-use and all multi-use cards are draftable
        again regardless of prior use.
        """
        camp = self.campaign
        player = self.state.player(side)
        if not camp.enablers_allowed:
            return []
        return [cid for cid in sorted(ENABLER_REGISTRY)
                if ENABLER_REGISTRY[cid].side == side and camp.enabler_allowed(cid)
                and not (cid in player.enablers
                         and player.enablers[cid].zone == CardZone.REMOVED)
                and (not profile.plaaf_only or ENABLER_REGISTRY[cid].plaaf)]

    def _check_enabler_draft(self, side: Side, profile,
                             enabler_ids: list[int], pool: list[int]) -> None:
        """
        Validate an enabler draft against the eligible pool: exact posture
        count (relaxed only when the pool has run dry), optional PLARF bonus
        slot, no duplicates, and per-card diagnostics on ineligible picks.
        """
        camp = self.campaign
        if enabler_ids and not camp.enablers_allowed:
            raise IllegalAction(f"Campaign {camp.number} does not use Enabler Cards")
        if len(set(enabler_ids)) != len(enabler_ids):
            raise IllegalAction("Duplicate Enabler Card in the draft")
        for cid in enabler_ids:
            if cid in pool:
                continue
            ep = ENABLER_REGISTRY.get(cid)
            if ep is None or ep.side != side:
                raise IllegalAction(f"Enabler {cid} is not a {side.value} card")
            if not camp.enabler_allowed(cid):
                raise IllegalAction(f"Enabler {ep.name} is banned in Campaign {camp.number}")
            if profile.plaaf_only and not ep.plaaf:
                raise IllegalAction(f"Posture {profile.name}: all enablers must be PLAAF cards")
            raise IllegalAction(f"Enabler {ep.name} is out of play for the campaign")
        plarf_free = min(sum(1 for c in enabler_ids if ENABLER_REGISTRY[c].plarf),
                         profile.plarf_bonus or 0)
        if len(enabler_ids) - plarf_free > profile.enablers:
            raise IllegalAction(
                f"{len(enabler_ids)} enablers exceeds the limit of {profile.enablers} "
                f"(+{profile.plarf_bonus or 0} PLARF bonus)")
        required = min(profile.enablers, len(pool))
        if len(enabler_ids) < required:
            raise IllegalAction(
                f"Posture {profile.name}: exactly {profile.enablers} Enabler Cards "
                f"must be selected ({required} eligible remain); got {len(enabler_ids)}")

    def select_posture_only(self, side: Side, posture_card_id: int,
                            enabler_ids: Optional[list[int]] = None) -> None:
        """
        Convenience later-cycle setup (tests / autoplay): choose a Posture,
        auto-field the surviving squadron roster at its existing base, and draft
        a FRESH enabler hand. The interactive path re-drafts squadrons via
        `select_posture`; this helper just fields all survivors, which is one
        legal re-draft.

          - Squadron Cards persist across cycles: survivors redeploy face-down
            to their previous base — CARD damage AND token losses both persist
            (FAQ 2026-07-24), so a survivor re-fields only its remaining tokens
            (token_count - tokens_lost); destroyed cards stay out; a flying-only
            posture grounds non-flying squadrons for the cycle.
          - Enabler Cards re-draft EVERY cycle per the posture's count:
            played single-use cards (REMOVED) are out for the campaign;
            unplayed single-use and all multi-use cards return to the pool
            and may be drafted again. PLAAF-only postures restrict the pool;
            the PLARF bonus slot applies; the count is exact while the pool
            lasts.
          - `enabler_ids=None` auto-drafts the first eligible cards in id
            order (deterministic convenience for tests and autoplay).
        """
        player = self.state.player(side)
        profile = self._validate_posture_choice(side, posture_card_id)

        pool = self.eligible_enabler_pool(side, profile)
        if enabler_ids is None:
            enabler_ids = pool[:min(profile.enablers, len(pool))]
        self._check_enabler_draft(side, profile, enabler_ids, pool)

        player.posture_card_id = posture_card_id
        player.postures_used.add(profile.posture_type)
        if profile.posture_type == PostureType.SURGE:
            player.cl_banned_campaign = True

        home = board.own_airbase(side)
        for squad in player.squadrons.values():
            if squad.is_destroyed:
                continue
            squad.grounded_token_uids = []      # token losses persist (FAQ 2026-07-24)
            if profile.flying_only and not squad.profile.flying:
                squad.zone = CardZone.DECK      # sits this cycle out
                continue
            squad.zone = CardZone.SELECTED
            squad.activated = False
            if squad.location is None or (player.cl_banned_campaign and
                                          squad.location == BandID.US_CONTINGENCY_LOCATION):
                squad.location = home

        # Fresh hand: exactly the drafted cards; everything else not REMOVED
        # waits in the deck for future cycles.
        from afwip.core.state import EnablerCardState
        chosen = set(enabler_ids)
        for cid, card in player.enablers.items():
            if card.zone == CardZone.REMOVED:
                continue
            card.zone = CardZone.SELECTED if cid in chosen else CardZone.DECK
            card.revealed_to_opponent = False
            card.enduring = False
        for cid in chosen:
            if cid not in player.enablers:
                player.enablers[cid] = EnablerCardState(
                    card_id=cid, side=side, zone=CardZone.SELECTED)

    def select_posture(self, side: Side, posture_card_id: int,
                       squadron_ids: list[int], enabler_ids: list[int],
                       squadron_locations: Optional[dict[int, BandID]] = None) -> None:
        """
        Draft for an ATO cycle: choose a Posture and draft Squadron / Enabler
        cards. Runs EVERY cycle (ruling 2026-07-22): each cycle the player picks
        a posture, then which surviving Squadron Cards to field and where. A
        re-drafted squadron keeps its CARD damage AND its token losses (both
        persist across cycles — FAQ 2026-07-24), so a survivor re-fields only its
        remaining tokens; a squadron not fielded this cycle is set aside with its
        state intact; a destroyed card stays out. Squadrons are deployed
        face-down (SELECTED); enablers go to hand (SELECTED).

        `squadron_locations` optionally places individual Squadron Cards on the
        US Contingency Location instead of the Airbase ("Deploy the Squadron
        Cards face down on the Airbase or Contingency Location container").

        Enforces: card ownership, the no-repeat posture rule, campaign legality
        (allowed posture, forced squadrons, enabler allowance, banned cards),
        the posture's card-count limits and domain restrictions (PLAAF-only /
        flying-only / SURGE's no-CL rule), and that cards knocked out of the
        campaign (destroyed squadrons, spent single-use enablers) are not
        drafted again.
        """
        camp = self.campaign
        player = self.state.player(side)
        profile = self._validate_posture_choice(side, posture_card_id)

        # Campaign-forced squadrons: Campaign 1 requires the side's specific card
        # (US F-16 / PRC J-10) to be INCLUDED in an otherwise-normal Standard
        # draft — not the ONLY card. The rest of the roster is drafted as usual
        # (posture count, near-max floor below), so the forced card is a required
        # subset, not the whole draft.
        forced = camp.forced_squadrons.get(side)
        if forced is not None and not set(forced).issubset(squadron_ids):
            missing = sorted(set(forced) - set(squadron_ids))
            raise IllegalAction(
                f"Campaign {camp.number} requires {side.value} to include squadron(s) {missing}"
            )

        # Enabler allowance and card-count limits. Posture "bonus" slots only
        # absorb their eligible card type (ADA/bomber squadrons; PLARF enabler),
        # so they never raise the base limit for ordinary cards.
        if enabler_ids and not camp.enablers_allowed:
            raise IllegalAction(f"Campaign {camp.number} does not use Enabler Cards")

        def _count_score(ids, score_type) -> int:
            return sum(1 for c in ids
                       if TOKEN_REGISTRY[SQUADRON_REGISTRY[c].token_type].token_score_type == score_type)

        ada_free = min(_count_score(squadron_ids, TokenScoreType.ADA), profile.ada_bonus_squadron)
        bomber_free = min(_count_score(squadron_ids, TokenScoreType.BOMBER), profile.bomber_bonus)
        squad_base = profile.squadrons + player.extra_squadron_slots
        effective_squadrons = len(squadron_ids) - ada_free - bomber_free
        if effective_squadrons > squad_base:
            raise IllegalAction(
                f"{len(squadron_ids)} squadrons exceeds the limit of {squad_base} "
                f"(+{profile.ada_bonus_squadron} ADA, +{profile.bomber_bonus} bomber bonus)"
            )

        plarf_free = min(sum(1 for c in enabler_ids if ENABLER_REGISTRY[c].plarf),
                         profile.plarf_bonus or 0)
        if len(enabler_ids) - plarf_free > profile.enablers:
            raise IllegalAction(
                f"{len(enabler_ids)} enablers exceeds the limit of {profile.enablers} "
                f"(+{profile.plarf_bonus or 0} PLARF bonus)"
            )

        # Cards knocked out of the campaign earlier stay out.
        dead_squadrons = {cid: s for cid, s in player.squadrons.items() if s.is_destroyed}

        # The Squadron Card count is a NEAR-maximum, not exact (user ruling
        # 2026-07-31, reverting the 2026-07-13 exact ruling): you may field one
        # BELOW the posture's number — i.e. at least (N-1). If you don't have
        # enough surviving squadrons to reach N-1, field ALL that remain. The
        # ceiling stays N (+ bonus slots), checked above; bonus slots (ADA /
        # bomber) stay optional. This applies to campaign-forced drafts too
        # (Campaign 1): the forced card counts toward the roster.
        sq_pool = [cid for cid, sp in SQUADRON_REGISTRY.items()
                   if sp.side == side and camp.squadron_allowed(cid)
                   and cid not in dead_squadrons
                   and (not profile.flying_only or sp.flying)]
        required = min(max(profile.squadrons - 1, 0), len(sq_pool))
        if len(squadron_ids) < required:
            raise IllegalAction(
                f"Posture {profile.name}: at least {required} Squadron Card(s) must "
                f"be selected (one below the posture's {profile.squadrons}, or all "
                f"{len(sq_pool)} that remain); got {len(squadron_ids)}"
            )
        spent_enablers = {cid: e for cid, e in player.enablers.items()
                          if e.zone == CardZone.REMOVED}

        # Validate the whole draft before mutating any state.
        home = board.own_airbase(side)
        locations: dict[int, BandID] = {}
        for cid in squadron_ids:
            sp = SQUADRON_REGISTRY[cid]
            if sp.side != side:
                raise IllegalAction(f"Squadron {cid} is not a {side.value} card")
            if not camp.squadron_allowed(cid):
                raise IllegalAction(
                    f"Squadron {sp.name} ({sp.token_type.value}) is banned in Campaign {camp.number}"
                )
            if cid in dead_squadrons:
                raise IllegalAction(f"Squadron {sp.name} was destroyed earlier in the campaign")
            if profile.flying_only and not sp.flying:
                raise IllegalAction(f"Posture {profile.name}: all Squadrons must be flying units")
            loc = (squadron_locations or {}).get(cid, home)
            if loc not in board.own_base_locations(side):
                raise IllegalAction(f"Squadron {cid} must deploy to a {side.value} base, not {loc.name}")
            if loc == BandID.US_CONTINGENCY_LOCATION and (
                    profile.posture_type == PostureType.SURGE or player.cl_banned_campaign):
                raise IllegalAction("SURGE: Contingency Locations cannot be used during this campaign")
            locations[cid] = loc

        # The posture's Enabler Card count is exact — "each player may choose
        # the number of Enabler Cards specified on the selected Posture Card"
        # (unlike squadrons, which are "up to"). A player may not draft fewer;
        # only when the eligible pool has run dry may they field fewer, and
        # must then take everything that remains. The PLARF bonus stays optional.
        if camp.enablers_allowed:
            pool = [cid for cid, ep in ENABLER_REGISTRY.items()
                    if ep.side == side and camp.enabler_allowed(cid)
                    and cid not in spent_enablers
                    and (not profile.plaaf_only or ep.plaaf)]
            required = min(profile.enablers, len(pool))
            if len(enabler_ids) < required:
                raise IllegalAction(
                    f"Posture {profile.name}: exactly {profile.enablers} Enabler Cards "
                    f"must be selected ({required} eligible remain); got {len(enabler_ids)}"
                )

        for cid in enabler_ids:
            ep = ENABLER_REGISTRY[cid]
            if ep.side != side:
                raise IllegalAction(f"Enabler {cid} is not a {side.value} card")
            if not camp.enabler_allowed(cid):
                raise IllegalAction(f"Enabler {ep.name} is banned in Campaign {camp.number}")
            if cid in spent_enablers:
                raise IllegalAction(f"Enabler {ep.name} is out of play for the campaign")
            if profile.plaaf_only and not ep.plaaf:
                raise IllegalAction(f"Posture {profile.name}: all enablers must be PLAAF cards")

        # Commit.
        player.posture_card_id = posture_card_id
        player.postures_used.add(profile.posture_type)
        if profile.posture_type == PostureType.SURGE:
            # "Contingency Locations cannot be used during this campaign."
            player.cl_banned_campaign = True

        # Squadron Cards persist across cycles, so keep every existing card and
        # only re-draft which are FIELDED this cycle. A fielded survivor keeps its
        # TOKEN losses (they persist — it re-fields only its remaining tokens), but
        # its partial CARD damage was already reset at cleanup (ruling 2026-07-31);
        # it takes its newly-chosen placement. A card not fielded is set aside
        # (DECK) with its state intact; destroyed cards stay out.
        drafted = set(squadron_ids)
        for cid, squad in player.squadrons.items():
            if squad.is_destroyed:
                continue
            if cid in drafted:
                squad.zone = CardZone.SELECTED
                squad.location = locations[cid]
                squad.activated = False
                squad.grounded_token_uids = []
                # token losses persist across cycles (FAQ 2026-07-24)
            else:
                squad.zone = CardZone.DECK
        for cid in squadron_ids:
            if cid not in player.squadrons:
                player.squadrons[cid] = SquadronState(
                    card_id=cid, side=side, zone=CardZone.SELECTED, location=locations[cid]
                )

        from afwip.core.state import EnablerCardState
        player.enablers = dict(spent_enablers)
        for cid in enabler_ids:
            player.enablers[cid] = EnablerCardState(
                card_id=cid, side=side, zone=CardZone.SELECTED
            )

    # =====================================================================
    # Phase 2 — bid for initiative
    # =====================================================================

    def _token_kills(self, side: Side, ato_cycle: int) -> int:
        """Enemy tokens (not squadron cards) this side destroyed during an ATO."""
        return sum(1 for c in self.state.player(side).captures
                   if c.ato_cycle == ato_cycle and not c.is_squadron_card)

    def _sacrifice_for_bid(self, side: Side, card_ids: list[int]) -> int:
        """
        Sacrifice Enabler Cards from hand for the initiative bid: each adds +1 to
        the roll and the card is returned to the player's deck (unavailable for
        the rest of this ATO cycle, but not removed from the campaign).
        """
        player = self.state.player(side)
        for cid in card_ids:
            card = player.enablers.get(cid)
            if card is None or card.zone != CardZone.SELECTED:
                raise IllegalAction(f"{side.value} cannot sacrifice enabler {cid}: not in hand")
            card.zone = CardZone.DECK
        return len(card_ids)

    def bid_for_initiative(self, us_sacrifice: Optional[list[int]] = None,
                           prc_sacrifice: Optional[list[int]] = None,
                           first_player: Optional[Side] = None) -> Side:
        """
        Resolve initiative. Normally each side rolls a D4 (+1 per Enabler Card it
        sacrifices back to its deck), ties reroll retaining the modifiers, and
        the winner takes Intel Advantage, picks the first player, and attempts
        one Cyber-Rate raise.

        Campaign overrides for ATO cycles after the first:
          - Campaign 2 (Tournament): no re-bid and no sacrifices; initiative
            passes to whoever did NOT hold it in ATO 1.
          - Campaign 4 (The World Watches): initiative goes to the side leading in
            tokens destroyed after ATO 1 (ties fall back to a normal roll).
        These auto-assignments skip the bid roll (and its Cyber-Rate raise).
        """
        camp = self.campaign
        ato = self.state.ato_cycle
        auto_winner: Optional[Side] = None

        if ato > 1 and camp.number == 2:
            prev = self._initiative_by_ato.get(1)
            if prev is not None:
                auto_winner = board.opponent(prev)
        elif ato > 1 and camp.number == 4:
            # "The World Watches": the side that LOST more of its own tokens to
            # the enemy in ATO 1 gains initiative in ATO 2 — a shift in
            # international opinion toward the side taking losses (user ruling
            # 2026-07-22). A side's own losses = the tokens its opponent
            # destroyed. Ties fall back to a normal roll.
            us_losses = self._token_kills(Side.PRC, 1)   # US tokens the PRC destroyed
            prc_losses = self._token_kills(Side.US, 1)   # PRC tokens the US destroyed
            if us_losses != prc_losses:
                auto_winner = Side.US if us_losses > prc_losses else Side.PRC

        if camp.number == 2 and (us_sacrifice or prc_sacrifice):
            raise IllegalAction("Tournament: cannot sacrifice Enabler Cards for the bid")

        us_mod = self._sacrifice_for_bid(Side.US, us_sacrifice or [])
        prc_mod = self._sacrifice_for_bid(Side.PRC, prc_sacrifice or [])

        us_face = prc_face = None
        if auto_winner is not None:
            winner = auto_winner
        else:
            while True:
                us_face = self._d4()
                prc_face = self._d4()
                us_roll = us_face + us_mod
                prc_roll = prc_face + prc_mod
                if us_roll != prc_roll:
                    winner = Side.US if us_roll > prc_roll else Side.PRC
                    break

        self.state.initiative_holder = winner
        self._initiative_by_ato[ato] = winner
        self.state.us.intel_track = IntelTrack.ADVANTAGE if winner == Side.US else IntelTrack.NORMAL
        self.state.prc.intel_track = IntelTrack.ADVANTAGE if winner == Side.PRC else IntelTrack.NORMAL
        self.state.first_player = first_player or winner

        # Record the bid for the UI's setup-roll log (facts only; view formats).
        bid_entry: dict = {"kind": "initiative", "winner": winner.value,
                           "auto": auto_winner is not None}
        if auto_winner is None:
            bid_entry["us"] = {"natural": us_face, "bonus": us_mod,
                               "value": us_face + us_mod, "mode": "NORMAL",
                               "dice": [us_face]}
            bid_entry["prc"] = {"natural": prc_face, "bonus": prc_mod,
                                "value": prc_face + prc_mod, "mode": "NORMAL",
                                "dice": [prc_face]}
        self.setup_roll_log.append(bid_entry)

        # The Cyber-Rate raise is a reward for winning the bid roll, so it only
        # applies to an actual bid — not an auto-assigned initiative.
        if auto_winner is None:
            self.attempt_cyber_raise(winner)
            if self._last_cyber_attempt is not None:
                self.setup_roll_log.append({"kind": "cyber_raise",
                                            **self._last_cyber_attempt})
        return winner

    def choose_first_player(self, first: Side) -> None:
        """
        The initiative winner "chooses who will take the first turn". Overrides
        the default (winner first) set by bid_for_initiative; must be called
        before begin_player_turns.
        """
        if self.state.initiative_holder is None:
            raise IllegalAction("Initiative has not been resolved")
        if self.state.phase == Phase.PLAYER_TURN:
            raise IllegalAction("Player turns have already begun")
        self.state.first_player = first

    def attempt_cyber_raise(self, side: Side) -> bool:
        """
        Roll to raise Cyber Rate by one step. Succeeds if the D4 meets the access
        value between the current and next rate. Returns True on success (and may
        trigger an instant cyber win at rate 4).
        """
        player = self.state.player(side)
        self._last_cyber_attempt = None
        if player.cyber_rate >= MAX_CYBER_RATE:
            return False
        needed = CYBER_ACCESS_VALUES.get(player.cyber_rate)
        if needed is None:
            return False
        face = self._d4()
        old = player.cyber_rate
        success = face >= needed
        if success:
            self.state.set_cyber_rate(side, player.cyber_rate + 1)
        self._last_cyber_attempt = {"side": side.value, "natural": face,
                                    "needed": needed, "success": success,
                                    "from": old, "to": player.cyber_rate}
        return success

    # =====================================================================
    # Phase 3 — play intel
    # =====================================================================

    def play_intel_roll(self) -> dict[Side, int]:
        """
        Roll both sides' intel checks (Intel-Advantage side rolls at advantage).
        Returns {card_owner: n} — how many of that owner's hand enablers must be
        shown to the opponent (at least one always stays hidden). The owner's
        card choices are then applied via `play_intel_reveal` — or use the
        one-shot `play_intel` wrapper.
        """
        self.state.phase = Phase.PLAY_INTEL
        counts: dict[Side, int] = {}
        self.last_intel_rolls = {}
        for viewer in (Side.US, Side.PRC):
            vp = self.state.player(viewer)
            mode = RollMode.ADVANTAGE if vp.intel_track == IntelTrack.ADVANTAGE else RollMode.NORMAL
            roll = self.roll(mode)
            self.last_intel_rolls[viewer] = roll
            opp = self.state.opponent(viewer)
            hand = opp.enablers_in_hand()
            counts[opp.side] = max(0, min(roll.value, len(hand) - 1))  # >=1 stays hidden
        self._pending_intel = dict(counts)
        # Record for the UI: each side's roll and how many enemy cards it sees
        # (US sees COUNTS of the PRC hand, and vice versa).
        self.setup_roll_log.append({
            "kind": "play_intel",
            "us": {"roll": self._roll_result_data(self.last_intel_rolls[Side.US]),
                   "sees": counts.get(Side.PRC, 0)},
            "prc": {"roll": self._roll_result_data(self.last_intel_rolls[Side.PRC]),
                    "sees": counts.get(Side.US, 0)},
        })
        return counts

    def play_intel_reveal(self, owner: Side, chosen_ids: list[int]) -> None:
        """Reveal exactly the rolled number of `owner`'s hand cards to the opponent."""
        n = self._pending_intel.get(owner)
        if n is None:
            raise IllegalAction(f"No pending intel reveal for {owner.value}")
        hand = {c.card_id: c for c in self.state.player(owner).enablers_in_hand()}
        distinct = list(dict.fromkeys(chosen_ids))
        if len(distinct) != n or any(cid not in hand for cid in distinct):
            raise IllegalAction(f"{owner.value} must reveal exactly {n} distinct hand card(s)")
        for cid in distinct:
            hand[cid].revealed_to_opponent = True
        del self._pending_intel[owner]

    def play_intel(self,
                   reveal_selectors: Optional[dict[Side, Callable[[list[int], int], list[int]]]] = None
                   ) -> dict[Side, int]:
        """
        One-shot Play Intel: roll both checks and apply the reveals. Marks
        revealed cards; returns counts seen per viewer.

        The card *owner* chooses which of their cards to show. Pass
        `reveal_selectors[owner_side] = fn(hand_card_ids, n) -> chosen_ids` to
        make that choice (the interactive harness prompts the owner); without a
        selector the first n cards in hand are shown.
        """
        counts = self.play_intel_roll()
        for owner, n in counts.items():
            hand_ids = [c.card_id for c in self.state.player(owner).enablers_in_hand()]
            selector = (reveal_selectors or {}).get(owner)
            if selector is not None and n > 0:
                picked = [cid for cid in dict.fromkeys(selector(list(hand_ids), n))
                          if cid in hand_ids][:n]
                # Pad from hand order if the selector under-delivered.
                picked += [cid for cid in hand_ids if cid not in picked][:n - len(picked)]
            else:
                picked = hand_ids[:n]
            self.play_intel_reveal(owner, picked)
        return {board.opponent(owner): n for owner, n in counts.items()}

    def begin_player_turns(self) -> None:
        """Enter the turn loop; the designated first player acts first."""
        self.state.phase = Phase.PLAYER_TURN
        self.state.turn_number = 1
        if self._tick_turn():
            return
        self._apply_posture_bonus_tokens()
        self.state.active_side = self.state.first_player or self.state.initiative_holder or Side.US
        self._start_turn(self.state.active_side)

    def _tick_turn(self) -> bool:
        """
        Advance the monotonic whole-game turn counter and enforce the timed-game
        cap (`state.max_turns`). Returns True — and finalizes the game by VP —
        when the cap is reached, so the caller skips starting the next turn.
        Real games are time-limited; at the cap the leader on VP wins. A no-op
        once the game is already over.
        """
        if self.state.game_over:
            return True
        self.state.total_turns += 1
        if (self.state.max_turns is not None
                and self.state.total_turns >= self.state.max_turns):
            self._finalize_game()
            return True
        return False

    def _apply_posture_bonus_tokens(self) -> None:
        """
        Generate free tokens granted by a posture. Currently only US HEDGEHOG,
        whose card grants 1x ADA token. The owner may place it on the Airbase or
        the Contingency Location (`posture_bonus_ada_location`, set during setup;
        the Airbase is the default). These are free (ENABLER_CARD origin) and do
        not count as squadron-card losses.
        """
        for side in (Side.US, Side.PRC):
            player = self.state.player(side)
            posture = player.posture
            if posture is None or posture.ada_bonus_token <= 0:
                continue
            ada_type = TokenType.ADA_US if side == Side.US else None
            if ada_type is None:
                continue
            band = player.posture_bonus_ada_location or board.own_airbase(side)
            if band not in board.own_base_locations(side) or (
                    player.cl_banned_campaign and band == BandID.US_CONTINGENCY_LOCATION):
                band = board.own_airbase(side)
            for _ in range(posture.ada_bonus_token):
                self.state.spawn_token(side, ada_type, band, TokenOrigin.ENABLER_CARD)

    # =====================================================================
    # Turn structure
    # =====================================================================

    def _require_turn(self, side: Side) -> PlayerState:
        if self.state.game_over:
            raise IllegalAction("Game is over")
        if self.state.phase != Phase.PLAYER_TURN:
            raise IllegalAction(f"Not in the player-turn phase (phase={self.state.phase})")
        if side != self.state.active_side:
            raise IllegalAction(f"It is not {side.value}'s turn")
        if self._pending_allocation is not None:
            raise IllegalAction("A base strike is awaiting damage allocation")
        # A new action opens: the previous attack's undo is stale. Response
        # cards are "immediately after" effects — Reserves, Personnel Recovery,
        # Quick-Turn Mobility and the cancel-attack cards must not fire off an
        # attack that resolved turns ago (user report 2026-07-17: Reserves was
        # offered in response to an unrelated Defensive EW play). Responses skip
        # _require_turn, so the window that legitimately needs the undo keeps it.
        self._last_attack_undo = None
        return self.state.player(side)

    def _start_turn(self, side: Side) -> None:
        """
        Beginning-of-turn housekeeping: Winchester air tokens return to their
        squadron card (or are surrendered if that card is destroyed), and
        Winchester naval tokens leave the board. Free movement, not an action.
        """
        player = self.state.player(side)
        for token in list(player.living_tokens()):
            if not token.is_winchester:
                continue
            if token.is_naval:
                # Winchester naval tokens move off the board (free movement). They
                # are not destroyed and do not score; they are simply off the map.
                token.off_board = True
                continue
            # Air token returns to base.
            if token.origin == TokenOrigin.SQUADRON_CARD and token.source_card_id is not None:
                squad = player.squadrons.get(token.source_card_id)
                if squad is None or squad.is_destroyed:
                    # Any airborne unit landing on a destroyed squadron is lost.
                    self.state.destroy_token(
                        token.uid, destroyed_by=self.state.opponent(side).side, on_ground=True
                    )
                else:
                    # Back on its Squadron Card: grounded, and destroyed with the
                    # card if the squadron is later killed.
                    token.location = squad.location or board.own_airbase(side)
                    token.grounded = True
                    if token.uid not in squad.grounded_token_uids:
                        squad.grounded_token_uids.append(token.uid)
            else:
                # Enabler-origin air token with no home simply leaves play.
                self.state.destroy_token(token.uid)

    def end_turn(self, side: Side) -> None:
        """
        Conclude `side`'s turn: score per-turn missions, reset its action flags,
        then hand off. If the player did nothing this turn it counts as a pass;
        two passes in succession end the ATO cycle.
        """
        player = self._require_turn(side)

        # `acted_this_turn` is set by record_action for every non-pass action
        # (move/acquire/shoot/activate and playing an enabler).
        acted = player.acted_this_turn

        self._score_end_of_turn(side)
        self._reap_orphaned_tokens()

        if acted:
            ended = False  # the pass streak was already broken when the action was taken
        else:
            ended = self.state.record_pass(side)

        player.reset_turn()

        if ended:
            self.end_ato_cycle()
            return

        # Hand off to the opponent and run their start-of-turn housekeeping.
        self.state.active_side = self.state.opponent(side).side
        self.state.turn_number += 1
        if self._tick_turn():
            return
        self._start_turn(self.state.active_side)

    def _reap_orphaned_tokens(self) -> None:
        """
        Turn over the tokens of a destroyed squadron the instant they are
        Winchester or grounded (ruling 2026-07-17): such tokens keep flying and
        attacking until then, then are surrendered to the opponent (shot-down
        and end-of-ATO cases are scored at their own sites). Runs at the settled
        end-of-turn boundary, after this turn's cancel/response windows have
        closed — so a response that revives the squadron (Red Horse / Resilient
        Bases) pre-empts the surrender rather than racing it.
        """
        for side in (Side.US, Side.PRC):
            player = self.state.player(side)
            opp = self.state.opponent(side).side
            for token in list(player.living_tokens()):
                if token.origin != TokenOrigin.SQUADRON_CARD or token.source_card_id is None:
                    continue
                if not (token.is_winchester or token.grounded):
                    continue                       # airborne with weapons: keeps flying
                squad = player.squadrons.get(token.source_card_id)
                if squad is None or squad.is_destroyed:
                    self.state.destroy_token(token.uid, destroyed_by=opp, on_ground=True)

    def pass_turn(self, side: Side) -> None:
        """Explicitly pass. Equivalent to ending a turn with no action taken."""
        player = self._require_turn(side)
        # Force the "no action" branch even if flags somehow set.
        player.has_moved = player.has_acquired = player.has_shot = False
        player.squadrons_activated_this_turn = []
        player.enabler_tokens_generated_this_turn = 0
        self.end_turn(side)

    # =====================================================================
    # Action: activate squadron
    # =====================================================================

    def activate_squadron(self, side: Side, card_id: int,
                          spawn_band: Optional[BandID] = None) -> list[TokenInstance]:
        """
        Flip a Squadron Card face up and generate its tokens.

        `spawn_band` picks the placement band where the squadron allows a choice
        (UAS anywhere; AEW/bombers first-band-or-standoff). If omitted, the first
        legal band from the token's spawn set is used. Contingency-Location
        squadrons roll to see how many tokens generate; the rest stay grounded.

        Activating a squadron ends the player's turn.
        """
        player = self._require_turn(side)
        # "Do one or the other: Activate a Squadron Card / Move-Acquire-Shoot"
        # — mutually exclusive within a turn (activation also ends the turn,
        # so the reverse order is impossible by construction).
        if player.has_moved or player.has_acquired or player.has_shot:
            raise IllegalAction(
                "Activating a squadron and Move-Acquire-Shoot are mutually exclusive this turn"
            )
        squad = player.squadrons.get(card_id)
        if squad is None:
            raise IllegalAction(f"{side.value} has not drafted squadron {card_id}")
        if squad.activated:
            raise IllegalAction(f"Squadron {card_id} already activated")
        if squad.is_destroyed:
            raise IllegalAction(f"Squadron {card_id} is destroyed")

        # Recovered squadron: reactivation is risky (FAQ 2026-07-24). Roll a D4;
        # on a 1 every aircraft it would field is "broken" and surrendered to the
        # opponent for points (on the ground) — the card itself is not scored.
        if squad.recovered:
            squad.recovered = False
            if self._d4() == 1:
                return self._break_recovered_squadron(side, squad)

        token_type = squad.token_type
        profile = TOKEN_REGISTRY[token_type]
        legal_bands = board.valid_spawn_locations(token_type, side, player.posture_type or PostureType.STANDARD)
        if player.cl_banned_campaign:
            legal_bands = legal_bands - {BandID.US_CONTINGENCY_LOCATION}
        if not legal_bands:
            raise IllegalAction(f"No legal spawn band for {token_type.value}")

        band = spawn_band if spawn_band is not None else self._default_spawn_band(side, legal_bands)
        if band not in legal_bands:
            raise IllegalAction(f"{token_type.value} may not spawn at {band.name}")

        # Destroyed tokens do not regenerate: activation fields only the
        # squadron's surviving complement (winchester survivors rearm in full).
        # Tokens of this squadron already in play (a revived card whose flights
        # were still airborne) count against the cap — a player can never field
        # more tokens than the token count (user ruling 2026-07-15).
        already_alive = sum(
            1 for t in player.tokens.values()
            if t.is_active and t.origin == TokenOrigin.SQUADRON_CARD
            and t.source_card_id == card_id
        )
        count = max(0, profile.token_count - squad.tokens_lost - already_alive)
        generated = count
        from_cl = squad.location == BandID.US_CONTINGENCY_LOCATION
        if from_cl and player.posture_type != PostureType.ACE:
            # Flying Crew Chief: the squadron its owner named generates max
            # aircraft with no roll; every other CL squadron rolls as usual.
            if player.cl_max_squadron_id == card_id:
                player.cl_max_squadron_id = None    # consumed by this squadron
            else:
                generated = board.contingency_tokens_generated(self._d4(), count)

        tokens: list[TokenInstance] = []
        for i in range(count):
            grounded = i >= generated
            loc = squad.location if grounded else band
            tok = self.state.spawn_token(
                side, token_type, loc, TokenOrigin.SQUADRON_CARD,
                source_card_id=card_id, grounded=grounded,
            )
            tokens.append(tok)

        squad.activated = True
        squad.ever_activated = True   # identity now public for the campaign
        squad.zone = CardZone.ACTIVE
        player.squadrons_activated_this_turn.append(card_id)
        self.state.record_action(side)

        # Generating forces ends the turn.
        self.end_turn(side)
        return tokens

    def _break_recovered_squadron(self, side: Side, squad) -> list[TokenInstance]:
        """
        A recovered squadron whose reactivation rolled a 1 (FAQ 2026-07-24):
        every aircraft it would have fielded is "broken" and surrendered to the
        opponent for points as an on-ground kill. The Squadron Card itself is
        NOT scored — it is simply removed from play (destroyed a second time).
        Like any activation, this ends the turn.
        """
        opp = self.state.opponent(side)
        profile = TOKEN_REGISTRY[squad.token_type]
        broken = max(0, profile.token_count - squad.tokens_lost)
        for _ in range(broken):
            opp.captures.append(CaptureRecord(
                ato_cycle=self.state.ato_cycle, is_squadron_card=False,
                destroyed_on_ground=True, token_type=squad.token_type,
                score_type=profile.token_score_type, uid=None, card_id=squad.card_id))
        squad.tokens_lost = profile.token_count
        squad.activated = True
        squad.ever_activated = True
        squad.zone = CardZone.DESTROYED        # removed from play; the card is not scored
        self.state.record_action(side)
        self.end_turn(side)
        return []

    def relaunch_candidates(self, side: Side) -> list[int]:
        """
        Winchester FIGHTERS that returned to a live, on-board Squadron Card and
        may attempt a relaunch (FAQ 2026-07-24). Only fighters qualify — all
        other aircraft cannot be regenerated once Winchester.
        """
        player = self.state.player(side)
        out: list[int] = []
        for t in player.living_tokens():
            if t.score_type != TokenScoreType.FIGHTER:
                continue
            if not (t.grounded and t.winchester):
                continue
            squad = player.squadrons.get(t.source_card_id) if t.source_card_id is not None else None
            if squad is not None and not squad.is_destroyed and self._squadron_on_board(squad):
                out.append(t.uid)
        return out

    def relaunch_fighter(self, side: Side, token_uid: int) -> int:
        """
        Attempt to relaunch a Winchester fighter that returned to a live squadron
        (FAQ 2026-07-24). Rolls a D4 and returns the roll:
          - roll of 1: the launch fails and the fighter is "broken" — destroyed
            on the ground and surrendered to the opponent for points.
          - roll of 2-4: the fighter relaunches as a fresh sortie — rearmed
            (Winchester cleared), no longer acquired (face-down again), at the
            side's front band.
        A relaunch is the player's turn action: mutually exclusive with the
        Move-Acquire-Shoot cycle, and it ends the turn like activating a squadron.
        A fighter may attempt a relaunch every time it returns Winchester.
        """
        player = self._require_turn(side)
        if player.has_moved or player.has_acquired or player.has_shot:
            raise IllegalAction(
                "Relaunch and Move-Acquire-Shoot are mutually exclusive this turn")
        token = self._own_token(player, token_uid)
        if token.uid not in self.relaunch_candidates(side):
            raise IllegalAction(
                "Only a Winchester fighter that returned to a live squadron may relaunch")

        roll = self._d4()
        if roll == 1:
            # "Broken": to the opponent's graveyard for points, destroyed on the
            # ground (so PRC Counter-Intervention scores it as a ground air kill).
            self.state.destroy_token(token.uid, destroyed_by=self.state.opponent(side).side,
                                     on_ground=True)
        else:
            # Relaunch: a fresh sortie — rearmed (Winchester cleared), no longer
            # acquired (flips back face-down, so the enemy must re-acquire it),
            # and back at the front band.
            squad = player.squadrons.get(token.source_card_id)
            if squad is not None and token.uid in squad.grounded_token_uids:
                squad.grounded_token_uids.remove(token.uid)
            token.grounded = False
            token.winchester = False
            token.acquired = False
            token.location = self._any_on_map_band(side)   # front band

        self.state.record_action(side)
        self.end_turn(side)          # a relaunch attempt ends the turn
        return roll

    def _default_spawn_band(self, side: Side, legal: frozenset) -> BandID:
        """Deterministic first-band pick for squadrons that allow a choice."""
        first = board.own_airbase(side)  # not a spawn band, but anchors ordering
        # Prefer the side's front band; fall back to any legal band.
        front = BandID.BAND_A if side == Side.US else BandID.BAND_E
        if front in legal:
            return front
        return sorted(legal, key=lambda b: b.name)[0]

    # =====================================================================
    # Action: move
    # =====================================================================

    def move(self, side: Side, token_uid: int, dest_band: BandID) -> None:
        """
        Move one token up to its Move Range. Move-Acquire-Shoot may be taken in
        ANY order within a turn, each action at most once (user ruling
        2026-07-31, reverting the 2026-07-13 forward-order ruling to the Player
        Guide's "Actions may be in any order"). Ships move like any token, up to
        their Move Range of 1 among the on-map range bands (user ruling
        2026-07-31, reverting the 2026-07-14 "ships cannot move" ruling); their
        Winchester off-board relocation at start of turn stays free movement.
        """
        player = self._require_turn(side)
        if player.has_moved:
            raise IllegalAction("Already moved a token this turn")
        token = self._own_token(player, token_uid)
        if token.grounded:
            raise IllegalAction("Grounded tokens cannot move")
        if token.is_winchester:
            raise IllegalAction("Winchester tokens cannot move (they return to base)")

        profile = token.profile
        home = self._home_base(player, token)
        destinations = board.valid_move_destinations(
            token.location, profile.move_range, profile.movement_bands, home
        )
        if dest_band not in destinations:
            raise IllegalAction(
                f"{token.token_type.value} cannot reach {dest_band.name} from {token.location.name}"
            )
        token.location = dest_band
        player.has_moved = True
        self.state.record_action(side)

        # Landing rule: a squadron token moved back to its base is done for the
        # ATO — it grounds on its card (rearmed by re-activation next cycle).
        # Landing on a destroyed card surrenders the token, as at cycle end.
        if dest_band == home and token.origin == TokenOrigin.SQUADRON_CARD \
                and token.source_card_id is not None:
            squad = player.squadrons.get(token.source_card_id)
            if squad is None or squad.is_destroyed:
                self.state.destroy_token(token.uid,
                                         destroyed_by=self.state.opponent(side).side,
                                         on_ground=True)
            else:
                token.grounded = True
                if token.uid not in squad.grounded_token_uids:
                    squad.grounded_token_uids.append(token.uid)

    # =====================================================================
    # Action: acquire
    # =====================================================================

    def acquire(self, side: Side, acquirer_uid: int, target_uid: int) -> AcquireResult:
        """
        Attempt to acquire an enemy token as a target. Once per turn;
        Move-Acquire-Shoot may be taken in any order (user ruling 2026-07-31).
        """
        player = self._require_turn(side)
        if player.has_acquired:
            raise IllegalAction("Already acquired this turn")
        acquirer = self._own_token(player, acquirer_uid)
        if acquirer.profile.acquire_range is None:
            raise IllegalAction(f"{acquirer.token_type.value} cannot acquire")
        if acquirer.grounded:
            raise IllegalAction("Grounded tokens cannot acquire")

        target = self.state.get_token(target_uid)
        if target is None or not target.is_active or target.side == side:
            raise IllegalAction("Invalid acquisition target")
        if target.acquired:
            raise IllegalAction("Target already acquired")
        if not self._acquirable(target):
            raise IllegalAction("Tokens on the ground cannot be acquired (ADA excepted)")
        if not board.in_range(acquirer.location, target.location, acquirer.profile.acquire_range):
            raise IllegalAction("Target out of acquisition range")

        mode = combine_modes(self._enduring_modes(side, RollContext.ACQUIRE, acquirer))
        bonus = acquirer.profile.acquire_bonus or 0
        rolls: list[RollResult] = []
        success = False
        for _ in range(max(1, acquirer.profile.acquire_rolls)):
            r = self.roll(mode, bonus, note="acquire")
            rolls.append(r)
            if r.value >= target.profile.acquisition_value:
                success = True
                break

        if success:
            target.acquired = True
        player.has_acquired = True
        self.state.record_action(side)
        return AcquireResult(acquirer_uid, target_uid, rolls, success)

    # =====================================================================
    # Action: shoot
    # =====================================================================

    def shoot_air(self, side: Side, attacker_uid: int, target_uid: int,
                  missile_defense_uid: Optional[int] = None) -> ShootResult:
        """
        Air-to-air shot at an acquired enemy air token. Once per turn.

        Missile Defense (FAQ 2026-07-24): an air-to-air attack that crosses the
        WEZ of an enemy air-defense token may be rolled at disadvantage if the
        defender activates it — enemy ADA covering the shot auto-imposes
        disadvantage, and a naval defender may be declared via
        `missile_defense_uid` (spending one air salvo), exactly like a surface
        strike. Air-to-air is a single-hit kill with no damage roll, so the
        disadvantage falls on the hit roll only.
        """
        player = self._require_turn(side)
        if player.has_shot:
            raise IllegalAction("Already shot this turn")
        attacker = self._own_token(player, attacker_uid)
        if attacker.profile.air_atk_range is None:
            raise IllegalAction(f"{attacker.token_type.value} has no air attack")
        if attacker.grounded or not attacker.has_air_capacity():
            raise IllegalAction("Attacker cannot conduct an air attack")

        target = self.state.get_token(target_uid)
        if target is None or not target.is_active or target.side == side:
            raise IllegalAction("Invalid air target")
        if target.is_naval:
            raise IllegalAction("Use a surface attack against ships")
        if target.score_type == TokenScoreType.ADA:
            raise IllegalAction("ADA is destroyed by a surface/base strike, not air-to-air")
        # Standoff aircraft CAN be engaged air-to-air: the standoff container is a
        # 6th band adjacent to the front band (US_STANDOFF~BAND_A,
        # PRC_STANDOFF~BAND_E), so a fighter within air-attack range may shoot IN
        # (user ruling 2026-08-14 — reverses the 2026-08-10 "surface/standoff
        # strike only"). Reach is enforced by the in_range check below: standoff
        # sits one band beyond the front, so an r1 weapon must be in the adjacent
        # band. (Firing OUT of standoff was already range-correct.)
        if not target.acquired:
            raise IllegalAction("Air targets must be acquired before firing")
        if not self._acquirable(target):
            # Even if acquired while airborne: once back on the ground a token
            # dies only with its squadron (user ruling 2026-07-14).
            raise IllegalAction("Tokens on the ground cannot be attacked (ADA excepted)")
        # Munitions Upgrade may extend one squadron's air-to-air range this ATO.
        atk_range = player.air_range_override.get(attacker.source_card_id, attacker.profile.air_atk_range)
        if not board.in_range(attacker.location, target.location, atk_range):
            raise IllegalAction("Target out of air-attack range")

        # Elite Pilots / Special Mission Aircraft: consume a one-shot advantage.
        extra: list[RollMode] = []
        if player.pending_air_advantage:
            extra.append(RollMode.ADVANTAGE)
            player.pending_air_advantage = False
        # Missile Defense: enemy ADA covering the shot auto-imposes disadvantage;
        # a naval defender must be declared (spending an air salvo).
        md_modes, md = self._resolve_missile_defense(
            attacker, target.side, target.location, missile_defense_uid, RollContext.AIR_ATTACK)
        mode = self.attack_mode(attacker, RollContext.AIR_ATTACK, target.side, target.location,
                                extra=extra + md_modes)
        hit_roll = self.roll(mode, note="hit")
        hit = hit_roll.value >= attacker.profile.air_atk_thresh

        result = ShootResult(attacker_uid, RollContext.AIR_ATTACK, hit_roll, hit, 0,
                             missile_defense_uid=md)
        undo = _AttackUndo(attacker_side=side)
        if hit:
            on_ground = target.grounded
            victim_card_id = target.source_card_id
            rec = self.state.destroy_token(target.uid, destroyed_by=side, on_ground=on_ground)
            result.destroyed_token_uids.append(target.uid)
            result.damage = 1
            undo.revived_tokens.append(target)
            undo.token_captures.append(rec)
            # Campaign 4: award the air-unit kill bonus if this shot just wiped
            # the enemy squadron out entirely in the air.
            self._maybe_award_air_unit_kill(side, victim_card_id)
        self._last_attack_undo = undo

        result.attacker_winchester = self._apply_winchester(attacker, hit_roll.natural, salvo="air")
        player.has_shot = True
        self.state.record_action(side)
        return result

    def shoot_surface(self, side: Side, attacker_uid: int, *,
                      target_uid: Optional[int] = None,
                      target_band: Optional[BandID] = None,
                      target_squadron_id: Optional[int] = None,
                      target_vp_boxes: bool = False,
                      damage_allocator=None,
                      missile_defense_uid: Optional[int] = None,
                      defer_allocation: bool = False) -> ShootResult:
        """
        Surface/ground shot. Either targets an enemy surface combatant
        (`target_uid`, must be acquired) or an enemy base (`target_band` — no
        acquisition needed).

        For a base attack the attacker distributes the rolled damage: pass
        `target_squadron_id` to hit a specific Squadron Card first, or
        `target_vp_boxes=True` to fill the airbase's 3 VP damage boxes first (for
        Victory Points). Overflow spills to the other targets at the base.
        Alternatively `defer_allocation=True` rolls the strike but leaves the
        damage unapplied: the result carries a `pending_allocation` the attacker
        must settle via `resolve_base_allocation` before any other turn action
        (the RL env turns each point into an explicit decision).

        Missile Defense: enemy ADA covering the shot's WEZ auto-imposes
        disadvantage; a naval defender must be declared via `missile_defense_uid`
        (consuming one of its air salvos).
        """
        player = self._require_turn(side)
        if player.has_shot:
            raise IllegalAction("Already shot this turn")
        attacker = self._own_token(player, attacker_uid)
        if attacker.profile.surf_atk_range is None:
            raise IllegalAction(f"{attacker.token_type.value} has no surface attack")
        if attacker.grounded:
            raise IllegalAction("Grounded tokens cannot shoot")
        if attacker.is_naval and (attacker.surf_salvos_remaining or 0) <= 0:
            raise IllegalAction("Attacker has no surface salvos remaining")
        elif not attacker.is_naval and attacker.is_winchester:
            raise IllegalAction("Attacker is Winchester")

        if target_uid is not None:
            return self._shoot_surface_token(side, player, attacker, target_uid, missile_defense_uid)
        if target_band is not None:
            return self._shoot_base(side, player, attacker, target_band, target_squadron_id,
                                    target_vp_boxes, damage_allocator, missile_defense_uid,
                                    defer_allocation=defer_allocation)
        raise IllegalAction("Surface shot requires a target token or a target band")

    def _shoot_surface_token(self, side, player, attacker, target_uid, md_uid) -> ShootResult:
        target = self.state.get_token(target_uid)
        if target is None or not target.is_active or target.side == side:
            raise IllegalAction("Invalid surface target")
        if not target.acquired:
            raise IllegalAction("Surface targets must be acquired before firing")
        if not target.is_naval:
            if target.score_type == TokenScoreType.ADA:
                pass  # ground ADA is a valid surface target (single-hit kill below)
            elif target.location in board.STANDOFF_LOCATIONS:
                return self._shoot_standoff_token(side, player, attacker, target, md_uid)
            else:
                raise IllegalAction(
                    "Surface attacks target ships, ADA, bases, or aircraft in a Standoff container"
                )
        if not board.in_range(attacker.location, target.location, attacker.profile.surf_atk_range):
            raise IllegalAction("Target out of surface-attack range")

        md_modes, md = self._resolve_missile_defense(
            attacker, target.side, target.location, md_uid, RollContext.SURF_ATTACK
        )
        mode = self.attack_mode(attacker, RollContext.SURF_ATTACK, target.side, target.location, extra=md_modes)
        auto_hit = player.pending_auto_hit
        if auto_hit:
            player.pending_auto_hit = False
            hit_roll = None
            hit = True
        else:
            hit_roll = self.roll(mode, note="hit")
            hit = hit_roll.value >= attacker.profile.surf_atk_thresh

        result = ShootResult(attacker_uid=attacker.uid, context=RollContext.SURF_ATTACK,
                             hit_roll=hit_roll, hit=hit, damage=0, missile_defense_uid=md)
        undo = _AttackUndo(attacker_side=side)
        if hit:
            result.damage, _ = self._damage_roll(
                attacker, self.damage_mode(attacker, RollContext.SURF_ATTACK, extra=md_modes))
            # A ship (or ground ADA) is a single-hit kill regardless of damage
            # magnitude; ADA is surrendered as an on-ground kill.
            on_ground = target.score_type == TokenScoreType.ADA
            rec = self.state.destroy_token(target.uid, destroyed_by=side, on_ground=on_ground)
            result.destroyed_token_uids.append(target.uid)
            undo.revived_tokens.append(target)
            undo.token_captures.append(rec)
        self._last_attack_undo = undo

        # Winchester is set by the HIT roll ONLY (user ruling 2026-08-06). An
        # auto-hit strike (Forward Observers) has no hit roll and counts as a
        # clean hit — equivalent to a natural 4 — so it never causes Winchester
        # (and a ship spends no salvo). The damage roll never affects ammo.
        if hit_roll is not None:
            result.attacker_winchester = self._apply_winchester(attacker, hit_roll.natural, salvo="surf")
        player.has_shot = True
        self.state.record_action(side)
        return result

    def _shoot_standoff_token(self, side, player, attacker, target, md_uid) -> ShootResult:
        """
        Surface attack against an aircraft in a Standoff container (walkthrough
        turns 8/10: a J-15 ground-attacks the B-52 in the US standoff). Damage
        lands on the target's Squadron Card; only that squadron may be damaged
        by the attack. Destroying the card also destroys the squadron's
        aircraft in the container. A token with no live squadron card behind it
        is killed by the hit directly.
        """
        if not board.in_range(attacker.location, target.location, attacker.profile.surf_atk_range):
            raise IllegalAction("Target out of surface-attack range")

        defender = self.state.player(target.side)
        squad = None
        if target.origin == TokenOrigin.SQUADRON_CARD and target.source_card_id is not None:
            cand = defender.squadrons.get(target.source_card_id)
            if cand is not None and not cand.is_destroyed:
                squad = cand

        md_modes, md = self._resolve_missile_defense(
            attacker, target.side, target.location, md_uid, RollContext.SURF_ATTACK
        )
        mode = self.attack_mode(attacker, RollContext.SURF_ATTACK, target.side,
                                target.location, extra=md_modes)
        auto_hit = player.pending_auto_hit
        if auto_hit:
            player.pending_auto_hit = False
            hit_roll = None
            hit = True
        else:
            hit_roll = self.roll(mode, note="hit")
            hit = hit_roll.value >= attacker.profile.surf_atk_thresh

        result = ShootResult(attacker_uid=attacker.uid, context=RollContext.SURF_ATTACK,
                             hit_roll=hit_roll, hit=hit, damage=0, missile_defense_uid=md)
        undo = _AttackUndo(attacker_side=side)
        if hit:
            amount, _ = self._damage_roll(
                attacker, self.damage_mode(attacker, RollContext.SURF_ATTACK, extra=md_modes))
            result.damage = amount
            if squad is None:
                rec = self.state.destroy_token(target.uid, destroyed_by=side, on_ground=False)
                result.destroyed_token_uids.append(target.uid)
                undo.revived_tokens.append(target)
                undo.token_captures.append(rec)
            else:
                su = _SquadronDamageUndo(defender_side=target.side, card_id=squad.card_id,
                                         prev_damage=squad.damage, prev_zone=squad.zone)
                applied = max(0, min(DAMAGE_TO_DESTROY_SQUADRON - squad.damage, amount))
                squad.damage += applied
                if squad.is_destroyed:
                    su.destroyed = True
                    su.revived_tokens = [t for uid in list(squad.grounded_token_uids)
                                         if (t := self.state.get_token(uid)) is not None]
                    recs = self.state.destroy_squadron(target.side, squad.card_id,
                                                       destroyed_by=side, on_ground=True)
                    su.token_captures = [r for r in recs if not r.is_squadron_card]
                    su.card_capture = next((r for r in recs if r.is_squadron_card), None)
                    su.phantom_lost = sum(1 for r in su.token_captures if r.uid is None)
                    result.destroyed_squadron_ids.append(squad.card_id)
                    # Aircraft in the container go down with their squadron.
                    for tok in list(defender.living_tokens()):
                        if tok.source_card_id == squad.card_id \
                                and tok.location in board.STANDOFF_LOCATIONS:
                            rec = self.state.destroy_token(tok.uid, destroyed_by=side,
                                                           on_ground=False)
                            result.destroyed_token_uids.append(tok.uid)
                            su.revived_tokens.append(tok)
                            if rec is not None:
                                su.token_captures.append(rec)
                undo.squad_undos.append(su)
        self._last_attack_undo = undo

        # Winchester is set by the HIT roll ONLY (user ruling 2026-08-06): an
        # auto-hit strike never causes Winchester — see the surface/base paths.
        if hit_roll is not None:
            result.attacker_winchester = self._apply_winchester(attacker, hit_roll.natural, salvo="surf")
        player.has_shot = True
        self.state.record_action(side)
        return result

    @staticmethod
    def _squadron_on_board(squad) -> bool:
        """
        A Squadron Card is on the board (and so can take base-strike damage) only
        while it is FIELDED this ATO cycle — deployed face-down (SELECTED) or
        activated (ACTIVE). A card that was on the board in an earlier cycle but
        is not selected this cycle sits in DECK; it keeps its `location` and card
        damage but is off the board, so it cannot take new damage (ruling
        2026-07-22).
        """
        return squad.zone in (CardZone.SELECTED, CardZone.ACTIVE)

    def _base_allocation_targets(self, target_side: Side, target_band: BandID) -> list[tuple]:
        """
        Everything base-strike damage may be allocated to at `target_band`.

        Ground attacks target Squadron CARDS, not their grounded flight tokens
        (those die with the card). Individually-targetable ground tokens are
        cardless (enabler-generated) NON-ADA units, plus any ACQUIRED ADA on the
        band (user ruling 2026-08-10, reversing the 2026-07-24 FAQ exclusion: a
        base strike may hit ADA sitting on the base once it has been acquired —
        an unacquired ADA is still invisible to the strike), plus the Infantry
        Battalion marker and the airbase VP boxes.
        """
        defender = self.state.player(target_side)
        targets = [("squadron", s.card_id) for s in defender.squadrons.values()
                   if self._squadron_on_board(s) and s.location == target_band]
        targets += [("token", t.uid) for t in defender.tokens_at(target_band)
                    if (t.origin != TokenOrigin.SQUADRON_CARD
                        and t.score_type != TokenScoreType.ADA)
                    or (t.score_type == TokenScoreType.ADA and t.acquired)]
        if target_band in defender.infantry_battalions:
            targets.append(("infantry", None))
        if target_band == board.own_airbase(target_side):
            targets.append(("vp", None))
        return targets

    def resolve_base_allocation(self, allocation: list[tuple]) -> ShootResult:
        """
        Apply the attacker's damage distribution for the deferred base strike.
        `allocation` is a list of (kind, id, points) as in `_apply_base_damage`.
        Returns the (now completed) ShootResult of the original strike.
        """
        pending = self._pending_allocation
        if pending is None:
            raise IllegalAction("No base strike awaiting damage allocation")
        self._pending_allocation = None
        pending.result.pending_allocation = None
        self._apply_base_damage(pending.target_side, pending.target_band, pending.amount,
                                pending.attacker_side, None, pending.result, pending.undo,
                                allocation=allocation)
        return pending.result

    def _shoot_base(self, side, player, attacker, target_band, target_squadron_id,
                    target_vp_boxes, damage_allocator, md_uid,
                    defer_allocation: bool = False) -> ShootResult:
        target_side = Side.PRC if board.is_own_base(target_band, Side.PRC) else Side.US
        if target_side == side:
            raise IllegalAction("Cannot attack your own base")
        if not board.is_enemy_base(target_band, side):
            raise IllegalAction(f"{target_band.name} is not an enemy base")
        if not board.in_range(attacker.location, target_band, attacker.profile.surf_atk_range):
            raise IllegalAction("Base out of surface-attack range")

        md_modes, md = self._resolve_missile_defense(
            attacker, target_side, target_band, md_uid, RollContext.BASE_ATTACK
        )
        mode = self.attack_mode(attacker, RollContext.BASE_ATTACK, target_side, target_band, extra=md_modes)
        auto_hit = player.pending_auto_hit
        if auto_hit:
            player.pending_auto_hit = False
            hit_roll = None
            hit = True
        else:
            hit_roll = self.roll(mode, note="hit")
            hit = hit_roll.value >= attacker.profile.surf_atk_thresh

        result = ShootResult(attacker_uid=attacker.uid, context=RollContext.BASE_ATTACK,
                             hit_roll=hit_roll, hit=hit, damage=0, missile_defense_uid=md)
        undo = _AttackUndo(attacker_side=side)
        if hit:
            amount, _ = self._damage_roll(
                attacker, self.damage_mode(attacker, RollContext.BASE_ATTACK, extra=md_modes))
            result.damage = amount
            targets = self._base_allocation_targets(target_side, target_band)
            if defer_allocation and amount > 0 and targets:
                # Leave the damage unapplied; the attacker settles it through
                # resolve_base_allocation (the undo object is shared, so cancel
                # responses afterwards still see the allocated damage).
                self._pending_allocation = PendingBaseAllocation(
                    attacker_side=side, target_side=target_side, target_band=target_band,
                    amount=amount, targets=targets, result=result, undo=undo)
                result.pending_allocation = self._pending_allocation
            else:
                allocation = damage_allocator(amount, targets) if damage_allocator is not None else None
                self._apply_base_damage(target_side, target_band, amount, side, target_squadron_id,
                                        result, undo, target_vp=target_vp_boxes, allocation=allocation)
        self._last_attack_undo = undo

        # Winchester is set by the HIT roll ONLY (user ruling 2026-08-06). An
        # auto-hit strike (Forward Observers) has no hit roll and counts as a
        # clean hit — equivalent to a natural 4 — so it never causes Winchester
        # (and a ship spends no salvo). The damage roll never affects ammo.
        if hit_roll is not None:
            result.attacker_winchester = self._apply_winchester(attacker, hit_roll.natural, salvo="surf")
        player.has_shot = True
        self.state.record_action(side)
        return result

    # -- damage helpers -----------------------------------------------------

    def _damage_roll(self, attacker: TokenInstance, mode: RollMode) -> tuple[int, Optional[int]]:
        """
        Damage inflicted by a successful surface hit, as (amount, natural roll).
        Exploding-die tokens roll a D4 for the amount; others do a flat 1 point
        (natural None — no die was rolled).

        Advantage/disadvantage carries over from the attack: the rulebook applies
        it to the damage roll whenever Missile Defense is employed or an enabler
        specifies it (Badger Surge, Improved Munitions, EW cards, ACE posture,
        EC-130). Since every advantage/disadvantage source here is exactly one of
        those, the damage roll uses the same net `mode` as the hit roll.
        """
        if not attacker.profile.surf_exploding_die:
            return 1, None
        r = self.roll(mode, note="damage")
        return r.value, r.natural

    def _apply_base_damage(self, target_side: Side, base_band: BandID, amount: int,
                           attacker_side: Side, target_squadron_id: Optional[int],
                           result: ShootResult, undo: Optional["_AttackUndo"] = None,
                           target_vp: bool = False,
                           allocation: Optional[list[tuple]] = None) -> None:
        """
        Distribute base damage the way the attacker chooses.

        `allocation` (preferred) is an explicit list of (kind, id, points):
        kind "squadron" (with card_id), "token" (with uid — tokens on a base
        may be engaged without acquisition; one hit destroys), "infantry" (the
        Infantry Battalion marker protecting the base — "This card may be a
        target of a base attack"), or "vp". This models the rulebook's free
        distribution of uncancelled exploding-die damage across the airbase.
        Without it, the default fills the chosen squadron / VP boxes first and
        spills over.

        Contingency Location rule: only ONE Squadron Card (and its assets) may
        be targeted per attack, so any allocation there is limited to a single
        card and that card's tokens (no VP boxes exist at a CL).

        When `undo` is supplied, records enough to reverse the damage for a
        cancel-response card (Red Horse / Resilient Bases).
        """
        defender = self.state.player(target_side)
        is_airbase = base_band == board.own_airbase(target_side)
        squads_here = {
            s.card_id: s for s in defender.squadrons.values()
            if self._squadron_on_board(s) and s.location == base_band
        }
        # Grounded flight tokens are not separately targetable — hits go on
        # their Squadron Card. Individually-targetable are cardless
        # (enabler-generated) NON-ADA tokens, plus any ACQUIRED ADA on the base
        # (user ruling 2026-08-10; an unacquired ADA is not a strike target).
        tokens_here = {t.uid: t for t in defender.tokens_at(base_band)
                       if (t.origin != TokenOrigin.SQUADRON_CARD
                           and t.score_type != TokenScoreType.ADA)
                       or (t.score_type == TokenScoreType.ADA and t.acquired)}

        def hit_squad(card_id: Optional[int], points: int) -> int:
            squad = squads_here.get(card_id)
            if squad is None or points <= 0:
                return 0
            applied = min(DAMAGE_TO_DESTROY_SQUADRON - squad.damage, points)
            if applied <= 0:
                return 0
            su = _SquadronDamageUndo(defender_side=target_side, card_id=squad.card_id,
                                     prev_damage=squad.damage, prev_zone=squad.zone)
            squad.damage += applied
            if squad.is_destroyed:
                su.destroyed = True
                su.revived_tokens = [t for uid in list(squad.grounded_token_uids)
                                     if (t := self.state.get_token(uid)) is not None]
                recs = self.state.destroy_squadron(target_side, squad.card_id,
                                                   destroyed_by=attacker_side, on_ground=True)
                su.token_captures = [r for r in recs if not r.is_squadron_card]
                su.card_capture = next((r for r in recs if r.is_squadron_card), None)
                su.phantom_lost = sum(1 for r in su.token_captures if r.uid is None)
                if squad.card_id not in result.destroyed_squadron_ids:
                    result.destroyed_squadron_ids.append(squad.card_id)
            if undo is not None:
                undo.squad_undos.append(su)
            return applied

        def hit_token(uid: Optional[int], points: int) -> int:
            tok = tokens_here.get(uid)
            # A token may already have gone down with its squadron card.
            if tok is None or tok.destroyed or points <= 0:
                return 0
            rec = self.state.destroy_token(uid, destroyed_by=attacker_side, on_ground=True)
            result.destroyed_token_uids.append(uid)
            tokens_here.pop(uid, None)
            if undo is not None:
                undo.revived_tokens.append(tok)
                undo.token_captures.append(rec)
            return 1

        def hit_infantry(points: int) -> int:
            # The card has two printed damage boxes; it is destroyed (and stops
            # shielding the base) only when both are filled.
            taken = defender.infantry_battalions.get(base_band)
            if points <= 0 or taken is None:
                return 0
            applied = min(INFANTRY_DAMAGE_BOXES - taken, points)
            if applied <= 0:
                return 0
            if undo is not None:
                undo.infantry_damage = (target_side, base_band, taken)
            if taken + applied >= INFANTRY_DAMAGE_BOXES:
                defender.infantry_battalions.pop(base_band, None)
                result.infantry_destroyed = True
            else:
                defender.infantry_battalions[base_band] = taken + applied
            return applied

        def hit_vp(points: int) -> int:
            if points <= 0 or not is_airbase:
                return 0
            applied = min(AIRBASE_BONUS_DAMAGE_BOXES - defender.airbase_vp_damage, points)
            if applied <= 0:
                return 0
            defender.airbase_vp_damage += applied
            result.base_vp_damage += applied
            if undo is not None:
                prev = undo.base_vp[1] if undo.base_vp else 0
                undo.base_vp = (target_side, prev + applied)
            return applied

        # Build a default allocation if the caller didn't specify one.
        if allocation is None:
            allocation = []
            if target_vp:
                allocation.append(("vp", None, amount))
            if target_squadron_id in squads_here:
                allocation.append(("squadron", target_squadron_id, amount))
            for cid in squads_here:
                allocation.append(("squadron", cid, amount))
            for uid in list(tokens_here):
                allocation.append(("token", uid, amount))
            if base_band in defender.infantry_battalions:
                allocation.append(("infantry", None, amount))
            if not target_vp:
                allocation.append(("vp", None, amount))

        # Contingency Location: NO spillover — only the first-targeted entry
        # (normally the specifically targeted Squadron Card) takes hits;
        # excess damage is lost.
        if base_band == BandID.US_CONTINGENCY_LOCATION and allocation:
            first = allocation[0][:2]
            allocation = [a for a in allocation if tuple(a[:2]) == tuple(first)]

        dispatch = {"squadron": hit_squad, "token": hit_token,
                    "infantry": lambda _id, pts: hit_infantry(pts),
                    "vp": lambda _id, pts: hit_vp(pts)}
        remaining = amount
        for kind, ref_id, points in allocation:
            if remaining <= 0:
                break
            take = min(points, remaining)
            handler = dispatch.get(kind)
            done = handler(ref_id, take) if handler else 0
            remaining -= done

    # -- missile defense ----------------------------------------------------

    def _resolve_missile_defense(self, attacker: TokenInstance, target_side: Side,
                                 target_band: BandID, md_uid: Optional[int],
                                 context: RollContext) -> tuple[list[RollMode], Optional[int]]:
        """
        Determine Missile-Defense disadvantage for a surface/base attack.

        - Any enemy ADA whose WEZ covers the shot path auto-imposes disadvantage
          (no cost, no declaration).
        - A declared naval defender (`md_uid`) must have air capacity, cover the
          WEZ, and pay one air salvo. Returns (extra_modes, defender_uid_used).
        """
        defender = self.state.player(target_side)

        def covers(tok: TokenInstance) -> bool:
            wez = tok.profile.air_atk_range  # WEZ range under the missile icon
            if wez is None:
                return False
            return board.in_wez(tok.location, attacker.location, attacker.side, target_band, wez)

        # ADA: automatic disadvantage within WEZ.
        for tok in defender.living_tokens():
            if tok.score_type == TokenScoreType.ADA and tok.profile.has_missile_defense and covers(tok):
                return [RollMode.DISADVANTAGE], None

        # Declared naval missile defense.
        if md_uid is not None:
            modes = self._declared_naval_md(md_uid, target_side, attacker.location,
                                            attacker.side, target_band)
            return modes, md_uid

        return [], None

    def _declared_naval_md(self, md_uid: int, target_side: Side, attacker_band: BandID,
                           attacker_side: Side, target_band: BandID) -> list[RollMode]:
        """
        Resolve a declared naval missile defense: validate the defender, spend
        one air salvo (the Winchester marker goes on when *declaring*), and
        return the disadvantage — unless the declaration was cancelled (Decoy
        Warheads / Air Launched Decoy), in which case the salvo is still spent
        but no disadvantage applies.
        """
        md = self.state.get_token(md_uid)
        if md is None or md.side != target_side or not md.profile.has_missile_defense:
            raise IllegalAction("Invalid missile-defense token")
        if not md.is_naval:
            raise IllegalAction("Only naval (or ADA) tokens provide missile defense here")
        if (md.air_salvos_remaining or 0) <= 0:
            raise IllegalAction("Missile-defense ship has no air salvos remaining")
        wez = md.profile.air_atk_range
        if wez is None or not board.in_wez(md.location, attacker_band, attacker_side,
                                           target_band, wez):
            raise IllegalAction("Attack does not cross the defender's WEZ")
        md.air_salvos_remaining -= 1  # cover one gray air-salvo block
        if self._md_cancelled_next:
            self._md_cancelled_next = False
            return []
        return [RollMode.DISADVANTAGE]

    # -- winchester ---------------------------------------------------------

    def _apply_winchester(self, attacker: TokenInstance, natural_roll: int, salvo: str) -> bool:
        """
        Update the attacker's ammunition after a shot. Returns True if the token
        is now fully Winchester.

        - Naval: a shot covers one salvo block of the relevant magazine unless the
          natural roll was a 4 (which preserves the weapon).
        - Air/UAS/bomber: below the Winchester threshold (4, or 3 for Attack UAS)
          the token is Winchester. Tokens with no threshold (ADA/AEW/recon/EC-130)
          never go Winchester.
        """
        if attacker.is_naval:
            if natural_roll != DIE_SIDES:
                if salvo == "air" and attacker.air_salvos_remaining is not None:
                    attacker.air_salvos_remaining = max(0, attacker.air_salvos_remaining - 1)
                elif salvo == "surf" and attacker.surf_salvos_remaining is not None:
                    attacker.surf_salvos_remaining = max(0, attacker.surf_salvos_remaining - 1)
            return attacker.is_winchester

        threshold = attacker.profile.winchester_threshold
        if threshold is None:
            return False
        if natural_roll < threshold:
            attacker.winchester = True
        return attacker.winchester

    # =====================================================================
    # Action: play enabler (dispatches to enablers.ENABLER_HANDLERS)
    # =====================================================================

    def play_enabler(self, side: Side, card_id: int,
                     play: Optional[EnablerPlay] = None,
                     response: bool = False) -> EnablerResult:
        """
        Play an Enabler Card from hand. Validates hand membership, zone, and
        response timing, then dispatches to the card's handler in
        `enablers.ENABLER_HANDLERS`. Handles the common tail: enduring flag,
        zone transition, and play-logging that scoring depends on.

        `play` carries the card's parameters (targets, spawn band, "OR" choice).
        Returns an `EnablerResult` describing what the card did.
        """
        player = self.state.player(side)
        if not response:
            self._require_turn(side)
            if player.has_played_enabler:
                raise IllegalAction("Only one Enabler Card may be played per turn")

        card = player.enablers.get(card_id)
        if card is None or card.zone != CardZone.SELECTED:
            raise IllegalAction(f"{side.value} does not have enabler {card_id} in hand")

        profile = ENABLER_REGISTRY[card_id]
        if response and not profile.is_response:
            raise IllegalAction(f"Enabler {profile.name} cannot be played as a response")
        if not response and not profile.is_not_response:
            raise IllegalAction(f"Enabler {profile.name} may only be played as a response")

        play = play or EnablerPlay()
        if card_id in SUBMARINE_STRIKE_CARDS and play.choice == "cancel" and not response:
            # The CANCEL branch exists only as a response to the opponent's
            # just-played submarine card (user ruling 2026-07-14).
            raise IllegalAction(
                f"Enabler {profile.name}: the CANCEL branch may only be played as a "
                f"response to the opponent's submarine card")
        self._resolving_card_id = card_id
        handler = ENABLER_HANDLERS.get(card_id)
        try:
            result = handler(self, side, play) if handler else EnablerResult(card_id=card_id)
        finally:
            self._resolving_card_id = None

        # Common tail: enduring cards stay ACTIVE for the ATO cycle; others spend.
        if profile.is_enduring:
            card.zone = CardZone.ACTIVE
            card.enduring = True
        else:
            card.zone = CardZone.PLAYED

        player.enablers_played_log.append(card_id)
        self._last_played_card = (side, card_id, result)
        if not response:
            player.has_played_enabler = True
            self.state.record_action(side)
        return result

    def _any_on_map_band(self, side: Side) -> BandID:
        return BandID.BAND_A if side == Side.US else BandID.BAND_E

    # =====================================================================
    # Enabler-effect primitives (called by enablers.py handlers)
    # =====================================================================

    def _generate_enabler_tokens(self, side: Side, card_id: int,
                                 spawn_band: Optional[BandID]) -> list[int]:
        """Spawn an enabler card's token(s); returns their uids. ENABLER_CARD origin."""
        profile = ENABLER_REGISTRY[card_id]
        tok_type = profile.generates_token
        if tok_type is None:
            return []
        player = self.state.player(side)
        legal = board.valid_spawn_locations(tok_type, side, player.posture_type or PostureType.STANDARD)
        if player.cl_banned_campaign:
            legal = legal - {BandID.US_CONTINGENCY_LOCATION}
        band = spawn_band if spawn_band is not None else (
            self._default_spawn_band(side, legal) if legal else self._any_on_map_band(side)
        )
        if legal and band not in legal:
            raise IllegalAction(f"{tok_type.value} may not spawn at {band.name}")
        count = TOKEN_REGISTRY[tok_type].token_count
        uids: list[int] = []
        for _ in range(count):
            tok = self.state.spawn_token(
                side, tok_type, band, TokenOrigin.ENABLER_CARD, source_card_id=card_id
            )
            uids.append(tok.uid)
        player.enabler_tokens_generated_this_turn += count
        return uids

    def _acquire_n(self, side: Side, n: int, target_uids: Optional[list[int]] = None) -> list[int]:
        """
        Acquire up to `n` enemy tokens (specific uids if given, else by uid
        order). Grounded tokens cannot be acquired (ADA excepted).
        """
        acquired: list[int] = []
        if target_uids:
            for uid in target_uids:
                if len(acquired) >= n:
                    break
                t = self.state.get_token(uid)
                if t is not None and t.is_active and t.side != side and not t.acquired \
                        and self._acquirable(t):
                    t.acquired = True
                    acquired.append(uid)
        else:
            for t in self.state.opponent(side).living_tokens():
                if len(acquired) >= n:
                    break
                if not t.acquired and self._acquirable(t):
                    t.acquired = True
                    acquired.append(t.uid)
        return acquired

    def _remove_tokens(self, side: Side, n: int, target_uids: Optional[list[int]] = None,
                       score_type: Optional[TokenScoreType] = None) -> list[int]:
        """
        Destroy up to `n` enemy tokens (optionally filtered by score type).
        Grounded tokens are immune — they can only be destroyed with their
        squadron, ADA excepted (user ruling 2026-07-14). Records an attack-undo
        so the defender's response cards (Personnel Recovery / Quick-Turn
        Mobility) can recover a lost aircraft.
        """
        removed: list[int] = []
        undo = _AttackUndo(attacker_side=side)
        if target_uids:
            pool = [self.state.get_token(u) for u in target_uids]
        else:
            pool = list(self.state.opponent(side).living_tokens())
        for t in pool:
            if len(removed) >= n:
                break
            if t is None or not t.is_active or t.side == side:
                continue
            if score_type is not None and t.score_type != score_type:
                continue
            if not self._acquirable(t):
                continue
            rec = self.state.destroy_token(t.uid, destroyed_by=side, on_ground=t.grounded)
            removed.append(t.uid)
            undo.revived_tokens.append(t)
            undo.token_captures.append(rec)
        self._last_attack_undo = undo
        return removed

    def _degrade_cyber(self, target_side: Side, n: int) -> int:
        """Reduce a side's Cyber Rate by up to `n`; returns the actual reduction."""
        player = self.state.player(target_side)
        before = player.cyber_rate
        self.state.set_cyber_rate(target_side, before - n)
        return before - player.cyber_rate

    def _base_has_missile_defense(self, target_side: Side, base_band: BandID) -> bool:
        """
        True if a generated enemy ADA token's WEZ covers the base (automatic
        disadvantage). Naval missile defense is never automatic — it must be
        declared, spending an air salvo.
        """
        for tok in self.state.player(target_side).living_tokens():
            wez = tok.profile.air_atk_range
            if tok.score_type == TokenScoreType.ADA and tok.profile.has_missile_defense \
                    and wez is not None and board.in_range(tok.location, base_band, wez):
                return True
        return False

    def _strike_base(self, side: Side, target_band: BandID, *,
                     roll_gate: Optional[int] = None, bypass_md: bool = True,
                     fixed_damage: Optional[int] = None,
                     target_squadron_id: Optional[int] = None,
                     sof: bool = False,
                     missile_defense_uid: Optional[int] = None,
                     defer_allocation: bool = False,
                     airbase_only: bool = False) -> int:
        """
        Enabler base strike. Optional `roll_gate` (must roll >= gate to land),
        then damage = `fixed_damage` or a D4. Unless `bypass_md` (Unblockable /
        SOF direct-action cards), enemy ADA covering the base auto-imposes
        disadvantage and the defender may declare naval missile defense via
        `missile_defense_uid` (spending an air salvo). For an Enabler Card,
        advantage/disadvantage (MD, EW cards, posture biases) applies ONLY to the
        hit-gate roll; the damage roll is always a single die (FAQ 2026-07-24 —
        supersedes the 2026-07-14 "both rolls" ruling for enabler strikes).
        Returns damage dealt. Cancellable.

        `sof` marks a Special-Operations direct-action strike: if the target base
        holds an Infantry Battalion (Enabler 40), its Squadron Cards are immune,
        so the damage is applied to the airbase VP boxes instead.

        `airbase_only` marks a card whose text names the "Airbase" (HIMARS,
        Tomahawk, Sea Dragons, PLANMC — see AIRBASE_ONLY_STRIKE_CARDS): per the
        FAQ, the damage fills ONLY the three airbase VP damage boxes and any
        excess is lost — it never touches the Squadron Cards deployed there. (A
        SOF card that is also airbase-only still lets a present Infantry
        Battalion soak the raid first; only when no battalion is there does the
        damage land on the VP boxes.)
        """
        target_side = self.state.opponent(side).side
        enabler_modes = self._enduring_modes(side, RollContext.BASE_ATTACK, None)
        posture_modes = self._posture_modes(side, RollContext.BASE_ATTACK, target_side, target_band)
        md_modes: list[RollMode] = []
        if not bypass_md:
            if self._base_has_missile_defense(target_side, target_band):
                md_modes.append(RollMode.DISADVANTAGE)
            elif missile_defense_uid is not None:
                # The strike originates from the attacker's own base area.
                md_modes += self._declared_naval_md(missile_defense_uid, target_side,
                                                    board.own_airbase(side), side, target_band)
        mode = combine_modes(enabler_modes + posture_modes + md_modes)

        if roll_gate is not None and self.roll(mode, note="hit").value < roll_gate:
            self._last_attack_undo = _AttackUndo(attacker_side=side)
            return 0
        if fixed_damage is not None:
            amount = fixed_damage
        else:
            # FAQ 2026-07-24: for an Enabler Card, advantage/disadvantage applies
            # ONLY to the hit (gate) roll — the damage roll is always a single die
            # (no MD/EW/posture modifiers). (Token exploding-die strikes keep the
            # rulebook's disadvantage-on-both under Missile Defense; that path is
            # `_damage_roll`, unaffected.)
            amount = self.roll(RollMode.NORMAL, note="damage").value

        result = ShootResult(attacker_uid=-1, context=RollContext.BASE_ATTACK,
                             hit_roll=None, hit=True, damage=amount)
        undo = _AttackUndo(attacker_side=side)
        allocation = None
        if sof and target_band in self.state.player(target_side).infantry_battalions:
            # "Squadrons on that base may not be attacked by SOF." The Infantry
            # Battalion is the only thing a SOF attack can damage here, and it
            # soaks at most its remaining boxes — the excess is lost rather than
            # spilling onto the squadrons it is shielding.
            allocation = [("infantry", None, amount)]
        elif airbase_only:
            # "Airbase"-worded strike: damage fills only the 3 VP boxes; any
            # excess is lost (hit_vp caps at the remaining boxes). Squadron Cards
            # at the base are never touched (FAQ: Airbase = the installation).
            allocation = [("vp", None, amount)]

        targets = self._base_allocation_targets(target_side, target_band)
        if defer_allocation and allocation is None and amount > 0 and targets:
            # Same free distribution a token base strike gets: leave the damage
            # unapplied so the attacker can spend it point by point through
            # resolve_base_allocation (user ruling 2026-07-17). A forced SOF /
            # Infantry Battalion allocation is never deferred — it has no choice.
            self._pending_allocation = PendingBaseAllocation(
                attacker_side=side, target_side=target_side, target_band=target_band,
                amount=amount, targets=targets, result=result, undo=undo)
            result.pending_allocation = self._pending_allocation
        else:
            self._apply_base_damage(target_side, target_band, amount, side,
                                    target_squadron_id, result, undo, allocation=allocation)
        self._last_attack_undo = undo
        return amount

    def _destroy_ship(self, side: Side, target_uid: Optional[int] = None,
                      bypass_md: bool = True) -> Optional[int]:
        """Auto-destroy an enemy surface combatant (target uid or first available)."""
        opp = self.state.opponent(side)
        if target_uid is not None:
            target = self.state.get_token(target_uid)
        else:
            ships = opp.naval_tokens()
            target = ships[0] if ships else None
        if target is None or not target.is_active or target.side == side or not target.is_naval:
            return None
        undo = _AttackUndo(attacker_side=side)
        rec = self.state.destroy_token(target.uid, destroyed_by=side, on_ground=False)
        undo.revived_tokens.append(target)
        undo.token_captures.append(rec)
        self._last_attack_undo = undo
        return target.uid

    def _recover_aircraft(self, side: Side, token_uid: Optional[int] = None) -> Optional[int]:
        """
        Recover a just-lost friendly aircraft to the front band (does not count as
        destroyed). Reverses the relevant part of the last attack.
        """
        undo = self._last_attack_undo
        if undo is None:
            return None
        front = self._any_on_map_band(side)
        for token, capture in zip(list(undo.revived_tokens), list(undo.token_captures)):
            if token.side != side:
                continue
            if token_uid is not None and token.uid != token_uid:
                continue
            self._revive_token(token)
            token.location = front
            token.grounded = False
            token.winchester = False
            # A recovered sortie re-enters the fight FRESH and face-down — even if
            # the jet was acquired before it was shot down, the enemy must
            # re-acquire it (user bug 2026-08-06). (Cancel-attack cards, which undo
            # a shot as if it never happened, keep prior acquisition — that is the
            # separate `_revive_token`/`_cancel_last_attack` path, unchanged.)
            token.acquired = False
            if capture is not None:
                attacker = self.state.opponent(side)
                if capture in attacker.captures:
                    attacker.captures.remove(capture)
            undo.revived_tokens.remove(token)
            undo.token_captures.remove(capture)
            return token.uid
        return None

    def _recover_card(self, side: Side, card_id: Optional[int], *,
                      to_deck: bool = False, require_plassf: bool = False,
                      allow_squadron: bool = False) -> bool:
        """
        Recover a spent card.

        - Rapid Resupply (18): "Recover any discarded Squadron or Enabler Card"
          — an enabler returns to hand; a destroyed Squadron Card is revived
          (returned to its owner and no longer counted during scoring).
        - Constellation Reconstitution (91): a spent PLASSF enabler returns to
          the deck, available for the *next* ATO cycle (`to_deck=True`).
        """
        if card_id is None:
            return False
        player = self.state.player(side)
        card = player.enablers.get(card_id)
        if card is not None and card.zone in (CardZone.DISCARDED, CardZone.REMOVED, CardZone.PLAYED):
            if require_plassf and not ENABLER_REGISTRY[card_id].plassf:
                return False
            self._last_card_recovery = _CardRecoveryUndo(
                side=side, card_id=card_id, is_squadron=False,
                prev_zone=card.zone, prev_enduring=card.enduring)
            card.zone = CardZone.DECK if to_deck else CardZone.SELECTED
            card.enduring = False
            return True

        if allow_squadron:
            squad = player.squadrons.get(card_id)
            if squad is not None and squad.is_destroyed:
                undo = _CardRecoveryUndo(
                    side=side, card_id=card_id, is_squadron=True,
                    prev_zone=squad.zone, prev_damage=squad.damage,
                    prev_activated=squad.activated,
                    prev_grounded=list(squad.grounded_token_uids),
                    prev_location=squad.location, prev_tokens_lost=squad.tokens_lost,
                    prev_recovered=squad.recovered)
                squad.zone = CardZone.SELECTED
                squad.damage = 0
                squad.activated = False
                squad.grounded_token_uids = []
                # It returns to the airbase (FAQ) and is flagged recovered, so its
                # reactivation this ATO rolls a D4 (risky — see activate_squadron).
                squad.location = board.own_airbase(side)
                squad.recovered = True
                # A revived Squadron Card no longer counts as destroyed: it
                # returns full strength — the tokens are placed on top like the
                # start of the ATO — and the opponent loses ALL points for it,
                # the card AND every token, whether it went down in the air or on
                # the ground. Points only count if the squadron is destroyed a
                # SECOND time (FAQ 2026-07-24). Squadron and enabler card_ids
                # never overlap, so matching on card_id touches only this
                # squadron's captures.
                opp = self.state.opponent(side)
                for cap in [c for c in opp.captures if c.card_id == card_id]:
                    opp.captures.remove(cap)
                    undo.removed_captures.append(cap)
                squad.tokens_lost = 0     # re-activation fields the full complement
                self._last_card_recovery = undo
                return True
        return False

    def acquirable_enemy_tokens(self, side: Side) -> list[int]:
        """
        Enemy tokens `side` could acquire right now: on the board, not already
        acquired, and not grounded (ADA excepted). Cyber Reconnaissance lets the
        owner choose from these.
        """
        return sorted(t.uid for t in self.state.opponent(side).living_tokens()
                      if not t.acquired and self._acquirable(t))

    def rapid_resupply_targets(self, side: Side) -> list[int]:
        """
        Cards Rapid Resupply (18) may recover: this side's destroyed Squadron
        Cards and its discarded / spent Enabler Cards ("Recover any discarded
        Squadron or Enabler Card").
        """
        player = self.state.player(side)
        cards = [cid for cid, squad in player.squadrons.items() if squad.is_destroyed]
        cards += [cid for cid, card in player.enablers.items()
                  if card.zone in (CardZone.DISCARDED, CardZone.REMOVED, CardZone.PLAYED)
                  and cid != 18]          # never recover Rapid Resupply with itself
        return sorted(cards)

    def reconstitutable_plassf(self, side: Side) -> list[int]:
        """
        Spent PLASSF enabler cards Constellation Reconstitution (91) may return
        to the deck for the next ATO cycle: this side's PLASSF cards currently in
        a spent zone (discarded / removed / played). The player picks one.
        """
        player = self.state.player(side)
        return sorted(
            cid for cid, card in player.enablers.items()
            if cid != 91 and ENABLER_REGISTRY[cid].plassf
            and card.zone in (CardZone.DISCARDED, CardZone.REMOVED, CardZone.PLAYED)
        )

    def recoverable_aircraft(self, side: Side) -> list[int]:
        """
        This side's aircraft just lost in the most recent attack, which
        Personnel Recovery (12) / Quick-Turn Mobility (17) may bring back
        ("Recover aircraft in US band 1. It does not count as destroyed.").
        """
        undo = self._last_attack_undo
        if undo is None:
            return []
        return [t.uid for t in undo.revived_tokens if t.side == side]

    def flying_crew_chief_candidates(self, side: Side) -> list[int]:
        """
        Squadron Cards Flying Crew Chief (20) may name: this side's squadrons
        deployed at a Contingency Location that have not activated yet this
        cycle — the card makes exactly one of them generate max aircraft with
        no D4 roll, so an already-activated or destroyed card is not a target.
        """
        player = self.state.player(side)
        return sorted(
            cid for cid, squad in player.squadrons.items()
            if squad.location == BandID.US_CONTINGENCY_LOCATION
            and not squad.is_destroyed and not squad.activated
        )

    def munitions_upgrade_candidates(self, side: Side) -> list[int]:
        """
        Squadron Cards Munitions Upgrade (74) may name: this side's own fighter
        squadrons that are not destroyed — one gets air-to-air range 4 for the
        ATO cycle. The player chooses which (the script raises a SQUADRON_PICK).
        """
        player = self.state.player(side)
        return sorted(
            cid for cid, squad in player.squadrons.items()
            if not squad.is_destroyed
            and TOKEN_REGISTRY[squad.token_type].token_score_type == TokenScoreType.FIGHTER
        )

    def aerial_refuel_candidates(self, side: Side) -> list[int]:
        """
        Squadron Cards Aerial Refueling (16 US / 66 PRC) may place beyond the
        posture limit: this side's campaign-legal cards not already in play.
        The player chooses one (the script raises a SQUADRON_PICK node).
        """
        player = self.state.player(side)
        return [cid for cid in sorted(SQUADRON_REGISTRY)
                if SQUADRON_REGISTRY[cid].side == side
                and cid not in player.squadrons
                and self.campaign.squadron_allowed(cid)]

    def _add_squadron(self, side: Side, card_id: Optional[int]) -> bool:
        """
        Deploy an extra Squadron Card beyond the posture limit (Aerial Refueling).
        The card must belong to the side, be campaign-legal, and not already be in
        play. It is placed on the airbase ready to activate.
        """
        if card_id is None:
            return False
        player = self.state.player(side)
        if card_id in player.squadrons:
            return False
        sp = SQUADRON_REGISTRY.get(card_id)
        if sp is None or sp.side != side or not self.campaign.squadron_allowed(card_id):
            return False
        player.squadrons[card_id] = SquadronState(
            card_id=card_id, side=side, zone=CardZone.SELECTED, location=board.own_airbase(side)
        )
        return True

    def reserves_eligible(self, side: Side) -> list[int]:
        """
        Squadron Cards `side` may regenerate with Reserves: those the opponent
        has taken every token of (user ruling 2026-07-14 — Reserves plays only
        when a squadron has lost ALL its tokens). A squadron qualifies whether it
        was emptied of tokens in the air OR had its card destroyed outright by a
        base strike (user ruling 2026-07-22): in both cases the whole squadron is
        gone and Reserves brings it back.
        """
        player = self.state.player(side)
        out: list[int] = []
        for cid, squad in player.squadrons.items():
            if squad.tokens_lost <= 0:
                continue
            # Must have been in play: launched (activated) or destroyed outright
            # by a base strike. An untouched face-down card is not a target.
            if not squad.activated and not squad.is_destroyed:
                continue
            # Any of its tokens still flying → it has not lost ALL of them yet.
            if any(t.origin == TokenOrigin.SQUADRON_CARD and t.source_card_id == cid
                   for t in player.tokens.values() if t.is_active):
                continue
            out.append(cid)
        return out

    def reserves_playable(self, side: Side) -> bool:
        """
        True only in the window right after the opponent destroyed the last of
        a squadron's tokens: the most recent attack must have killed a token
        of a now-empty squadron (user ruling 2026-07-14 — Reserves may only be
        played immediately after that event).
        """
        undo = self._last_attack_undo
        if undo is None:
            return False
        eligible = set(self.reserves_eligible(side))
        if any(t.side == side and t.origin == TokenOrigin.SQUADRON_CARD
               and t.source_card_id in eligible
               for t in undo.revived_tokens):
            return True
        # A base strike that just destroyed the squadron's card also opens the
        # window: the card and all of its grounded / ungenerated tokens went
        # down together, so the squadron has lost everything (ruling 2026-07-22).
        return any(su.defender_side == side and su.destroyed and su.card_id in eligible
                   for su in undo.squad_undos)

    def _regenerate_squadron(self, side: Side, card_id: Optional[int]) -> bool:
        """
        Fully regenerate a squadron whose tokens were all destroyed (Reserves).

        "The squadron is fully regenerated" restores the entire complement, so
        the opponent no longer profits from having destroyed it: every capture
        scored for this squadron is undone — the Squadron Card itself and all of
        its tokens, whether they went down in the air or on the ground (user
        ruling 2026-07-22). Also revives a card destroyed outright by a base
        strike (damage / DESTROYED zone are cleared).
        """
        if card_id is None or card_id not in self.reserves_eligible(side):
            return False
        squad = self.state.player(side).squadrons[card_id]
        squad.activated = False
        squad.zone = CardZone.SELECTED
        squad.damage = 0                      # un-destroy a base-struck card
        squad.grounded_token_uids = []
        squad.tokens_lost = 0                 # re-activation fields the full complement
        squad.location = squad.location or board.own_airbase(side)
        # Strip the opponent's captures for this now-regenerated squadron (card
        # + all its tokens). Squadron and enabler card_ids never overlap, so
        # matching on card_id touches only this squadron's captures.
        opp = self.state.opponent(side)
        opp.captures[:] = [c for c in opp.captures if c.card_id != card_id]
        return True

    def _force_discard(self, target_side: Side, n: int,
                       card_ids: Optional[list[int]] = None) -> list[int]:
        """
        Force a side to discard up to `n` enablers from hand. `card_ids` are the
        cards the VICTIM chose to give up (their own hand, so the choice is
        theirs); anything not in hand is ignored and the remainder is taken in
        hand order so the effect still lands in full.
        """
        player = self.state.player(target_side)
        gone: list[int] = []
        chosen = [c for c in (card_ids or [])
                  if any(h.card_id == c for h in player.enablers_in_hand())]
        for cid in chosen:
            if len(gone) >= n:
                break
            player.enablers[cid].zone = CardZone.DISCARDED
            gone.append(cid)
        for card in player.enablers_in_hand():
            if len(gone) >= n:
                break
            card.zone = CardZone.DISCARDED
            gone.append(card.card_id)
        return gone

    def enemy_ship_targets(self, side: Side) -> list[int]:
        """
        Enemy surface combatants on the board — the choices for the auto-hit
        ship-kill cards (Submarine Strike 43/89, Marine Littoral 44, Maritime
        Cruise 77). No acquisition is needed, so every enemy ship is eligible;
        callers must still label them fog-safely.
        """
        return sorted(t.uid for t in self.state.opponent(side).naval_tokens())

    def infantry_battalion_bases(self, side: Side) -> list[BandID]:
        """
        Bases the Infantry Battalion (40) may be placed on: "any base or
        contingency location" belonging to `side`, excluding one that already
        has a battalion (and the CL while a SURGE ban is in force).
        """
        player = self.state.player(side)
        bands = [b for b in sorted(board.own_base_locations(side), key=lambda b: b.name)
                 if b not in player.infantry_battalions]
        if player.cl_banned_campaign:
            bands = [b for b in bands if b != BandID.US_CONTINGENCY_LOCATION]
        return bands

    def removable_enemy_tokens(self, side: Side) -> list[int]:
        """
        Enemy tokens Offensive Cyber may destroy: on the board and not grounded
        (grounded tokens die only with their squadron, ADA excepted).
        """
        return sorted(t.uid for t in self.state.opponent(side).living_tokens()
                      if self._acquirable(t))

    def removable_enemy_uas(self, side: Side) -> list[int]:
        """
        Enemy UAS tokens Cyber Counter-UAS (25/84) may destroy: on the board,
        not grounded, and of UAS score type — the owner picks which die.
        """
        return sorted(t.uid for t in self.state.opponent(side).living_tokens()
                      if self._acquirable(t) and t.score_type == TokenScoreType.UAS)

    def _revive_token(self, token: TokenInstance) -> None:
        """Return a destroyed token to play, refunding its squadron's loss."""
        token.destroyed = False
        owner = self.state.player(token.side)
        owner.tokens[token.uid] = token
        if token.origin == TokenOrigin.SQUADRON_CARD and token.source_card_id is not None:
            squad = owner.squadrons.get(token.source_card_id)
            if squad is not None and squad.tokens_lost > 0:
                squad.tokens_lost -= 1

    def _cancel_last_attack(self, defender_side: Side, damage_only: bool = False) -> bool:
        """
        Reverse the most recent attack: revive destroyed tokens, restore squadron
        damage / destroyed squadrons, and undo airbase VP damage. Removes the
        corresponding capture records from the attacker. One-shot.
        """
        undo = self._last_attack_undo
        if undo is None:
            return False
        attacker = self.state.player(undo.attacker_side)

        # Capture the POST-attack state first, so a cancel-base-damage card that
        # reverses this attack can itself be cancelled (re-applying the damage).
        self._last_attack_reapply = self._capture_attack_reapply(undo)

        # Revive directly-destroyed tokens (air/surface hits).
        for token, capture in zip(undo.revived_tokens, undo.token_captures):
            self._revive_token(token)
            if capture is not None and capture in attacker.captures:
                attacker.captures.remove(capture)

        # Restore squadron damage / destroyed squadrons and their tokens.
        for su in undo.squad_undos:
            squad = self.state.player(su.defender_side).squadrons.get(su.card_id)
            if squad is None:
                continue
            squad.damage = su.prev_damage
            squad.zone = su.prev_zone
            if su.destroyed:
                for token in su.revived_tokens:
                    self._revive_token(token)
                    if token.uid not in squad.grounded_token_uids:
                        squad.grounded_token_uids.append(token.uid)
                for cap in su.token_captures:
                    if cap in attacker.captures:
                        attacker.captures.remove(cap)
                if su.card_capture is not None and su.card_capture in attacker.captures:
                    attacker.captures.remove(su.card_capture)
                # Never-generated tokens have no instance to revive; just undo
                # the phantom loss so the restored squadron regenerates in full.
                squad.tokens_lost = max(0, squad.tokens_lost - su.phantom_lost)

        # Undo airbase VP damage.
        if undo.base_vp is not None:
            vp_side, vp_hits = undo.base_vp
            defender = self.state.player(vp_side)
            defender.airbase_vp_damage = max(0, defender.airbase_vp_damage - vp_hits)
            # Never leave more boxes "already paid for" than are still damaged.
            defender.airbase_vp_scored = min(defender.airbase_vp_scored,
                                             defender.airbase_vp_damage)

        # Restore the Infantry Battalion's damage (reviving it if it was killed).
        if undo.infantry_damage is not None:
            inf_side, inf_band, prev = undo.infantry_damage
            self.state.player(inf_side).infantry_battalions[inf_band] = prev

        # Void any un-allocated deferred base-strike damage: a hit cancelled
        # (e.g. Air Launched Decoy) BEFORE the attacker allocates leaves a
        # pending allocation that must not still be spent.
        self._pending_allocation = None
        self._last_attack_undo = None
        return True

    def _capture_attack_reapply(self, undo: "_AttackUndo") -> "_AttackReapply":
        """Snapshot the post-attack state `undo` is about to reverse, so the
        attack can be re-applied if the reversing cancel-card is itself
        cancelled (see `_reapply_attack`)."""
        r = _AttackReapply(attacker_side=undo.attacker_side)
        for token, capture in zip(undo.revived_tokens, undo.token_captures):
            r.dead_tokens.append(token)
            if capture is not None:
                r.captures.append(capture)
        for su in undo.squad_undos:
            squad = self.state.player(su.defender_side).squadrons.get(su.card_id)
            if squad is None:
                continue
            r.squads.append((su.defender_side, su.card_id, squad.damage, squad.zone,
                             list(squad.grounded_token_uids), squad.tokens_lost))
            if su.destroyed:
                r.dead_tokens.extend(su.revived_tokens)
                r.captures.extend(c for c in su.token_captures if c is not None)
                if su.card_capture is not None:
                    r.captures.append(su.card_capture)
        if undo.base_vp is not None:
            vp_side = undo.base_vp[0]
            d = self.state.player(vp_side)
            r.base_vp = (vp_side, d.airbase_vp_damage, d.airbase_vp_scored)
        if undo.infantry_damage is not None:
            inf_side, inf_band, _ = undo.infantry_damage
            r.infantry = (inf_side, inf_band,
                          self.state.player(inf_side).infantry_battalions.get(inf_band, 0))
        return r

    def _reapply_attack(self, r: "_AttackReapply") -> None:
        """Re-apply an attack captured by `_capture_attack_reapply` (reverses a
        Red Horse / Resilient Bases cancel that is itself cancelled)."""
        attacker = self.state.player(r.attacker_side)
        for token in r.dead_tokens:
            token.destroyed = True
            self.state.player(token.side).tokens.pop(token.uid, None)
        for defender_side, card_id, damage, zone, grounded, tokens_lost in r.squads:
            squad = self.state.player(defender_side).squadrons.get(card_id)
            if squad is None:
                continue
            squad.damage = damage
            squad.zone = zone
            squad.grounded_token_uids = list(grounded)
            squad.tokens_lost = tokens_lost
        # Re-append every removed capture: never-generated captures are
        # value-identical (uid=None) frozen records, so an `in`/dedup test would
        # collapse them — the cancel removed them one-for-one, so restore in kind.
        attacker.captures.extend(r.captures)
        if r.base_vp is not None:
            vp_side, damage, scored = r.base_vp
            d = self.state.player(vp_side)
            d.airbase_vp_damage = damage
            d.airbase_vp_scored = scored
        if r.infantry is not None:
            inf_side, inf_band, damage = r.infantry
            self.state.player(inf_side).infantry_battalions[inf_band] = damage

    def _undo_card_recovery(self, side: Side) -> None:
        """Reverse the most recent Rapid Resupply recovery (its card is being
        cancelled): put the recovered card back where it was."""
        rec = self._last_card_recovery
        if rec is None or rec.side != side:
            return
        player = self.state.player(side)
        if rec.is_squadron:
            squad = player.squadrons.get(rec.card_id)
            if squad is not None:
                squad.zone = rec.prev_zone
                squad.damage = rec.prev_damage
                squad.activated = rec.prev_activated
                squad.grounded_token_uids = list(rec.prev_grounded)
                squad.location = rec.prev_location
                squad.tokens_lost = rec.prev_tokens_lost
                squad.recovered = rec.prev_recovered
            # Re-append every removed capture unconditionally: never-generated
            # captures are value-identical (uid=None), so a dedup test would drop
            # duplicates that were genuinely removed one-for-one.
            self.state.opponent(side).captures.extend(rec.removed_captures)
        else:
            card = player.enablers.get(rec.card_id)
            if card is not None:
                card.zone = rec.prev_zone
                card.enduring = rec.prev_enduring
        self._last_card_recovery = None

    def _cancel_last_card(self, allowed=None, canceller: Optional[Side] = None) -> bool:
        """
        Void the opponent's most recently played card: destroy any tokens it
        generated, clear an enduring effect, and drop it from the play-log so it
        no longer scores. Dice effects already resolved are not rewound.

        `allowed` is an optional predicate over the cancelled card's
        EnablerProfile — a cancel card may only void the card type its own text
        names (SOF / Space / Cyber / mobility-maintenance / submarine).
        `canceller` (when given) requires the voided card to be the OPPONENT's:
        a player can never cancel their own play.
        """
        if self._last_played_card is None:
            return False
        side, card_id, result = self._last_played_card
        if canceller is not None and side == canceller:
            return False
        if allowed is not None and not allowed(ENABLER_REGISTRY[card_id]):
            return False
        player = self.state.player(side)

        # A cancelled card never happened: reverse the effects it applied.
        # (Fixes AC-130 / Counter Space / etc. "cancelling" a card while its
        # strike damage, kills, acquisitions and cyber gain all stood.)
        if result.base_damage or result.destroyed:
            # Rewinds squadron damage/kills, revived tokens, captures and
            # airbase VP boxes recorded when the card's strike resolved.
            self._cancel_last_attack(board.opponent(side))
        # Cancelling a cyber cancel-card re-applies the acquisition it had voided
        # (PRC acquire → US Defensive Cyber cancels → PRC Defensive Cyber cancels
        # THAT → the tokens are acquired after all). Mirrors the base-damage
        # re-apply for Red Horse / Resilient Bases.
        for uid in result.reacquire_on_cancel:
            token = self.state.get_token(uid)
            if token is not None and token.is_active:
                token.acquired = True
        # Record what THIS cancel un-acquires, so if the canceller (a Defensive
        # Cyber) is itself cancelled the acquisition can be restored above.
        self._reversed_acquired = list(result.acquired)
        for uid in result.acquired:
            token = self.state.get_token(uid)
            if token is not None:
                token.acquired = False
        if result.cyber_delta and result.cyber_side is not None:
            target = self.state.player(result.cyber_side)
            self.state.set_cyber_rate(result.cyber_side,
                                      target.cyber_rate - result.cyber_delta)
            # An instant cyber win that is being cancelled no longer stands.
            if self.state.winner == result.cyber_side \
                    and target.cyber_rate < CYBER_RATE_WIN:
                self.state.winner = None
                self.state.game_over = False

        for uid in result.tokens:
            self.state.destroy_token(uid)
        # Reverse "recovered" placements too: a cancel must undo the card's
        # EFFECT, not merely un-log it (Anti-Access/Area Denial cancelling a US
        # mobility/maintenance card left the recovered aircraft / added squadron
        # standing). The meaning of `recovered` is card-specific.
        for rec in result.recovered:
            self._reverse_recovery(side, card_id, rec)
        # Cancelling a cancel-base-damage card (Red Horse / Resilient Bases)
        # re-applies the base damage it had voided.
        if card_id in CANCEL_BASE_DAMAGE_CARDS and self._last_attack_reapply is not None:
            self._reapply_attack(self._last_attack_reapply)
            self._last_attack_reapply = None
        card = player.enablers.get(card_id)
        if card is not None:
            card.enduring = False
        if card_id in player.enablers_played_log:
            player.enablers_played_log.remove(card_id)

        self._last_played_card = None
        return True

    def _reverse_recovery(self, side: Side, card_id: int, rec: Optional[int]) -> None:
        """
        Undo one entry of a cancelled card's `recovered` effect (for a cancel
        card such as Anti-Access/Area Denial). `side` is the side that played the
        cancelled card; `rec` is the recovered unit/card id, whose meaning
        depends on `card_id`.
        """
        if rec is None:
            return
        if card_id in RECOVER_AIRCRAFT_CARDS:
            # 12/17: `rec` is the revived aircraft — send it back down, so the
            # opponent regains the kill (it is airborne now: a fresh air loss).
            tok = self.state.get_token(rec)
            if tok is not None and not tok.destroyed and tok.side == side:
                self.state.destroy_token(rec, destroyed_by=board.opponent(side),
                                         on_ground=False)
        elif card_id in AERIAL_REFUEL_CARDS:
            # 16/66: `rec` is the extra Squadron Card placed beyond the posture
            # limit — remove it (it cannot have activated in the response window)
            # and free the slot it claimed.
            player = self.state.player(side)
            squad = player.squadrons.get(rec)
            if squad is not None and not squad.activated:
                player.squadrons.pop(rec, None)
            player.extra_squadron_slots = max(0, player.extra_squadron_slots - 1)
        elif card_id in FLYING_CREW_CHIEF_CARDS:
            # 20: `rec` is the Contingency-Location squadron named for free max
            # generation — clear the pending flag.
            player = self.state.player(side)
            if player.cl_max_squadron_id == rec:
                player.cl_max_squadron_id = None
        elif card_id in RAPID_RESUPPLY_CARDS:
            # 18: put the recovered Squadron / Enabler Card back where it was,
            # restoring the opponent's captures for a re-lost squadron.
            self._undo_card_recovery(side)

    # =====================================================================
    # End of ATO cycle
    # =====================================================================

    def end_ato_cycle(self) -> None:
        """
        Resolve end/cleanup: score end-of-ATO missions (board still populated),
        then clear the board and recycle cards for the next cycle. Advances to the
        next ATO or ends the game after the final cycle.
        """
        self.state.phase = Phase.END_CLEANUP

        # All airborne squadron tokens land on their Squadron Card; any that
        # land on a destroyed squadron are destroyed upon landing (and score
        # for the opponent). Must run before scoring so captures count.
        for side in (Side.US, Side.PRC):
            player = self.state.player(side)
            opp = self.state.opponent(side).side
            for token in list(player.living_tokens()):
                if token.origin != TokenOrigin.SQUADRON_CARD or token.source_card_id is None:
                    continue
                squad = player.squadrons.get(token.source_card_id)
                if squad is None or squad.is_destroyed:
                    self.state.destroy_token(token.uid, destroyed_by=opp, on_ground=True)

        for side in (Side.US, Side.PRC):
            self._score_end_of_ato(side)

        for side in (Side.US, Side.PRC):
            self._cleanup_side(side)

        if self.state.is_final_ato():
            self._finalize_game()
        else:
            self.state.ato_cycle += 1
            self.state.turn_number = 0
            self.state.consecutive_passes = 0
            self.state.initiative_holder = None
            self.state.first_player = None
            self.state.phase = Phase.ATO_SETUP

    def _cleanup_side(self, player_side: Side) -> None:
        player = self.state.player(player_side)

        # Remove all tokens from the board (survivors recycle to their cards).
        player.tokens = {}

        # Recycle cards. Across cycles (user ruling 2026-07-31): a squadron's
        # PARTIAL card damage RESETS (a card dinged for one block starts the next
        # ATO fresh) — but a COMPLETELY destroyed card stays out and scored. TOKEN
        # losses still PERSIST: a destroyed token stays destroyed and scored for
        # the opponent, only the squadron's surviving tokens return to its card,
        # so next cycle it re-fields FEWER tokens (token_count - tokens_lost).
        # Placement is re-chosen each cycle at the next draft. Only a specific
        # enabler (Reserves / Rapid Resupply / recovery cards) restores lost tokens.
        for squad in player.squadrons.values():
            if squad.is_destroyed:
                squad.zone = CardZone.DESTROYED
                continue
            squad.zone = CardZone.DECK
            squad.activated = False
            squad.grounded_token_uids = []
            squad.recovered = False        # the risky-reactivation flag is per-ATO
            squad.damage = 0               # partial CARD damage resets each ATO
                                           # (token losses, above, still persist)

        for enabler in player.enablers.values():
            profile = ENABLER_REGISTRY[enabler.card_id]
            enabler.enduring = False
            enabler.revealed_to_opponent = False
            if enabler.zone in (CardZone.PLAYED, CardZone.ACTIVE):
                # Single-use cards leave the campaign; multi-use return to deck.
                enabler.zone = CardZone.REMOVED if profile.single_use else CardZone.DECK
            else:
                enabler.zone = CardZone.DECK

        # Reset per-cycle bookkeeping (Cyber/Intel/VP persist). Airbase damage
        # persists too (ruling 2026-07-17) — the boxes are permanent, so they
        # are deliberately NOT cleared here.
        player.posture_card_id = None
        player.enablers_played_log = []
        player.reset_turn()
        player.reset_ato_effects()

    def _finalize_game(self) -> None:
        """
        End of campaign: declare a winner from the full totals. `victory_points`
        keeps only the ACCRUED per-turn/per-ATO points — capture-mission and
        campaign-bonus VP stay derived in `total_victory_points`, which would
        double-count them if they were folded into the stored value here
        (bug found 2026-07-15: post-game reads showed roughly doubled scores
        for capture missions like Attrition).
        """
        self.state.phase = Phase.END_CLEANUP
        # Campaign 5: award +1 VP per intact Squadron Card, once, at the end of
        # the campaign (user ruling 2026-07-22) — a distinct end-of-game award,
        # not a bonus that inflated the running total throughout.
        if self.campaign.intact_squadron_bonus:
            for s in (Side.US, Side.PRC):
                self._log_vp(self.state.player(s), "intact Squadron Cards (Campaign 5)",
                             self._intact_squadron_count(s))
        us_total = self.total_victory_points(Side.US)
        prc_total = self.total_victory_points(Side.PRC)
        self.state.game_over = True
        if us_total != prc_total:
            self.state.winner = Side.US if us_total > prc_total else Side.PRC

    # =====================================================================
    # Scoring
    # =====================================================================

    def _log_vp(self, player: PlayerState, reason: str, points: int) -> None:
        """Accrue VP with an audit-trail entry (feeds the TUI scoring report)."""
        if points:
            player.victory_points += points
            player.vp_log.append((self.state.ato_cycle, reason, points))

    def _score_end_of_turn(self, side: Side) -> None:
        """Enforce Rule of Law (US): +2 per squadron activated, +1 per enabler token."""
        player = self.state.player(side)
        if player.mission and player.mission.mission_type == MissionType.RULE_OF_LAW:
            self._log_vp(player, "Rule of Law: squadron activations",
                         2 * len(player.squadrons_activated_this_turn))
            self._log_vp(player, "Rule of Law: enabler tokens generated",
                         player.enabler_tokens_generated_this_turn)

    def _score_end_of_ato(self, side: Side) -> None:
        """
        End-of-ATO scoring: per-ATO missions (Economy of Force, N-K Dominance,
        Three Dominances), the universal airbase VP-box bonus, and the
        Campaign-4 air-kill bonus. Called before the board is cleared.
        """
        player = self.state.player(side)

        # Airbase VP damage boxes (universal): +1 per box hit on the enemy
        # airbase. The boxes persist across ATO cycles (ruling 2026-07-17), so
        # only boxes not yet paid for score — otherwise every surviving box
        # would score again every cycle.
        defender = self.state.opponent(side)
        new_boxes = max(0, defender.airbase_vp_damage - defender.airbase_vp_scored)
        self._log_vp(player, "enemy airbase VP boxes", new_boxes)
        defender.airbase_vp_scored = defender.airbase_vp_damage

        # (Campaign 4's air-unit kill bonus is awarded LIVE on the killing shot —
        # see `_maybe_award_air_unit_kill`, called from `shoot_air` — not here.)

        mission = player.mission
        if mission is None:
            return
        mt = mission.mission_type

        if mt == MissionType.ECONOMY_OF_FORCE:
            # Only DRAFTED units fielded this ATO count (user ruling 2026-07-22):
            # a Squadron Card ON THE BOARD (zone SELECTED/ACTIVE) that was not
            # activated, plus an unplayed token-deploying Enabler Card still in
            # hand. A squadron set aside this cycle (zone DECK) is not on the
            # board and must not count.
            unactivated = sum(
                1 for s in player.squadrons.values()
                if self._squadron_on_board(s) and not s.activated
            )
            unused_enablers = sum(
                1 for e in player.enablers_in_hand()
                if ENABLER_REGISTRY[e.card_id].generates_token is not None
            )
            self._log_vp(player, "Economy of Force: unused squadrons/enablers",
                         2 * (unactivated + unused_enablers))

        elif mt == MissionType.N_K_DOMINANCE:
            self._log_vp(player, "N-K Dominance: Cyber Rate", 3 * player.cyber_rate)
            # "+1 VP for each EW, Cyber, Space, or SOF card play" — the US SOF
            # cards are filed under AIR_FORCE, so match them by name as well.
            nk_classes = {EnablerClass.ELECTRONIC_WARFARE, EnablerClass.CYBER,
                          EnablerClass.SPACE, EnablerClass.SOF}
            self._log_vp(player, "N-K Dominance: EW/Cyber/Space/SOF plays", sum(
                1 for cid in player.enablers_played_log
                if ENABLER_REGISTRY[cid].enabler_class in nk_classes
                or ENABLER_REGISTRY[cid].name.upper().startswith("SOF")
            ))

        elif mt == MissionType.THREE_DOMINANCES:
            self._log_vp(player, "Three Dominances: Cyber Rate", 2 * player.cyber_rate)
            # +3 per naval/maritime unit ON THE BOARD at end of ATO (fielded this
            # cycle, not destroyed): surface combatants PLUS the carrier-based
            # Shandong J-15 (user ruling 2026-07-22).
            naval_units = sum(1 for t in player.living_tokens()
                              if t.is_naval or t.token_type == TokenType.J_15)
            self._log_vp(player, "Three Dominances: naval units on board", 3 * naval_units)
            # "+1 VP for each Naval Enabler card played" — Maritime cards and
            # the carrier-based Naval Air card (Shandong J-15) both count.
            naval_classes = {EnablerClass.MARITIME, EnablerClass.NAVAL_AIR}
            self._log_vp(player, "Three Dominances: naval enabler plays", sum(
                1 for cid in player.enablers_played_log
                if ENABLER_REGISTRY[cid].enabler_class in naval_classes
            ))
            self._log_vp(player, "Three Dominances: intact squadrons",
                         len(player.intact_squadrons()))

        elif mt == MissionType.COUNTER_INTERVENTION:
            # "+1 VP for each PLARF card played" — accrued per ATO here because
            # enablers_played_log resets at cleanup (previously this was derived
            # live in score_captures and silently vanished after each cycle).
            self._log_vp(player, "Counter-Intervention: PLARF plays", sum(
                1 for cid in player.enablers_played_log
                if ENABLER_REGISTRY[cid].plarf
            ))

    def capture_value(self, side: Side, cap: CaptureRecord) -> int:
        """
        VP one capture is worth under `side`'s Mission Card. 0 for the
        per-turn/per-ATO missions (those accrue into victory_points instead).
        """
        mission = self.state.player(side).mission
        if mission is None:
            return 0
        mt = mission.mission_type

        if mt == MissionType.ATTRITION:
            heavy = {TokenScoreType.SHIP, TokenScoreType.BOMBER,
                     TokenScoreType.ADA, TokenScoreType.AEW}
            if cap.is_squadron_card:
                return 2
            return 3 if cap.score_type in heavy else 1

        if mt == MissionType.INTERDICTION:
            top = {TokenScoreType.BOMBER, TokenScoreType.AEW}
            mid = {TokenScoreType.SHIP, TokenScoreType.ADA}
            if cap.is_squadron_card or cap.score_type in top:
                return 4
            return 3 if cap.score_type in mid else 2

        if mt == MissionType.COUNTER_INTERVENTION:
            air_ground = {TokenScoreType.FIGHTER, TokenScoreType.UAS,
                          TokenScoreType.BOMBER, TokenScoreType.AEW}
            if not cap.is_squadron_card and cap.destroyed_on_ground \
                    and cap.score_type in air_ground:
                return 3
        return 0

    def score_captures(self, side: Side) -> int:
        """
        VP from capture-based missions (Attrition / Interdiction /
        Counter-Intervention), computed from the destroyed-unit log.
        """
        player = self.state.player(side)
        return sum(self.capture_value(side, c) for c in player.captures)

    def total_victory_points(self, side: Side) -> int:
        """
        Full VP total for a side: accrued per-turn/per-ATO points (which now
        include the accrued campaign bonuses — Campaign-4 air-unit kills at end
        of each ATO, Campaign-5 intact squadrons at end of campaign), the
        capture-based mission points, and any base-strike airbase VP not yet
        accrued.

        Airbase VP boxes count LIVE: +1 per box this side has hit on the enemy
        airbase, the instant it lands (so the score, Copy-for-AI export and RL
        obs all reflect a base strike immediately). `_score_end_of_ato` later
        folds those boxes into `victory_points` and marks them scored, so the
        `unscored` term below drops to 0 — no double count.
        """
        player = self.state.player(side)
        defender = self.state.opponent(side)
        unscored_base_vp = max(0, defender.airbase_vp_damage - defender.airbase_vp_scored)
        return player.victory_points + self.score_captures(side) + unscored_base_vp

    def _maybe_award_air_unit_kill(self, side: Side, victim_card_id: Optional[int]) -> None:
        """
        Campaign 4: the instant `side`'s air-to-air killing shot wipes out an
        enemy squadron ENTIRELY in the air — every one of its tokens shot down
        while airborne (`destroyed_on_ground=False`), none lost on the ground,
        none surviving — award +1 VP, capped at +2 per ATO cycle. Called from
        `shoot_air` after a kill; the +1 lands on the last killing shot. A
        squadron taken out any other way (base strike, a token that landed and
        surrendered on the ground) earns nothing.
        """
        if not self.campaign.air_kill_bonus_vp or victim_card_id is None:
            return
        player = self.state.player(side)
        opp = self.state.opponent(side)
        if victim_card_id not in opp.squadrons:
            return
        # Already credited this squadron, or the +2/ATO cap is reached.
        if victim_card_id in player.air_units_killed or len(player.air_units_killed) >= 2:
            return
        # No token of the squadron may survive (airborne or grounded)...
        if any(t.source_card_id == victim_card_id for t in opp.living_tokens()):
            return
        # ...and every one of its losses this ATO must be an air kill.
        caps = [c for c in player.captures
                if c.card_id == victim_card_id and c.ato_cycle == self.state.ato_cycle
                and not c.is_squadron_card]
        if not caps or any(c.destroyed_on_ground for c in caps):
            return
        player.air_units_killed.add(victim_card_id)
        self._log_vp(player, "air-unit kill bonus (Campaign 4)", 1)

    def _intact_squadron_count(self, side: Side) -> int:
        """
        Campaign 5: a side's intact Squadron Cards — every campaign-legal card
        (restricted cards removed at setup) NOT destroyed during the campaign,
        including cards that never saw action. Awarded once at end of campaign.
        """
        player = self.state.player(side)
        destroyed = {cid for cid, s in player.squadrons.items() if s.is_destroyed}
        return sum(
            1 for cid, sp in SQUADRON_REGISTRY.items()
            if sp.side == side and self.campaign.squadron_allowed(cid)
            and cid not in destroyed
        )

    # =====================================================================
    # Legal-action enumeration (CLI harness + future env masking)
    # =====================================================================

    def eligible_missile_defenders(self, defender_side: Side, attacker_location: BandID,
                                   attacker_side: Side, target_band: BandID) -> list[TokenInstance]:
        """
        Naval tokens `defender_side` may declare as missile defense against a
        surface/base strike from `attacker_location` at `target_band`: has MD,
        air salvos remaining, and the shot crosses its WEZ. (ADA coverage is
        automatic and needs no declaration.)
        """
        return [t for t in self.state.player(defender_side).living_tokens()
                if t.is_naval and t.profile.has_missile_defense
                and (t.air_salvos_remaining or 0) > 0
                and t.profile.air_atk_range is not None
                and board.in_wez(t.location, attacker_location, attacker_side,
                                 target_band, t.profile.air_atk_range)]

    def legal_responses(self, side: Side, triggers: set) -> list[int]:
        """
        Response enablers in `side`'s hand whose declared trigger matches one of
        `triggers` (see enablers.card_play_triggers for card-play events).
        Anytime-only cards are not offered here — they play on the owner's turn.
        """
        out: list[int] = []
        for c in self.state.player(side).enablers_in_hand():
            profile = ENABLER_REGISTRY[c.card_id]
            if profile.is_response and set(profile.enabler_trigger or ()) & set(triggers):
                out.append(c.card_id)
        return out

    def legal_actions(self, side: Side) -> list[LegalAction]:
        """
        Enumerate the actions `side` may take right now. Returns an empty list if
        it is not their turn or the game is over. Each action is directly
        executable via `apply_action`; enabler plays are given best-effort default
        parameters (front-band spawn, first eligible targets, default "OR" branch).
        """
        if self.state.game_over or self.state.phase != Phase.PLAYER_TURN or side != self.state.active_side:
            return []

        player = self.state.player(side)
        opp = self.state.opponent(side)
        actions: list[LegalAction] = [LegalAction("pass", label="Pass / end turn")]

        # Activate a squadron (ends the turn). Mutually exclusive with the
        # Move-Acquire-Shoot cycle: not offered once any MAS action was taken.
        if not (player.has_moved or player.has_acquired or player.has_shot):
            for squad in player.squadrons.values():
                if not squad.activated and not squad.is_destroyed and squad.zone == CardZone.SELECTED:
                    actions.append(LegalAction("activate", card_id=squad.card_id,
                                               label=f"Activate {squad.profile.name} ({squad.token_type.value})"))
            # Relaunch a Winchester fighter that returned to a live squadron
            # (FAQ 2026-07-24) — a turn action, like activating; ends the turn.
            for uid in self.relaunch_candidates(side):
                tok = player.tokens[uid]
                actions.append(LegalAction("relaunch", token_uid=uid,
                                           label=f"Relaunch {tok.token_type.value}#{uid} "
                                                 f"(roll D4; 1 = broken)"))

        # Move — one per turn, in ANY order relative to Acquire/Shoot (user
        # ruling 2026-07-31). Ships move too now (Move Range 1, on-map bands;
        # reverting the 2026-07-14 no-move ruling). Tokens of a destroyed squadron
        # keep flying until Winchester (ruling 2026-07-17), so they stay eligible.
        if not player.has_moved:
            for tok in player.living_tokens():
                if tok.grounded or tok.is_winchester:
                    continue
                home = self._home_base(player, tok)
                # Sorted: frozenset iteration order is hash- (process-)
                # dependent for enums, which would break cross-process
                # reproducibility of seeded games.
                for dest in sorted(board.valid_move_destinations(
                    tok.location, tok.profile.move_range, tok.profile.movement_bands, home
                ), key=lambda b: b.name):
                    actions.append(LegalAction("move", token_uid=tok.uid, dest_band=dest,
                                               label=f"Move {tok.token_type.value}#{tok.uid} -> {dest.name}"))

        # Acquire — one per turn, in any order. Grounded enemy tokens cannot be
        # acquired (ADA excepted).
        if not player.has_acquired:
            for acq in player.living_tokens():
                if acq.grounded or acq.profile.acquire_range is None:
                    continue
                for tgt in opp.living_tokens():
                    if not tgt.acquired and self._acquirable(tgt) \
                            and board.in_range(acq.location, tgt.location, acq.profile.acquire_range):
                        # Fog-safe label: an acquire target is unacquired by
                        # definition, so its identity is hidden — show the
                        # physical "roll-to-acquire" back (?#uid(avN)), never the
                        # type, so the choice list can't leak it to the enemy.
                        actions.append(LegalAction("acquire", token_uid=acq.uid, target_uid=tgt.uid,
                                                   label=f"Acquire ?#{tgt.uid}(av{tgt.profile.acquisition_value}) "
                                                         f"with #{acq.uid}"))

        # Shoot.
        if not player.has_shot:
            for atk in player.living_tokens():
                if atk.grounded:
                    continue
                # Air-to-air at acquired enemy aircraft (grounded targets are
                # immune — they die only with their squadron).
                if atk.profile.air_atk_range is not None and atk.has_air_capacity():
                    rng = player.air_range_override.get(atk.source_card_id, atk.profile.air_atk_range)
                    for tgt in opp.living_tokens():
                        # Air-to-air hits enemy AIRCRAFT, including those loitering
                        # in the standoff band — a 6th band reachable by air-to-air
                        # from the adjacent front band (user ruling 2026-08-14,
                        # reverses 2026-08-10). ADA is excluded: a ground asset
                        # killed by a surface/base strike. Reach into standoff is
                        # enforced by in_range (standoff is one band beyond front).
                        if tgt.acquired and self._acquirable(tgt) and not tgt.is_naval \
                                and tgt.score_type != TokenScoreType.ADA \
                                and board.in_range(atk.location, tgt.location, rng):
                            actions.append(LegalAction("shoot_air", token_uid=atk.uid, target_uid=tgt.uid,
                                                       label=f"Air: #{atk.uid} -> {tgt.token_type.value}#{tgt.uid}"))
                # Surface at acquired ships / standoff aircraft / enemy bases.
                surf_ready = (atk.surf_salvos_remaining or 0) > 0 if atk.is_naval else not atk.is_winchester
                if atk.profile.surf_atk_range is not None and surf_ready:
                    for tgt in opp.living_tokens():
                        # Ships, standoff aircraft, and acquired ground ADA — a
                        # surface strike needs surf-attack capability, so an
                        # air-only fighter (F-22) can't, but a multirole (F-35) can.
                        targetable = (tgt.is_naval
                                      or tgt.location in board.STANDOFF_LOCATIONS
                                      or tgt.score_type == TokenScoreType.ADA)
                        if targetable and tgt.acquired and board.in_range(atk.location, tgt.location, atk.profile.surf_atk_range):
                            actions.append(LegalAction("shoot_surface", token_uid=atk.uid, target_uid=tgt.uid,
                                                       label=f"Surface: #{atk.uid} -> {tgt.token_type.value}#{tgt.uid}"))
                    for base in sorted(board.own_base_locations(opp.side), key=lambda b: b.name):
                        if board.in_range(atk.location, base, atk.profile.surf_atk_range):
                            actions.append(LegalAction("shoot_surface", token_uid=atk.uid, target_band=base,
                                                       label=f"Base strike: #{atk.uid} -> {base.name}"))

        # Play an enabler (non-response plays on your own turn) — at most one per turn.
        if not player.has_played_enabler:
            for card in player.enablers_in_hand():
                profile = ENABLER_REGISTRY[card.card_id]
                if not profile.is_not_response:
                    continue
                # Marine Littoral Regiment attacks a surface combatant: only an
                # option while enemy ships are on the board (user ruling 2026-07-14).
                if card.card_id == 44 and not opp.naval_tokens():
                    continue
                actions.append(LegalAction("play_enabler", card_id=card.card_id,
                                           play=self._default_enabler_play(side, card.card_id),
                                           label=f"Play enabler: {profile.name}"))
        return actions

    def _default_enabler_play(self, side: Side, card_id: int) -> EnablerPlay:
        """Best-effort default parameters so an enabler is directly executable."""
        profile = ENABLER_REGISTRY[card_id]
        player = self.state.player(side)
        opp = self.state.opponent(side)

        spawn_band = None
        if profile.generates_token is not None:
            legal = board.valid_spawn_locations(profile.generates_token, side, player.posture_type or PostureType.STANDARD)
            spawn_band = self._default_spawn_band(side, legal) if legal else self._any_on_map_band(side)

        target_uids = [t.uid for t in opp.living_tokens()]
        # Recover-aircraft cards act on the OWNER's just-lost aircraft, not on
        # enemy tokens — handing them the generic (enemy) list made them no-op.
        if card_id in RECOVER_AIRCRAFT_CARDS:
            target_uids = self.recoverable_aircraft(side)
        target_band = board.own_airbase(opp.side)
        # Infantry Battalion is placed on the OWNER's own base — the generic
        # enemy-airbase default silently made the card protect nothing.
        if card_id == 40:
            target_band = next(iter(self.infantry_battalion_bases(side)), target_band)
        # A sensible own-squadron for range-override / infantry-marker cards.
        own_squadron = next((s.card_id for s in player.squadrons.values() if not s.is_destroyed), None)
        # Munitions Upgrade must name one of the player's fighter squadrons.
        if card_id == 74:
            own_squadron = next(
                (s.card_id for s in player.squadrons.values()
                 if not s.is_destroyed
                 and TOKEN_REGISTRY[s.token_type].token_score_type == TokenScoreType.FIGHTER),
                None,
            )
        # Reserves must name a squadron whose tokens were all destroyed.
        if card_id == 67:
            own_squadron = next(iter(self.reserves_eligible(side)), None)
        # Flying Crew Chief only applies to un-activated CL squadrons.
        if card_id == 20:
            own_squadron = next(iter(self.flying_crew_chief_candidates(side)), None)
        # Rapid Resupply recovers a DISCARDED card, not a healthy one.
        if card_id == 18:
            own_squadron = next(iter(self.rapid_resupply_targets(side)), None)
        # Constellation Reconstitution returns a SPENT PLASSF enabler, not a
        # squadron — the generic own-squadron default silently no-oped the card.
        if card_id == 91:
            own_squadron = next(iter(self.reconstitutable_plassf(side)), None)
        # Aerial Refueling wants an *undrafted* legal squadron to add.
        if card_id in (16, 66):
            own_squadron = next(
                (cid for cid, sp in SQUADRON_REGISTRY.items()
                 if sp.side == side and cid not in player.squadrons and self.campaign.squadron_allowed(cid)),
                None,
            )
        return EnablerPlay(spawn_band=spawn_band, target_uids=target_uids,
                           target_band=target_band, target_squadron_id=own_squadron)

    def apply_action(self, action: LegalAction):
        """Execute a LegalAction against the engine and return its result (if any)."""
        side = self.state.active_side
        if action.kind == "pass":
            return self.pass_turn(side)
        if action.kind == "activate":
            return self.activate_squadron(side, action.card_id)
        if action.kind == "relaunch":
            return self.relaunch_fighter(side, action.token_uid)
        if action.kind == "move":
            return self.move(side, action.token_uid, action.dest_band)
        if action.kind == "acquire":
            return self.acquire(side, action.token_uid, action.target_uid)
        if action.kind == "shoot_air":
            return self.shoot_air(side, action.token_uid, action.target_uid)
        if action.kind == "shoot_surface":
            return self.shoot_surface(side, action.token_uid,
                                      target_uid=action.target_uid, target_band=action.target_band)
        if action.kind == "play_enabler":
            return self.play_enabler(side, action.card_id, action.play)
        raise IllegalAction(f"Unknown action kind {action.kind!r}")

    # =====================================================================
    # Small helpers
    # =====================================================================

    def _own_token(self, player: PlayerState, uid: int) -> TokenInstance:
        token = player.tokens.get(uid)
        if token is None or not token.is_active:
            raise IllegalAction(f"{player.side.value} has no active token {uid}")
        return token

    @staticmethod
    def _acquirable(target: TokenInstance) -> bool:
        """
        Grounded tokens cannot be acquired or engaged — once a token returns to
        the ground (Winchester, RTB, or an ungenerated CL token) it can only be
        destroyed with its squadron. ADA is the exception: it is engaged on its
        base (user ruling 2026-07-14).
        """
        return not target.grounded or target.score_type == TokenScoreType.ADA

    def _home_base(self, player: PlayerState, token: TokenInstance) -> BandID:
        """Base a token is tied to (for move legality); its own airbase otherwise."""
        if token.origin == TokenOrigin.SQUADRON_CARD and token.source_card_id is not None:
            squad = player.squadrons.get(token.source_card_id)
            if squad is not None and squad.location is not None:
                return squad.location
        return board.own_airbase(player.side)
