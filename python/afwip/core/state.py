"""
state.py — Mutable game-state layer for AFWIP.

Where the other core modules are *definitions*, this module is the *world*:

  - constants.py  : enums and tuning values (immutable)
  - tokens.py     : TokenProfile registry (immutable capability sheets)
  - cards.py      : Squadron / Enabler / Posture / Mission registries (immutable)
  - board.py      : pure geometry / reachability / WEZ functions (stateless)
  - state.py      : this file — the live, mutable state of a game in progress

The design goal is that a rules/engine layer (and, downstream, a Gymnasium /
PettingZoo environment) reads and mutates a single `GameState` object. State
here stores *facts* — where tokens are, what has been played, how much damage a
squadron has taken, how many VPs a side captured — and deliberately leaves
*rules resolution* (dice, legality, scoring math) to that engine layer. This
keeps state serialisable and side-effect free.

Instances are keyed by stable ids:
  - Tokens by an integer `uid` (unique for the life of the GameState).
  - Cards by their registry `card_id`.

Everything references the registries in tokens.py / cards.py for static
capabilities, so profiles are never duplicated into state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional

from afwip.core.constants import (
    Side,
    Phase,
    BandID,
    TokenType,
    TokenOrigin,
    TokenScoreType,
    PostureType,
    IntelTrack,
    Visibility,
    CYBER_RATE_START,
    CYBER_RATE_WIN,
    CYBER_RATE_MIN,
    MAX_CYBER_RATE,
    DAMAGE_TO_DESTROY_SQUADRON,
    AIRBASE_BONUS_DAMAGE_BOXES,
    CAMPAIGN_ATO_CYCLES,
)
from afwip.core.tokens import TOKEN_REGISTRY, TokenProfile
from afwip.core.cards import (
    SQUADRON_REGISTRY,
    ENABLER_REGISTRY,
    POSTURE_REGISTRY,
    MISSION_REGISTRY,
    SquadronProfile,
    EnablerProfile,
    PostureProfile,
    MissionProfile,
)

# State-management enums (local to state; not core rules constants)

class CardZone(Enum):
    """
    Lifecycle location of a card. Independent of the board — a card is in
    exactly one zone at a time.

      DECK       - owned in the campaign, not drafted for the current ATO cycle
      SELECTED   - drafted for this ATO cycle (enabler set aside; squadron
                   deployed face-down on a base but not yet activated)
      ACTIVE     - squadron activated (face up, tokens generated) OR an
                   enduring enabler currently in effect
      PLAYED     - enabler resolved this ATO cycle (spent, one-shot effect)
      DISCARDED  - returned to discard, potentially recoverable (e.g. Rapid
                   Resupply / Constellation Reconstitution)
      REMOVED    - single-use card out of play for the rest of the campaign
      DESTROYED  - squadron card destroyed by the enemy (out for the campaign)
    """
    DECK = "DECK"
    SELECTED = "SELECTED"
    ACTIVE = "ACTIVE"
    PLAYED = "PLAYED"
    DISCARDED = "DISCARDED"
    REMOVED = "REMOVED"
    DESTROYED = "DESTROYED"


# Tokens on (or off) the board

@dataclass
class TokenInstance:
    """
    A single generated token in play.

    Static capabilities live on the TokenProfile (via `.profile`); this object
    only carries the mutable per-instance facts: where it is, whether it has
    been acquired, its Winchester / salvo status, and whether it is grounded
    (a Contingency Location token that failed to generate).
    """

    uid: int
    token_type: TokenType
    side: Side
    location: BandID
    origin: TokenOrigin

    # Card that produced this token. Squadron card_id for SQUADRON_CARD origin
    # (needed for Winchester return); enabler card_id for ENABLER_CARD origin.
    source_card_id: Optional[int] = None

    # Acquisition — once acquired a token is flipped face-up for the ATO cycle.
    acquired: bool = False

    # Winchester (weapons expended). For air/UAS/bomber a single flag suffices.
    # Naval tokens instead track magazine depth via the salvo counters below and
    # are considered fully Winchester only when both counters reach zero.
    winchester: bool = False

    # Naval magazines — initialised from the profile on spawn. None for tokens
    # that do not use salvo tracking.
    air_salvos_remaining: Optional[int] = None
    surf_salvos_remaining: Optional[int] = None

    # A Contingency Location token that failed its generation roll: it sits on
    # its squadron card, cannot act, and is vulnerable to base attacks.
    grounded: bool = False

    # A Winchester naval token that has moved off the board (free movement). It
    # is not destroyed (does not score) but is no longer on the map.
    off_board: bool = False

    # Set when removed from play. Kept (rather than deleted) so the engine can
    # inspect the cause; GameState.destroy_token drops it from the live board.
    destroyed: bool = False

    @property
    def profile(self) -> TokenProfile:
        """Static capability sheet for this token type."""
        return TOKEN_REGISTRY[self.token_type]

    @property
    def score_type(self) -> TokenScoreType:
        return self.profile.token_score_type

    @property
    def visibility(self) -> Visibility:
        """Opponent-facing visibility: acquired tokens are fully visible."""
        return Visibility.VISIBLE if self.acquired else Visibility.HIDDEN

    @property
    def is_naval(self) -> bool:
        return self.profile.surface_combatant

    @property
    def is_active(self) -> bool:
        """On the board and able to be interacted with."""
        return not self.destroyed

    def has_air_capacity(self) -> bool:
        """
        Whether the token can still fire / provide missile defense with air
        weapons. Naval tokens gate missile defense on remaining air salvos.
        """
        if self.air_salvos_remaining is not None:
            return self.air_salvos_remaining > 0
        return not self.winchester

    @property
    def is_winchester(self) -> bool:
        """True when the token has no remaining weapons of any kind."""
        if self.is_naval:
            air = self.air_salvos_remaining or 0
            surf = self.surf_salvos_remaining or 0
            return air <= 0 and surf <= 0
        return self.winchester


# Squadron cards in play

@dataclass
class SquadronState:
    """
    Runtime state of one drafted Squadron Card.

    A squadron is deployed (SELECTED) face-down on a base, then ACTIVE once
    flipped up and its tokens generated. It carries up to
    DAMAGE_TO_DESTROY_SQUADRON hits before being destroyed; destroying a
    squadron also destroys any of its tokens still attached to the card
    (ungenerated or returned-to-base).
    """

    card_id: int
    side: Side
    zone: CardZone = CardZone.DECK

    # Base where the card is deployed (US_AIRBASE / US_CONTINGENCY_LOCATION /
    # PRC_AIRBASE). None while still in the deck.
    location: Optional[BandID] = None

    activated: bool = False
    damage: int = 0

    # Set the first time the card is activated and never cleared: flipping
    # face-up is public, so the opponent keeps knowing the card's identity in
    # later cycles even though `activated` resets at cleanup.
    ever_activated: bool = False

    # uids of tokens currently attached to (sitting on) this card — i.e. not
    # yet generated onto the map, or returned to base after Winchester.
    grounded_token_uids: list[int] = field(default_factory=list)

    # Tokens of this squadron permanently lost this campaign (destroyed and
    # surrendered to the enemy). Destroyed tokens do NOT regenerate: later
    # activations produce token_count - tokens_lost tokens. Decremented when a
    # loss is reversed (cancel-attack responses, Personnel Recovery).
    tokens_lost: int = 0

    # Set when this card was brought back by Rapid Resupply this ATO. A recovered
    # squadron's reactivation is risky (FAQ 2026-07-24): activating it rolls a D4
    # and on a 1 all its aircraft are "broken". Cleared on (re)activation and at
    # end-of-ATO cleanup.
    recovered: bool = False

    @property
    def profile(self) -> SquadronProfile:
        return SQUADRON_REGISTRY[self.card_id]

    @property
    def token_type(self) -> TokenType:
        return self.profile.token_type

    @property
    def is_destroyed(self) -> bool:
        return self.zone == CardZone.DESTROYED or self.damage >= DAMAGE_TO_DESTROY_SQUADRON


# Enabler cards in play
@dataclass
class EnablerCardState:
    """
    Runtime state of one Enabler Card.

    `enduring` marks a red-pushpin effect that is currently active for the ATO
    cycle. `revealed_to_opponent` is set when the opponent sees this card during
    the Intel phase.
    """

    card_id: int
    side: Side
    zone: CardZone = CardZone.DECK
    revealed_to_opponent: bool = False
    enduring: bool = False

    @property
    def profile(self) -> EnablerProfile:
        return ENABLER_REGISTRY[self.card_id]

# Scoring capture record

@dataclass(frozen=True)
class CaptureRecord:
    """
    An enemy unit destroyed by (and surrendered to) a player. Records the raw
    facts scoring needs; the VP value itself depends on the destroyer's Mission
    Card and so is computed by the scoring layer, not stored here.

    `uid` (token captures) / `card_id` (squadron-card captures) identify the
    exact unit for scoring reports.
    """
    ato_cycle: int
    is_squadron_card: bool
    destroyed_on_ground: bool
    token_type: Optional[TokenType] = None
    score_type: Optional[TokenScoreType] = None
    uid: Optional[int] = None
    card_id: Optional[int] = None

# Per-side state

@dataclass
class PlayerState:
    """Everything belonging to one side (US or PRC)."""

    side: Side

    # Persistent across ATO cycles (kept during cleanup)
    cyber_rate: int = CYBER_RATE_START
    intel_track: IntelTrack = IntelTrack.NORMAL
    victory_points: int = 0
    mission_card_id: Optional[int] = None
    postures_used: set[PostureType] = field(default_factory=set)
    captures: list[CaptureRecord] = field(default_factory=list)
    # SURGE posture: "Contingency Locations cannot be used during this
    # campaign" — the ban outlives the cycle SURGE was selected in.
    cl_banned_campaign: bool = False

    # Current ATO cycle
    posture_card_id: Optional[int] = None
    # Hits on this side's airbase VP damage boxes (0..3). Airbase damage is
    # PERMANENT — it carries across ATO cycles (user ruling 2026-07-17), so the
    # 3 boxes are a campaign-long total, not a per-cycle allowance.
    airbase_vp_damage: int = 0
    # How many of those boxes the opponent has already been paid VP for, so a
    # persisting box is not re-scored every cycle.
    airbase_vp_scored: int = 0

    # Card collections, keyed by card_id
    squadrons: dict[int, SquadronState] = field(default_factory=dict)
    enablers: dict[int, EnablerCardState] = field(default_factory=dict)

    # Tokens in play, keyed by uid
    tokens: dict[int, TokenInstance] = field(default_factory=dict)

    # Per-turn Move-Acquire-Shoot flags (reset each turn)
    has_moved: bool = False
    has_acquired: bool = False
    has_shot: bool = False
    has_played_enabler: bool = False   # at most one enabler may be played per turn
    passed_last_turn: bool = False
    acted_this_turn: bool = False   # any non-pass action (incl. playing an enabler)

    # Per-turn scoring accumulators (Enforce Rule of Law scores per turn)
    squadrons_activated_this_turn: list[int] = field(default_factory=list)
    enabler_tokens_generated_this_turn: int = 0

    # Per-ATO log of enabler card_ids played, in order (feeds N-K Dominance,
    # Three Dominances, Counter-Intervention scoring).
    enablers_played_log: list[int] = field(default_factory=list)

    # Accrued-VP audit trail: (ato_cycle, reason, points) appended whenever
    # victory_points changes (airbase boxes, per-turn/per-ATO mission scoring).
    # Persists across cleanup — feeds the TUI's end-of-ATO scoring report.
    vp_log: list[tuple[int, str, int]] = field(default_factory=list)

    # Pending / transient enabler effects. These persist until consumed by the
    # relevant action (one-shot buffs) or until end-of-ATO cleanup (enduring
    # markers). All are reset in the engine's per-cycle cleanup.
    pending_auto_hit: bool = False              # Forward Observers: next surface strike auto-hits
    pending_air_advantage: bool = False         # Elite Pilots / Special Mission Aircraft
    # Flying Crew Chief (20): the ONE Contingency-Location squadron chosen to
    # generate max aircraft this ATO cycle (no D4 roll). None = card not in play.
    cl_max_squadron_id: Optional[int] = None
    air_range_override: dict[int, int] = field(default_factory=dict)  # Munitions Upgrade: squadron_id -> air range
    # Infantry Battalion (40): band -> damage taken (0..INFANTRY_DAMAGE_BOXES).
    # "Place on any base or contingency location. Squadrons on that base may not
    # be attacked by SOF. This card may be target of a base attack." While the
    # card stands it soaks SOF damage for the squadrons; it is destroyed (and
    # the protection ends) once both of its printed damage boxes are filled.
    infantry_battalions: dict[BandID, int] = field(default_factory=dict)
    extra_squadron_slots: int = 0               # Aerial Refueling: squadrons beyond posture limit
    # Campaign 4: enemy squadron card_ids this side has wiped out ENTIRELY in the
    # air this ATO (each already credited its +1 VP; the +2/ATO cap is len<=2).
    air_units_killed: set[int] = field(default_factory=set)
    # HEDGEHOG posture (US): where to place the free ADA token this ATO — the
    # owner chooses Airbase or Contingency Location. None = the Airbase default.
    posture_bonus_ada_location: Optional[BandID] = None

    # -- profile accessors --------------------------------------------------

    @property
    def posture(self) -> Optional[PostureProfile]:
        if self.posture_card_id is None:
            return None
        return POSTURE_REGISTRY[self.posture_card_id]

    @property
    def posture_type(self) -> Optional[PostureType]:
        posture = self.posture
        return posture.posture_type if posture else None

    @property
    def mission(self) -> Optional[MissionProfile]:
        if self.mission_card_id is None:
            return None
        return MISSION_REGISTRY[self.mission_card_id]

    # -- token queries ------------------------------------------------------

    def living_tokens(self) -> list[TokenInstance]:
        """All of this side's tokens still on the board (excludes off-board naval)."""
        return [t for t in self.tokens.values() if t.is_active and not t.off_board]

    def tokens_at(self, band: BandID) -> list[TokenInstance]:
        return [t for t in self.living_tokens() if t.location == band]

    def tokens_of_score_type(self, score_type: TokenScoreType) -> list[TokenInstance]:
        return [t for t in self.living_tokens() if t.score_type == score_type]

    def naval_tokens(self) -> list[TokenInstance]:
        """Surface combatants on the board (Three Dominances scoring)."""
        return [t for t in self.living_tokens() if t.is_naval]

    # -- card queries -------------------------------------------------------

    def active_squadrons(self) -> list[SquadronState]:
        return [s for s in self.squadrons.values() if s.activated and not s.is_destroyed]

    def intact_squadrons(self) -> list[SquadronState]:
        """
        Intact squadrons for Three Dominances end-of-ATO scoring (user ruling
        2026-07-22): every squadron FIELDED this ATO (on the board — zone
        SELECTED or ACTIVE) that has lost NO tokens this cycle (`tokens_lost ==
        0`). A card set aside this cycle (DECK), a destroyed card, or one that
        lost a token does not count; un-generated Contingency-Location tokens are
        not "destroyed", so a partial CL squadron still counts.
        """
        return [
            squad for squad in self.squadrons.values()
            if squad.zone in (CardZone.SELECTED, CardZone.ACTIVE)
            and squad.tokens_lost == 0
        ]

    def enablers_in_hand(self) -> list[EnablerCardState]:
        return [e for e in self.enablers.values() if e.zone == CardZone.SELECTED]

    def enduring_effects(self) -> list[EnablerCardState]:
        return [e for e in self.enablers.values() if e.enduring]

    # -- per-turn bookkeeping ----------------------------------------------

    def reset_turn(self) -> None:
        """Clear the per-turn action flags and scoring accumulators."""
        self.has_moved = False
        self.has_acquired = False
        self.has_shot = False
        self.has_played_enabler = False
        self.acted_this_turn = False
        self.squadrons_activated_this_turn = []
        self.enabler_tokens_generated_this_turn = 0

    def reset_ato_effects(self) -> None:
        """Clear per-ATO-cycle enabler markers and pending one-shot buffs."""
        self.pending_auto_hit = False
        self.pending_air_advantage = False
        self.cl_max_squadron_id = None
        self.air_range_override = {}
        self.infantry_battalions = {}
        self.extra_squadron_slots = 0
        self.air_units_killed = set()
        self.posture_bonus_ada_location = None

# Top-level game state

@dataclass
class GameState:
    """
    The complete, mutable state of one AFWIP game.

    Construct fresh games with `GameState.new_game(campaign)`. The two
    PlayerState objects are created automatically if not supplied.
    """

    campaign: int = 1
    total_ato_cycles: int = 0          # 0 -> derived from CAMPAIGN_ATO_CYCLES
    ato_cycle: int = 1                 # 1-indexed current ATO cycle

    phase: Phase = Phase.ATO_SETUP
    turn_number: int = 0               # turns elapsed in the current ATO cycle
    # Monotonic count of turns played across the WHOLE game (never reset at
    # cleanup, unlike turn_number). Drives the timed-game cap below.
    total_turns: int = 0
    # Timed-game cap: once total_turns reaches this, the game ends immediately
    # and the winner is decided by victory points at that moment (real games are
    # time-limited). None = uncapped (harness / TUI / direct-engine defaults);
    # the RL env sets it (default 150) so trajectories, web hotseat games and
    # training all share the same bound.
    max_turns: Optional[int] = None
    active_side: Side = Side.US        # whose turn it is now

    # Initiative (set during BID_INITIATIVE each ATO cycle)
    initiative_holder: Optional[Side] = None
    first_player: Optional[Side] = None

    # ATO end tracking: both sides passing in succession ends the cycle.
    consecutive_passes: int = 0

    # Terminal
    winner: Optional[Side] = None
    game_over: bool = False

    # Monotonic token id source.
    next_uid: int = 1

    us: Optional[PlayerState] = None
    prc: Optional[PlayerState] = None

    def __post_init__(self) -> None:
        if self.us is None:
            self.us = PlayerState(side=Side.US)
        if self.prc is None:
            self.prc = PlayerState(side=Side.PRC)
        if not self.total_ato_cycles:
            self.total_ato_cycles = CAMPAIGN_ATO_CYCLES.get(self.campaign, 1)

    # -- construction -------------------------------------------------------

    @classmethod
    def new_game(cls, campaign: int = 1,
                 max_turns: Optional[int] = None) -> "GameState":
        """Create a fresh game at the start of ATO cycle 1, ATO_SETUP phase.

        `max_turns` (None = uncapped) sets the timed-game cap: the game ends by
        VP once that many turns have been played across all ATO cycles.
        """
        return cls(
            campaign=campaign,
            total_ato_cycles=CAMPAIGN_ATO_CYCLES.get(campaign, 1),
            ato_cycle=1,
            phase=Phase.ATO_SETUP,
            max_turns=max_turns,
        )

    # -- side access --------------------------------------------------------

    def player(self, side: Side) -> PlayerState:
        return self.us if side == Side.US else self.prc

    def opponent(self, side: Side) -> PlayerState:
        return self.prc if side == Side.US else self.us

    @property
    def active_player(self) -> PlayerState:
        return self.player(self.active_side)

    # -- token lifecycle ----------------------------------------------------

    def spawn_token(
        self,
        side: Side,
        token_type: TokenType,
        location: BandID,
        origin: TokenOrigin,
        source_card_id: Optional[int] = None,
        grounded: bool = False,
    ) -> TokenInstance:
        """
        Create a token, assign it a unique uid, initialise naval magazines from
        its profile, register it with the side (and its squadron, if any), and
        return it.

        The caller (engine layer) is responsible for validating placement via
        board.valid_spawn_locations before calling this.
        """
        profile = TOKEN_REGISTRY[token_type]
        token = TokenInstance(
            uid=self.next_uid,
            token_type=token_type,
            side=side,
            location=location,
            origin=origin,
            source_card_id=source_card_id,
            air_salvos_remaining=profile.air_salvos,
            surf_salvos_remaining=profile.surf_salvos,
            grounded=grounded,
        )
        self.next_uid += 1

        player = self.player(side)
        player.tokens[token.uid] = token

        if grounded and origin == TokenOrigin.SQUADRON_CARD and source_card_id is not None:
            squad = player.squadrons.get(source_card_id)
            if squad is not None:
                squad.grounded_token_uids.append(token.uid)

        return token

    def get_token(self, uid: int) -> Optional[TokenInstance]:
        token = self.us.tokens.get(uid)
        if token is not None:
            return token
        return self.prc.tokens.get(uid)

    def all_tokens(self) -> Iterator[TokenInstance]:
        """Iterate every living token on the board, both sides."""
        yield from self.us.living_tokens()
        yield from self.prc.living_tokens()

    def tokens_at(self, band: BandID, side: Optional[Side] = None) -> list[TokenInstance]:
        """Living tokens at a band, optionally filtered to one side."""
        if side is not None:
            return self.player(side).tokens_at(band)
        return self.us.tokens_at(band) + self.prc.tokens_at(band)

    def destroy_token(
        self,
        uid: int,
        destroyed_by: Optional[Side] = None,
        on_ground: bool = False,
    ) -> Optional[CaptureRecord]:
        """
        Remove a token from the board. If `destroyed_by` is given, the token
        scores for the enemy: a CaptureRecord is appended to the destroyer's
        captures and returned. This includes tokens produced by Enabler Cards
        (carrier J-15s, EC-130, ...) — shot-down units score regardless of
        which card generated them (user ruling 2026-07-13).

        Squadron-origin tokens are also counted against their card's
        `tokens_lost`: destroyed tokens do not regenerate on later activations.
        """
        token = self.get_token(uid)
        if token is None or token.destroyed:
            return None

        token.destroyed = True
        owner = self.player(token.side)
        owner.tokens.pop(uid, None)

        # Detach from a squadron's grounded list, and record the permanent loss.
        if token.origin == TokenOrigin.SQUADRON_CARD and token.source_card_id is not None:
            squad = owner.squadrons.get(token.source_card_id)
            if squad is not None:
                if uid in squad.grounded_token_uids:
                    squad.grounded_token_uids.remove(uid)
                squad.tokens_lost += 1

        if destroyed_by is None:
            return None

        record = CaptureRecord(
            ato_cycle=self.ato_cycle,
            is_squadron_card=False,
            destroyed_on_ground=on_ground,
            token_type=token.token_type,
            score_type=token.score_type,
            uid=token.uid,
            card_id=token.source_card_id,
        )
        self.player(destroyed_by).captures.append(record)
        return record

    def destroy_squadron(
        self,
        side: Side,
        card_id: int,
        destroyed_by: Optional[Side] = None,
        on_ground: bool = True,
    ) -> list[CaptureRecord]:
        """
        Destroy a Squadron Card and its whole token complement (ruling
        2026-07-17): the card itself, every token still attached to it (grounded
        or returned-to-base), AND every token that was never generated (and now
        never can be) — all score for the destroyer. Airborne tokens are the
        one exception: they keep flying and are turned over later (Winchester /
        grounded / end-of-ATO), so they are NOT scored here.

        Returns the list of CaptureRecords created. Never-generated token
        captures carry `uid=None` (no instance ever existed); the base-strike
        undo uses that to reverse the phantom `tokens_lost` on a cancel.
        """
        owner = self.player(side)
        squad = owner.squadrons.get(card_id)
        records: list[CaptureRecord] = []
        # Guard on the zone, NOT is_destroyed: the damage paths set
        # squad.damage to max BEFORE calling this, and is_destroyed is
        # damage-based — guarding on it silently skipped the card capture
        # (and the +2/+4 VP) for every squadron killed by damage.
        if squad is None or squad.zone == CardZone.DESTROYED:
            return records

        # Full complement minus tokens on the board (airborne + grounded) minus
        # those already permanently lost = tokens that never generated. Computed
        # before the grounded tokens below are destroyed.
        full = TOKEN_REGISTRY[squad.token_type].token_count
        alive = sum(1 for t in owner.living_tokens()
                    if t.origin == TokenOrigin.SQUADRON_CARD and t.source_card_id == card_id)
        never_generated = max(0, full - alive - squad.tokens_lost)

        squad.zone = CardZone.DESTROYED

        # Tokens sitting on the card are destroyed with it.
        for uid in list(squad.grounded_token_uids):
            rec = self.destroy_token(uid, destroyed_by=destroyed_by, on_ground=True)
            if rec is not None:
                records.append(rec)

        # ADA (and any non-flying ground asset) dies WITH its card immediately —
        # it does not "keep flying" like an airborne aircraft (user ruling
        # 2026-08-10). Destroy any still on the board (grounded ones went above).
        for t in [t for t in owner.living_tokens()
                  if t.origin == TokenOrigin.SQUADRON_CARD
                  and t.source_card_id == card_id
                  and t.score_type == TokenScoreType.ADA]:
            rec = self.destroy_token(t.uid, destroyed_by=destroyed_by, on_ground=True)
            if rec is not None:
                records.append(rec)

        if destroyed_by is not None:
            card_record = CaptureRecord(
                ato_cycle=self.ato_cycle,
                is_squadron_card=True,
                destroyed_on_ground=on_ground,
                card_id=card_id,
            )
            self.player(destroyed_by).captures.append(card_record)
            records.append(card_record)

            # Never-generated tokens: one capture each (no uid — never existed).
            for _ in range(never_generated):
                rec = CaptureRecord(
                    ato_cycle=self.ato_cycle,
                    is_squadron_card=False,
                    destroyed_on_ground=on_ground,
                    token_type=squad.token_type,
                    score_type=TOKEN_REGISTRY[squad.token_type].token_score_type,
                    uid=None,
                    card_id=card_id,
                )
                self.player(destroyed_by).captures.append(rec)
                records.append(rec)
            # They are permanently lost (a revived card generates fewer tokens).
            squad.tokens_lost += never_generated

        return records

    # -- cyber --------------------------------------------------------------

    def set_cyber_rate(self, side: Side, rate: int) -> None:
        """Set a side's Cyber Rate (clamped) and check for an instant win."""
        player = self.player(side)
        player.cyber_rate = max(CYBER_RATE_MIN, min(MAX_CYBER_RATE, rate))
        self.check_cyber_win()

    def check_cyber_win(self) -> Optional[Side]:
        """
        Reaching Cyber Rate 4 immediately ends the game. Returns the winning
        side if a cyber win has occurred, else None.
        """
        for side in (Side.US, Side.PRC):
            if self.player(side).cyber_rate >= CYBER_RATE_WIN:
                self.winner = side
                self.game_over = True
                return side
        return None

    # -- turn / cycle bookkeeping ------------------------------------------

    def record_pass(self, side: Side) -> bool:
        """
        Register that `side` passed. Returns True if both sides have now passed
        in succession (the ATO cycle should end).
        """
        self.player(side).passed_last_turn = True
        self.consecutive_passes += 1
        return self.consecutive_passes >= 2

    def record_action(self, side: Side) -> None:
        """Register that `side` took a non-pass action, breaking a pass streak."""
        player = self.player(side)
        player.passed_last_turn = False
        player.acted_this_turn = True
        self.consecutive_passes = 0

    def is_final_ato(self) -> bool:
        return self.ato_cycle >= self.total_ato_cycles
