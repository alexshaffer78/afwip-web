"""
enablers.py — Enabler-card effect handlers for AFWIP.

`rules.RulesEngine.play_enabler` validates a play (hand membership, zone,
response timing) and then dispatches here: `ENABLER_HANDLERS[card_id]` runs the
card's mechanical effect against the engine, returning an `EnablerResult`.

Handler contract
----------------
    handler(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult

Handlers mutate state only through engine helpers (`_acquire_n`,
`_remove_tokens`, `_strike_base`, `_destroy_ship`, `_degrade_cyber`,
`_recover_aircraft`, `_force_discard`, `_cancel_last_attack`,
`_cancel_last_card`, `_generate_enabler_tokens`) or the public `GameState` API,
so all card logic lives in one auditable place and the engine owns the
primitives. The common tail (zone transition, enduring flag, play-logging) is
handled by `play_enabler`, not here.

Scope note: response/cancel handlers implement the *effect* (reversing the last
attack, or voiding the opponent's last-played card). The *timing* of when a
response may be offered is the caller's responsibility — see the module docstring
in rules.py.

The full card taxonomy is encoded in `ENABLER_HANDLERS` at the bottom of the
file; each handler's docstring names its card(s).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from afwip.core.constants import (
    Side,
    BandID,
    TokenScoreType,
    EnablerClass,
    EnablerTrigger,
    DIE_SIDES,
)
from afwip.core.cards import ENABLER_REGISTRY
from afwip.core import board
from afwip.core.state import CardZone

if TYPE_CHECKING:  # pragma: no cover
    from afwip.core.rules import RulesEngine


# ---------------------------------------------------------------------------
# Play parameters and result
# ---------------------------------------------------------------------------

@dataclass
class EnablerPlay:
    """
    Caller-supplied parameters for playing an enabler. A given card reads only
    the fields it needs; handlers validate what they require.
    """
    spawn_band: Optional[BandID] = None
    target_uids: list[int] = field(default_factory=list)
    target_band: Optional[BandID] = None
    target_squadron_id: Optional[int] = None
    # For "OR" cards, the branch to take. Conventional values are documented on
    # each handler (e.g. "acquire"/"advantage", "remove"/"discard",
    # "attack"/"cancel", "degrade"/"cancel", ALD's "cancel_hit"/"cancel_md").
    choice: Optional[str] = None
    # Offensive Cyber's discard branch: the card_ids the VICTIM chose to discard
    # (their own hand, so the choice is theirs — see OFFENSIVE_CYBER_CARDS).
    discard_card_ids: list[int] = field(default_factory=list)
    # Acquire cards whose count comes from a D4: the roll, made BEFORE the card
    # resolves so the owner can choose that many targets (see
    # ROLL_ACQUIRE_CARDS). None = the handler rolls for itself.
    rolled_count: Optional[int] = None
    # Base-strike cards: leave the rolled damage unapplied so the attacker can
    # distribute it point by point (see BASE_STRIKE_CARDS). Set by the script;
    # the engine-only paths apply damage immediately as before.
    defer_allocation: bool = False
    # Defender-declared naval missile defense against an MD-eligible enabler
    # strike (HIMARS / Tomahawk / Ballistic Missile / Land Attack Cruise
    # Missile). Declaring spends one of the ship's air salvos.
    missile_defense_uid: Optional[int] = None


@dataclass
class EnablerResult:
    """What a handler did, for logging and later observation encoding."""
    card_id: int = 0
    tokens: list[int] = field(default_factory=list)        # uids generated
    acquired: list[int] = field(default_factory=list)      # enemy uids acquired
    destroyed: list[int] = field(default_factory=list)     # enemy uids removed/destroyed
    discarded: list[int] = field(default_factory=list)     # enemy enabler card_ids
    recovered: list[int] = field(default_factory=list)     # own token uids / card_ids
    cyber_delta: int = 0
    # Whose Cyber Rate `cyber_delta` moved (own for raises, the opponent's for
    # degrades) — a cancel-card needs it to reverse the shift on the right side.
    cyber_side: Optional[Side] = None
    base_damage: int = 0
    cancelled: bool = False
    # Enemy uids this (cancel) card un-acquired by voiding an acquire card. If
    # this card is ITSELF cancelled, the engine re-acquires them (a cyber
    # cancel-of-a-cancel restores the original acquisition).
    reacquire_on_cancel: list[int] = field(default_factory=list)
    note: str = ""


Handler = Callable[["RulesEngine", Side, EnablerPlay], EnablerResult]


# MD-eligible enabler strikes (HIMARS, Tomahawk, Ballistic Missile, LACM):
# the defender may declare naval missile defense against them.
MD_ELIGIBLE_STRIKE_CARDS = frozenset({41, 42, 75, 76})

# Aerial Refueling (16 US / 66 PRC): "place one squadron card in addition to the
# Posture limit on the airbase ready to generate" — the owner picks which card.
AERIAL_REFUEL_CARDS = frozenset({16, 66})

# Flying Crew Chief (20): the owner names ONE un-activated Contingency-Location
# squadron to generate max aircraft with no D4 roll this ATO cycle.
FLYING_CREW_CHIEF_CARDS = frozenset({20})

# Personnel Recovery (12) / Quick-Turn Mobility (17): recover one of the
# OWNER's just-lost aircraft. Their target_uids must name own tokens — the
# generic default lists the opponent's, which silently no-ops the card.
RECOVER_AIRCRAFT_CARDS = frozenset({12, 17})

# Rapid Resupply (18): an immediate response when one of the owner's Squadron /
# Enabler Cards is discarded — the owner picks which lost card returns.
RAPID_RESUPPLY_CARDS = frozenset({18})

# Cyber Reconnaissance (21 US / 83 PRC): acquires enemy tokens equal to the
# owner's Cyber Rate. The count is known before the card resolves (unlike the
# D4-roll acquire cards), so the owner picks WHICH tokens to acquire.
CYBER_ACQUIRE_CARDS = frozenset({21, 83})

# Offensive Cyber (24 US / 81 PRC): "remove" lets the PLAYER pick which enemy
# tokens die; "discard" lets the VICTIM pick which of their own enablers go.
OFFENSIVE_CYBER_CARDS = frozenset({24, 81})

# Acquire cards whose count is a D4 roll (Space Recon / SOF Recon / Ground-Based
# Radar). The script rolls first, then lets the owner choose that many targets.
ROLL_ACQUIRE_CARDS = frozenset({14, 31, 70, 73, 94, 98})

# Joint Offensive Cyber (34): removes enemy tokens equal to 2x Cyber Rate — the
# owner picks which.
JOINT_OFFENSIVE_CYBER_CARDS = frozenset({34})

# Cyber Counter-UAS (25 US / 84 PRC): destroys enemy UAS tokens equal to the
# Cyber Rate — the owner picks which UAS die.
COUNTER_UAS_CARDS = frozenset({25, 84})

# Constellation Reconstitution (91): return one SPENT PLASSF enabler to the deck
# for the next ATO cycle — the owner picks which spent PLASSF card comes back.
CONSTELLATION_CARDS = frozenset({91})

# UAS Proliferation (68): a response after one of the owner's UAS rolls to
# acquire — an extra acquisition attempt against an enemy token the owner picks.
UAS_PROLIFERATION_CARDS = frozenset({68})

# Special Mission Aircraft (72): its "acquire" branch rolls a D4 and acquires
# that many — roll-then-choose (like ROLL_ACQUIRE_CARDS) but only on that branch.
SPECIAL_MISSION_CARDS = frozenset({72})

# Munitions Upgrade (74): the owner names ONE of their fighter squadrons to shoot
# air-to-air at range 4 this ATO cycle.
MUNITIONS_UPGRADE_CARDS = frozenset({74})

# Red Horse Squadron (38) / Resilient Bases (39): cancel all damage from the
# opponent's last base attack. If one of THESE is itself cancelled (Anti-Access/
# Area Denial), the base damage they voided must be re-applied.
CANCEL_BASE_DAMAGE_CARDS = frozenset({38, 39})

# Infantry Battalion (40): placed on one of the OWNER's bases / contingency
# locations, chosen when the card is played.
INFANTRY_BATTALION_CARDS = frozenset({40})

# Enabler cards that strike an enemy base: the attacker distributes the rolled
# damage point by point, exactly like a token base strike (user ruling
# 2026-07-17). A forced SOF / Infantry Battalion allocation is not deferred.
BASE_STRIKE_CARDS = frozenset({41, 42, 75, 76, 78, 95, 96})

# Enabler cards whose text names the "Airbase" (the permanent 3-hit installation)
# rather than the broader "Base": HIMARS (41), Tomahawk (42), Sea Dragons (95),
# PLANMC Raid (96). Per the wargame FAQ, an "Airbase"-worded strike fills ONLY
# the three airbase VP damage boxes (excess is lost) — it cannot damage the
# Squadron Cards deployed there. The "Base"-worded cards (75/76/78) and every
# token bomb run keep distributing damage across squadron cards + VP boxes.
AIRBASE_ONLY_STRIKE_CARDS = frozenset({41, 42, 95, 96})

# Auto-hit ship-kill cards: the owner picks which enemy surface combatant dies.
# 43/89 only on their "attack" branch, so the pick follows the branch decision.
SHIP_KILL_CARDS = frozenset({43, 44, 77, 89})

# Submarine Strike (43 US) / Diesel-Submarine Strike (89 PRC): their CANCEL
# branch is response-only, valid solely against the opponent's just-played
# submarine card (user ruling 2026-07-14).
SUBMARINE_STRIKE_CARDS = frozenset({43, 89})

# Defensive Cyber (23 US / 82 PRC): "EITHER cancel a PRC Cyber card OR degrade
# the PRC Cyber rate by 1". Like the submarine cards the branch is fixed by
# CONTEXT (user ruling 2026-07-17), not chosen: cancel only as a response to
# the opponent's just-played cyber card, degrade on the owner's own turn.
DEFENSIVE_CYBER_CARDS = frozenset({23, 82})

# "OR" enabler cards and the EnablerPlay.choice values their handlers accept.
ENABLER_CHOICE_BRANCHES: dict[int, list[str]] = {
    24: ["remove", "discard"], 81: ["remove", "discard"],   # Offensive Cyber
    43: ["attack", "cancel"], 89: ["attack", "cancel"],     # Submarine Strike
    72: ["acquire", "advantage"],                           # Special Mission Aircraft
}


def card_play_triggers(profile) -> set[EnablerTrigger]:
    """Which OPP_PLAYS_* triggers fire when `profile` is played (for cancels)."""
    trigs: set[EnablerTrigger] = set()
    if profile.enabler_class == EnablerClass.SOF:
        trigs.add(EnablerTrigger.OPP_PLAYS_SOF)
    if profile.enabler_class == EnablerClass.SPACE:
        trigs.add(EnablerTrigger.OPP_PLAYS_SPACE_CARD)
    if profile.enabler_class == EnablerClass.CYBER:
        trigs.add(EnablerTrigger.OPP_PLAYS_CYBER_CARD)
    if profile.mobility_maintenance:
        trigs.add(EnablerTrigger.OPP_PLAYS_MOBILITY_MAINT)
    if "SUBMARINE" in profile.name.upper():
        trigs.add(EnablerTrigger.OPP_PLAYS_SUBMARINE_CARD)
    return trigs


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _r(card_id: int, **kw) -> EnablerResult:
    return EnablerResult(card_id=card_id, **kw)


def _enemy_airbase(side: Side) -> BandID:
    return board.own_airbase(board.opponent(side))


# --- token generation --------------------------------------------------------

def h_generate(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """Generate the card's token(s) at spawn_band. Cards 30,36,37,85,86,87,88."""
    cid = _current_card(engine, side)
    uids = engine._generate_enabler_tokens(side, cid, play.spawn_band)
    return _r(cid, tokens=uids)


# --- enduring (roll-mod / markers) ------------------------------------------

def h_enduring_noop(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """Enduring roll-modifier cards (19,26,27,28,35,71,90) — the tail sets the flag."""
    return _r(_current_card(engine, side))


def h_munitions_upgrade(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """74 — one fighter squadron shoots air-to-air at range 4 this ATO cycle."""
    from afwip.core.tokens import TOKEN_REGISTRY
    cid = _current_card(engine, side)
    squad = engine.state.player(side).squadrons.get(play.target_squadron_id or -1)
    if squad is None or squad.is_destroyed:
        raise _illegal(engine, "Munitions Upgrade requires one of your squadrons")
    if TOKEN_REGISTRY[squad.token_type].token_score_type != TokenScoreType.FIGHTER:
        raise _illegal(engine, "Munitions Upgrade targets a fighter squadron")
    engine.state.player(side).air_range_override[play.target_squadron_id] = 4
    return _r(cid, note=f"range 4 for squadron {play.target_squadron_id}")


def h_flying_crew_chief(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """
    20 — "Automatically generate max aircraft for 1 Squadron at a contingency
    location this ATO cycle (1d4 roll not used)". The owner names WHICH
    Contingency-Location squadron (`target_squadron_id`); only CL squadrons
    that have not activated yet are valid.
    """
    cid = _current_card(engine, side)
    candidates = engine.flying_crew_chief_candidates(side)
    target = play.target_squadron_id
    if target is None and candidates:
        target = candidates[0]
    if target is not None and target not in candidates:
        raise _illegal(engine, "Flying Crew Chief must name one of your "
                               "un-activated Contingency Location squadrons")
    engine.state.player(side).cl_max_squadron_id = target
    note = "" if target is not None else "no Contingency Location squadron available"
    return _r(cid, recovered=[target] if target is not None else [], note=note)


# --- acquisition -------------------------------------------------------------

def h_acquire_roll(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """
    Roll a D4, acquire that many enemy tokens. Cards 31,70,73,94.
    `play.rolled_count` carries a roll already made so the owner could pick
    which tokens to take (the script rolls before offering the choice).
    """
    cid = _current_card(engine, side)
    n = play.rolled_count if play.rolled_count is not None else engine._d4()
    got = engine._acquire_n(side, n, play.target_uids)
    return _r(cid, acquired=got, note=f"rolled {n}")


def h_acquire_roll_cyber(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """
    SOF Recon: roll D4, acquire that many; on a natural 4 also +1 Cyber.
    Cards 14,98. Honours a pre-made `play.rolled_count` (see h_acquire_roll).
    """
    cid = _current_card(engine, side)
    roll = play.rolled_count if play.rolled_count is not None else engine._d4()
    got = engine._acquire_n(side, roll, play.target_uids)
    delta = 0
    if roll == DIE_SIDES:
        before = engine.state.player(side).cyber_rate
        engine.state.set_cyber_rate(side, before + 1)
        delta = engine.state.player(side).cyber_rate - before
    return _r(cid, acquired=got, cyber_delta=delta, cyber_side=side if delta else None,
              note=f"rolled {roll}")


def h_acquire_cyber(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """Acquire enemy tokens equal to own Cyber Rate. Cards 21,83."""
    cid = _current_card(engine, side)
    n = engine.state.player(side).cyber_rate
    got = engine._acquire_n(side, n, play.target_uids)
    return _r(cid, acquired=got)


def h_special_mission_aircraft(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """72 — choose: roll D4 and acquire, OR next air-to-air at advantage."""
    cid = _current_card(engine, side)
    if play.choice == "advantage":
        engine.state.player(side).pending_air_advantage = True
        return _r(cid, note="next air-to-air at advantage")
    # Roll-then-choose: honour a roll the script already made (so the owner could
    # pick that many targets); roll for ourselves on engine-only paths.
    n = play.rolled_count if play.rolled_count is not None else engine._d4()
    got = engine._acquire_n(side, n, play.target_uids)
    return _r(cid, acquired=got, note=f"rolled {n}")


def h_uas_proliferation(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """
    68 — response to a UAS acquire roll: one extra acquisition attempt (a fresh
    roll against the target's acquisition value — the card grants a roll, not
    an automatic acquisition).
    """
    from afwip.core.rules import RollContext, combine_modes
    cid = _current_card(engine, side)
    target = None
    for uid in play.target_uids:
        t = engine.state.get_token(uid)
        if t is not None and t.is_active and t.side != side and not t.acquired \
                and engine._acquirable(t):
            target = t
            break
    if target is None:
        return _r(cid, note="no eligible target")
    mode = combine_modes(engine._enduring_modes(side, RollContext.ACQUIRE, None))
    r = engine.roll(mode)
    if r.value >= target.profile.acquisition_value:
        target.acquired = True
        return _r(cid, acquired=[target.uid], note=f"rolled {r.value}")
    return _r(cid, note=f"rolled {r.value}, failed")


# --- cyber -------------------------------------------------------------------

def h_cyber_raise(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """Roll to raise own Cyber Rate against the access value. Cards 15,22,80,97."""
    cid = _current_card(engine, side)
    before = engine.state.player(side).cyber_rate
    engine.attempt_cyber_raise(side)
    delta = engine.state.player(side).cyber_rate - before
    note = "" if delta else f"cyber raise roll failed (rate stays {before})"
    return _r(cid, cyber_delta=delta, cyber_side=side if delta else None, note=note)


def h_cyber_degrade2(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """33 — degrade the opponent's Cyber Rate by 2."""
    cid = _current_card(engine, side)
    delta = engine._degrade_cyber(board.opponent(side), 2)
    return _r(cid, cyber_delta=-delta, cyber_side=board.opponent(side) if delta else None)


def h_defensive_cyber(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """
    Defensive Cyber (23 US / 82 PRC): EITHER cancel the opponent's last cyber
    card OR degrade their Cyber Rate by 1. The branch is fixed by CONTEXT (user
    ruling 2026-07-17): choice="cancel" when played as a response to the
    opponent's cyber card, "degrade" on the owner's own turn. An invalid cancel
    aborts without spending the card (mirrors Submarine Strike).
    """
    cid = _current_card(engine, side)
    if play.choice == "cancel":
        ok = engine._cancel_last_card(lambda p: p.enabler_class == EnablerClass.CYBER,
                                      canceller=side)
        if not ok:
            raise _illegal(engine, "CANCEL requires the opponent's just-played "
                                   "Cyber card")
        # Carry the un-acquired uids so that cancelling THIS Defensive Cyber
        # restores the acquisition (cyber cancel-of-a-cancel).
        res = _r(cid, cancelled=True)
        res.reacquire_on_cancel = list(engine._reversed_acquired)
        return res
    delta = engine._degrade_cyber(board.opponent(side), 1)
    return _r(cid, cyber_delta=-delta, cyber_side=board.opponent(side) if delta else None)


# --- token removal -----------------------------------------------------------

def _remove_amount(engine: "RulesEngine", side: Side, multiplier: int) -> int:
    return engine.state.player(side).cyber_rate * multiplier


def h_offensive_cyber(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """
    Offensive Cyber (24 US / 81 PRC): remove enemy tokens = Cyber Rate, OR force
    the opponent to discard that many enablers. choice="discard" or "remove".
    """
    cid = _current_card(engine, side)
    n = _remove_amount(engine, side, 1)
    if play.choice == "discard":
        # The VICTIM chooses which of their own cards to give up.
        gone = engine._force_discard(board.opponent(side), n, play.discard_card_ids)
        return _r(cid, discarded=gone)
    # The player chooses which enemy tokens to destroy.
    removed = engine._remove_tokens(side, n, play.target_uids)
    return _r(cid, destroyed=removed)


def h_joint_offensive_cyber(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """34 — remove enemy tokens equal to 2x own Cyber Rate."""
    cid = _current_card(engine, side)
    removed = engine._remove_tokens(side, _remove_amount(engine, side, 2), play.target_uids)
    return _r(cid, destroyed=removed)


def h_counter_uas(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """Destroy enemy UAS tokens equal to own Cyber Rate. Cards 25,84."""
    cid = _current_card(engine, side)
    n = _remove_amount(engine, side, 1)
    removed = engine._remove_tokens(side, n, play.target_uids, score_type=TokenScoreType.UAS)
    return _r(cid, destroyed=removed)


# --- base strikes ------------------------------------------------------------

def _strike(engine, side, play, *, gate, bypass) -> EnablerResult:
    cid = _current_card(engine, side)
    band = play.target_band or _enemy_airbase(side)
    sof = ENABLER_REGISTRY[cid].enabler_class == EnablerClass.SOF
    dmg = engine._strike_base(side, band, roll_gate=gate, bypass_md=bypass,
                              target_squadron_id=play.target_squadron_id, sof=sof,
                              missile_defense_uid=play.missile_defense_uid,
                              defer_allocation=play.defer_allocation,
                              airbase_only=cid in AIRBASE_ONLY_STRIKE_CARDS)
    return _r(cid, base_damage=dmg)


def h_strike_gated(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """Roll 2+ to succeed, then D4 damage to the enemy base. Cards 41,75,76 (MD-eligible)."""
    return _strike(engine, side, play, gate=2, bypass=False)


def h_strike_unblockable(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """Unblockable D4 damage to the enemy base (bypasses MD). Cards 78,95,96."""
    return _strike(engine, side, play, gate=None, bypass=True)


def h_tomahawk(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """42 — generate DDG 115, then roll 2+ for D4 damage to the PRC airbase."""
    cid = _current_card(engine, side)
    uids = engine._generate_enabler_tokens(side, cid, play.spawn_band)
    band = _enemy_airbase(side)
    dmg = engine._strike_base(side, band, roll_gate=2, bypass_md=False,
                              missile_defense_uid=play.missile_defense_uid,
                              defer_allocation=play.defer_allocation,
                              airbase_only=True)   # 42 Tomahawk strikes the "airbase"
    return _r(cid, tokens=uids, base_damage=dmg)


# --- ship kills --------------------------------------------------------------

def h_marine_littoral(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """
    44 — auto-hit one enemy surface combatant. Only playable while an enemy
    surface combatant is on the board (user ruling 2026-07-14).
    """
    cid = _current_card(engine, side)
    if not engine.state.opponent(side).naval_tokens():
        raise _illegal(engine, "Marine Littoral Regiment requires an enemy "
                               "surface combatant on the board")
    uid = engine._destroy_ship(side, play.target_uids[0] if play.target_uids else None, bypass_md=False)
    return _r(cid, destroyed=[uid] if uid is not None else [])


def h_maritime_cruise(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """77 — unblockable: immediately destroy one enemy surface combatant."""
    cid = _current_card(engine, side)
    uid = engine._destroy_ship(side, play.target_uids[0] if play.target_uids else None, bypass_md=True)
    return _r(cid, destroyed=[uid] if uid is not None else [])


def h_submarine_strike(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """
    Submarine Strike (43 US / 89 PRC): attack one enemy surface combatant
    (auto-hits) OR cancel an opponent submarine card. choice="cancel" or "attack".
    The CANCEL branch is valid only against the opponent's just-played submarine
    card (user ruling 2026-07-14) — an invalid cancel aborts without spending
    the card.
    """
    cid = _current_card(engine, side)
    if play.choice == "cancel":
        ok = engine._cancel_last_card(lambda p: "SUBMARINE" in p.name.upper(),
                                      canceller=side)
        if not ok:
            raise _illegal(engine, "CANCEL requires the opponent's just-played "
                                   "submarine card")
        return _r(cid, cancelled=True)
    uid = engine._destroy_ship(side, play.target_uids[0] if play.target_uids else None, bypass_md=False)
    return _r(cid, destroyed=[uid] if uid is not None else [])


# --- recovery / regeneration -------------------------------------------------

def h_recover_aircraft(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """Recover a lost aircraft to the front band; not counted destroyed. Cards 12,17."""
    cid = _current_card(engine, side)
    uid = engine._recover_aircraft(side, play.target_uids[0] if play.target_uids else None)
    return _r(cid, recovered=[uid] if uid is not None else [])


def h_rapid_resupply(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """18 — recover any discarded Squadron or Enabler Card (usable immediately)."""
    cid = _current_card(engine, side)
    ok = engine._recover_card(side, play.target_squadron_id, allow_squadron=True)
    return _r(cid, recovered=[play.target_squadron_id] if ok and play.target_squadron_id else [])


def h_constellation_reconstitution(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """91 — return a spent PLASSF card to the deck for the next ATO cycle."""
    cid = _current_card(engine, side)
    ok = engine._recover_card(side, play.target_squadron_id, to_deck=True, require_plassf=True)
    return _r(cid, recovered=[play.target_squadron_id] if ok and play.target_squadron_id else [])


def h_reserves(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """
    67 — fully regenerate a squadron that lost ALL its tokens. Only playable
    immediately after the opponent destroyed the squadron's last token (user
    ruling 2026-07-14); an ineligible target aborts without spending the card.
    """
    cid = _current_card(engine, side)
    ok = engine._regenerate_squadron(side, play.target_squadron_id)
    if not ok:
        raise _illegal(engine, "Reserves requires a squadron whose tokens were "
                               "all destroyed by the opponent")
    return _r(cid, recovered=[play.target_squadron_id])


# --- markers / setup ---------------------------------------------------------

def h_aerial_refueling(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """
    Aerial Refueling (16 US / 66 PRC): place one extra Squadron Card beyond the
    posture limit on the airbase, ready to generate. `target_squadron_id` names
    the card to add.
    """
    cid = _current_card(engine, side)
    added = engine._add_squadron(side, play.target_squadron_id)
    engine.state.player(side).extra_squadron_slots += 1
    note = f"added squadron {play.target_squadron_id}" if added else "+1 squadron slot"
    return _r(cid, recovered=[play.target_squadron_id] if added else [], note=note)


def h_infantry_battalion(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """
    40 — "Place on any base or contingency location. Squadrons on that base may
    not be attacked by SOF. This card may be target of a base attack."

    The card is placed on one of the OWNER's bases (`target_band`) and starts
    undamaged; it soaks SOF damage for the squadrons there until both of its
    printed damage boxes are filled.
    """
    cid = _current_card(engine, side)
    legal = engine.infantry_battalion_bases(side)
    band = play.target_band if play.target_band in legal else (legal[0] if legal else None)
    if band is None:
        raise _illegal(engine, "Infantry Battalion needs one of your bases to be placed on")
    engine.state.player(side).infantry_battalions.setdefault(band, 0)
    return _r(cid, note=f"placed on {band.name}")


# --- one-shot buffs ----------------------------------------------------------

def h_forward_observers(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """13 — the next declared surface strike auto-hits."""
    cid = _current_card(engine, side)
    engine.state.player(side).pending_auto_hit = True
    return _r(cid)


def h_elite_pilots(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """69 — the next air-to-air attack is rolled at advantage."""
    cid = _current_card(engine, side)
    engine.state.player(side).pending_air_advantage = True
    return _r(cid)


# --- cancels (response) ------------------------------------------------------

def h_air_launched_decoy(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """
    29 — cancel a PRC rolled hit (default / choice="cancel_hit"), or cancel a
    PRC declared missile-defense attempt (choice="cancel_md": the next declared
    naval MD is voided; its salvo is still spent).
    """
    cid = _current_card(engine, side)
    if play.choice == "cancel_md":
        engine._md_cancelled_next = True
        return _r(cid, cancelled=True, note="missile defense cancelled")
    ok = engine._cancel_last_attack(side, damage_only=False)
    return _r(cid, cancelled=ok)


def h_cancel_base_damage(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """Cancel all damage from the opponent's last base attack. Cards 38,39."""
    cid = _current_card(engine, side)
    ok = engine._cancel_last_attack(side, damage_only=True)
    return _r(cid, cancelled=ok)


def h_cancel_md(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
    """79 — cancel the opponent's declared missile-defense attempt (salvo still spent)."""
    cid = _current_card(engine, side)
    engine._md_cancelled_next = True
    return _r(cid, cancelled=True, note="missile defense cancelled")


def _h_cancel_card_of(allowed) -> Handler:
    """Build a cancel-card handler restricted to the card type its text names."""
    def handler(engine: "RulesEngine", side: Side, play: EnablerPlay) -> EnablerResult:
        cid = _current_card(engine, side)
        ok = engine._cancel_last_card(allowed, canceller=side)
        return _r(cid, cancelled=ok)
    return handler


# 11: cancel a SOF card; 32/92/93: cancel a space card; 65: cancel a US
# mobility/maintenance card.
h_cancel_sof_card = _h_cancel_card_of(lambda p: p.enabler_class == EnablerClass.SOF)
h_cancel_space_card = _h_cancel_card_of(lambda p: p.enabler_class == EnablerClass.SPACE)
h_cancel_mobility_card = _h_cancel_card_of(lambda p: bool(p.mobility_maintenance))


# ---------------------------------------------------------------------------
# Small helpers shared with the engine
# ---------------------------------------------------------------------------

def _current_card(engine: "RulesEngine", side: Side) -> int:
    """The card_id currently being resolved (set by play_enabler before dispatch)."""
    return engine._resolving_card_id


def _illegal(engine: "RulesEngine", msg: str) -> Exception:
    from afwip.core.rules import IllegalAction
    return IllegalAction(msg)


# ---------------------------------------------------------------------------
# Registry: card_id -> handler
# ---------------------------------------------------------------------------

ENABLER_HANDLERS: dict[int, Handler] = {
    # US ---------------------------------------------------------------------
    11: h_cancel_sof_card,        # AC-130 Gunship: cancel SOF card
    12: h_recover_aircraft,       # Personnel Recovery
    13: h_forward_observers,      # Forward Observers: next strike auto-hits
    14: h_acquire_roll_cyber,     # SOF Reconnaissance (+cyber on 4)
    15: h_cyber_raise,            # SOF Cyber Infiltration
    16: h_aerial_refueling,       # Aerial Refueling
    17: h_recover_aircraft,       # Quick-Turn Mobility
    18: h_rapid_resupply,         # Rapid Resupply
    19: h_enduring_noop,          # Improved Munitions (enduring)
    20: h_flying_crew_chief,      # Flying Crew Chief (enduring CL max)
    21: h_acquire_cyber,          # Cyber Reconnaissance
    22: h_cyber_raise,            # Cyber Infiltration
    23: h_defensive_cyber,        # Defensive Cyber (cancel OR degrade)
    24: h_offensive_cyber,        # Offensive Cyber (remove OR discard)
    25: h_counter_uas,            # Cyber Counter-UAS
    26: h_enduring_noop,          # EW Spoofing (enduring)
    27: h_enduring_noop,          # Defensive EW (enduring)
    28: h_enduring_noop,          # Offensive EW (enduring)
    29: h_air_launched_decoy,     # Air Launched Decoy: cancel hit/MD
    30: h_generate,               # EC-130 Compass Call: generate EC-130
    31: h_acquire_roll,           # Space Reconnaissance
    32: h_cancel_space_card,      # Counter Space
    33: h_cyber_degrade2,         # Joint Defensive Cyber
    34: h_joint_offensive_cyber,  # Joint Offensive Cyber
    35: h_enduring_noop,          # Space-Based EW (enduring)
    36: h_generate,               # Land-Based Missile Defense: ADA
    37: h_generate,               # Maritime Missile Defense: DDG 81
    38: h_cancel_base_damage,     # Red Horse Squadron
    39: h_cancel_base_damage,     # Resilient Bases
    40: h_infantry_battalion,     # Infantry Battalion
    41: h_strike_gated,           # HIMARS
    42: h_tomahawk,               # Tomahawk Strike
    43: h_submarine_strike,       # Submarine Strike (attack OR cancel)
    44: h_marine_littoral,        # Marine Littoral Regiment
    # PRC --------------------------------------------------------------------
    65: h_cancel_mobility_card,   # Anti-Access/Area Denial: cancel mobility/maint
    66: h_aerial_refueling,       # Aerial Refueling
    67: h_reserves,               # Reserves: regenerate squadron
    68: h_uas_proliferation,      # UAS Proliferation
    69: h_elite_pilots,           # Elite Pilots: next air-to-air advantage
    70: h_acquire_roll,           # Ground-Based Radar
    71: h_enduring_noop,          # Badger Surge (enduring)
    72: h_special_mission_aircraft,  # Special Mission Aircraft (acquire OR advantage)
    73: h_acquire_roll,           # SOF Reconnaissance
    74: h_munitions_upgrade,      # Munitions Upgrade (enduring range 4)
    75: h_strike_gated,           # Ballistic Missile Strike
    76: h_strike_gated,           # Land Attack Cruise Missile
    77: h_maritime_cruise,        # Maritime Strike Cruise Missile (unblockable)
    78: h_strike_unblockable,     # Hypersonic Missile (unblockable)
    79: h_cancel_md,              # Decoy Warheads: cancel US MD
    80: h_cyber_raise,            # Cyber Infiltration
    81: h_offensive_cyber,        # Offensive Cyber Operations (remove OR discard)
    82: h_defensive_cyber,        # Defensive Cyber Operations (cancel OR degrade)
    83: h_acquire_cyber,          # Cyber Reconnaissance
    84: h_counter_uas,            # Cyber Counter-UAS
    85: h_generate,               # Shandong J-15 Squadron: 4x J-15
    86: h_generate,               # Guided Missile Cruiser
    87: h_generate,               # Guided Missile Destroyer
    88: h_generate,               # Missile Boat Flotilla
    89: h_submarine_strike,       # Diesel-Submarine Strike (attack OR cancel)
    90: h_enduring_noop,          # Space-Based EW (enduring)
    91: h_constellation_reconstitution,  # Constellation Reconstitution (PLASSF)
    92: h_cancel_space_card,      # Counter Space
    93: h_cancel_space_card,      # Anti-Satellite Strike
    94: h_acquire_roll,           # Space Reconnaissance
    95: h_strike_unblockable,     # Sea Dragons Strike (SOF, unblockable)
    96: h_strike_unblockable,     # PLANMC Raid (SOF, unblockable)
    97: h_cyber_raise,            # SOF Cyber Infiltration
    98: h_acquire_roll_cyber,     # SOF Reconnaissance (+cyber on 4)
}
