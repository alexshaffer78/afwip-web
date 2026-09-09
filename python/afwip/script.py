"""
script.py — The AFWIP decision-node game script.

Reduces a full game to a stream of DecisionNodes, each of which is "choose one
item from a presented list": drafting (missions / posture / squadrons /
enablers), initiative bidding, intel reveals, player-turn sub-actions, enabler
parameters (spawn bands, OR-branches), missile-defense declarations, response
windows, and base-strike damage allocation.

Control flow is a generator coroutine: `GameScript(engine).run()` yields
DecisionNodes and receives the picked Choice back through `.send()`. This
keeps mid-turn interleavings (a defender's MD declaration during the
attacker's strike, the attacker's cancel-MD response, per-point damage
allocation) trivially correct without a hand-written state machine.

Consumers: the PettingZoo env (afwip/env.py — adds observation encoding and
the AEC protocol) and the terminal visualizer (afwip/tui.py — renders the
board and lets a human or random agent answer each node).

Fidelity matches the interactive harness: spawn bands, OR-card branches,
missile defense, responses, and base-damage allocation are player decisions;
other enabler targets use the engine's defaults (`_default_enabler_play`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional

from afwip.core.rules import RulesEngine, IllegalAction, LegalAction, PendingBaseAllocation
from afwip.core.state import CardZone
from afwip.core.constants import (
    Side, Phase, BandID, PostureType, TokenType, TokenScoreType, EnablerTrigger,
    DAMAGE_TO_DESTROY_SQUADRON, AIRBASE_BONUS_DAMAGE_BOXES,
)
from afwip.core.cards import (
    MISSION_REGISTRY, SQUADRON_REGISTRY, ENABLER_REGISTRY, POSTURE_REGISTRY,
)
from afwip.core.enablers import (
    AERIAL_REFUEL_CARDS, AIRBASE_ONLY_STRIKE_CARDS, BASE_STRIKE_CARDS,
    CONSTELLATION_CARDS, COUNTER_UAS_CARDS,
    CYBER_ACQUIRE_CARDS, DEFENSIVE_CYBER_CARDS, SHIP_KILL_CARDS,
    ENABLER_CHOICE_BRANCHES, FLYING_CREW_CHIEF_CARDS, INFANTRY_BATTALION_CARDS,
    JOINT_OFFENSIVE_CYBER_CARDS, MD_ELIGIBLE_STRIKE_CARDS,
    MUNITIONS_UPGRADE_CARDS, OFFENSIVE_CYBER_CARDS, RAPID_RESUPPLY_CARDS,
    ROLL_ACQUIRE_CARDS, SPECIAL_MISSION_CARDS, SUBMARINE_STRIKE_CARDS,
    UAS_PROLIFERATION_CARDS, card_play_triggers,
)
from afwip.core.tokens import TOKEN_REGISTRY
from afwip.core import board


class NodeType(Enum):
    MISSION_PICK = 0
    POSTURE_PICK = 1
    SQUADRON_PICK = 2
    SQUADRON_BASE = 3
    ENABLER_PICK = 4
    BID_SACRIFICE = 5
    FIRST_PLAYER = 6
    INTEL_REVEAL = 7
    TURN_ACTION = 8
    SPAWN_BAND = 9
    ENABLER_BRANCH = 10
    MD_DECLARE = 11
    RESPONSE = 12
    ALLOC_POINT = 13


@dataclass
class Choice:
    """One selectable option at a decision node, with encoding hints."""
    label: str
    value: Any                          # semantic payload consumed by the script
    kind: str = ""                     # LegalAction kind, or "done"/"skip"/"pick"
    card_id: Optional[int] = None
    band: Optional[BandID] = None
    actor_uid: Optional[int] = None    # own token performing / declared
    target_uid: Optional[int] = None   # enemy token targeted
    branch_idx: Optional[int] = None   # OR-card branch index
    alloc_kind: Optional[str] = None   # "squadron"/"token"/"infantry"/"vp"


@dataclass
class DecisionNode:
    node_type: NodeType
    side: Side
    choices: list[Choice] = field(default_factory=list)


def _score_type(squadron_card_id: int) -> TokenScoreType:
    return TOKEN_REGISTRY[SQUADRON_REGISTRY[squadron_card_id].token_type].token_score_type


class GameScript:
    """Drives a RulesEngine through a full game as a decision-node coroutine."""

    def __init__(self, engine: RulesEngine):
        self.engine = engine

    def run(self) -> Iterator[DecisionNode]:
        eng = self.engine
        while not eng.state.game_over:
            if eng.state.phase == Phase.ATO_SETUP:
                yield from self._ato_setup()
                continue
            side = eng.state.active_side
            actions = eng.legal_actions(side)
            pick = yield DecisionNode(NodeType.TURN_ACTION, side,
                                      [self._wrap_action(a) for a in actions])
            yield from self._apply_turn_action(side, pick.value)

    @staticmethod
    def _wrap_action(a: LegalAction) -> Choice:
        return Choice(label=a.label or a.kind, value=a, kind=a.kind, card_id=a.card_id,
                      band=a.dest_band or a.target_band, actor_uid=a.token_uid,
                      target_uid=a.target_uid)

    # -- ATO setup ---------------------------------------------------------

    def _ato_setup(self) -> Iterator[DecisionNode]:
        eng = self.engine
        camp = eng.campaign
        if eng.state.us.mission_card_id is None:      # once per game
            picks = {}
            for side in (Side.US, Side.PRC):
                opts = [cid for cid, m in MISSION_REGISTRY.items()
                        if m.side == side and camp.mission_allowed(m.mission_type)]
                pick = yield DecisionNode(NodeType.MISSION_PICK, side, [
                    Choice(MISSION_REGISTRY[cid].name, value=cid, kind="pick", card_id=cid)
                    for cid in opts])
                picks[side] = pick.value
            eng.setup_missions(picks[Side.US], picks[Side.PRC])

        for side in (Side.US, Side.PRC):
            yield from self._draft_side(side)

        sacs = {Side.US: [], Side.PRC: []}
        if camp.number != 2:                           # Tournament: no sacrifices
            for side in (Side.US, Side.PRC):
                sacs[side] = yield from self._pick_sacrifices(side)
        eng.bid_for_initiative(us_sacrifice=sacs[Side.US], prc_sacrifice=sacs[Side.PRC])
        if eng.state.game_over:                        # cyber instant win off the bid raise
            return

        winner = eng.state.initiative_holder
        loser = board.opponent(winner)
        pick = yield DecisionNode(NodeType.FIRST_PLAYER, winner, [
            Choice(f"{winner.value} moves first", value=winner, kind="pick"),
            Choice(f"{loser.value} moves first", value=loser, kind="pick")])
        eng.choose_first_player(pick.value)

        counts = eng.play_intel_roll()
        for owner in (Side.US, Side.PRC):
            n = counts.get(owner, 0)
            hand = [c.card_id for c in eng.state.player(owner).enablers_in_hand()]
            chosen: list[int] = []
            for _ in range(n):
                avail = [cid for cid in hand if cid not in chosen]
                pick = yield DecisionNode(NodeType.INTEL_REVEAL, owner, [
                    Choice(ENABLER_REGISTRY[cid].name, value=cid, kind="pick", card_id=cid)
                    for cid in avail])
                chosen.append(pick.value)
            eng.play_intel_reveal(owner, chosen)
        eng.begin_player_turns()

    def _draft_side(self, side: Side) -> Iterator[DecisionNode]:
        eng = self.engine
        camp = eng.campaign
        player = eng.state.player(side)

        postures = [cid for cid, p in POSTURE_REGISTRY.items()
                    if p.side == side and camp.posture_allowed(p.posture_type)
                    and (p.can_repeat or p.posture_type not in player.postures_used)]
        pick = yield DecisionNode(NodeType.POSTURE_PICK, side, [
            Choice(POSTURE_REGISTRY[cid].name, value=cid, kind="pick", card_id=cid)
            for cid in postures])
        posture = pick.value
        prof = POSTURE_REGISTRY[posture]

        # Squadrons AND enablers are re-drafted EVERY cycle (ruling 2026-07-22):
        # each cycle the player picks a posture, then which surviving Squadron
        # Cards to field and where (placement re-chosen each cycle). A survivor
        # keeps its CARD damage but re-fields its full complement; a destroyed
        # card stays out of the pool below.
        pool = [cid for cid, s in SQUADRON_REGISTRY.items()
                if s.side == side and camp.squadron_allowed(cid)
                and not (cid in player.squadrons and player.squadrons[cid].is_destroyed)]
        if prof.flying_only:
            pool = [c for c in pool if SQUADRON_REGISTRY[c].flying]
        # Campaign 1 forces the side's specific card (US F-16 / PRC J-10) to be
        # INCLUDED in an otherwise-normal Standard draft: pre-select it (it counts
        # toward the roster) and let the player draft the rest.
        forced = camp.forced_squadrons.get(side)
        preselected = [c for c in sorted(forced) if c in pool] if forced else []
        bonus = [(lambda c: _score_type(c) == TokenScoreType.ADA, prof.ada_bonus_squadron),
                 (lambda c: _score_type(c) == TokenScoreType.BOMBER, prof.bomber_bonus)]
        squads = yield from self._pick_conditional(
            NodeType.SQUADRON_PICK, side, pool,
            base=prof.squadrons + player.extra_squadron_slots, bonus_specs=bonus,
            # May stop one below the posture count (ruling 2026-07-31); if the
            # pool can't reach one-below, all remaining are drafted.
            min_required=min(max(prof.squadrons - 1, 0), len(pool)),
            preselected=preselected,
            label=lambda c: SQUADRON_REGISTRY[c].name)

        locations = None
        if side == Side.US and squads and prof.posture_type != PostureType.SURGE \
                and not player.cl_banned_campaign:
            locations = {}
            for cid in squads:
                pick = yield DecisionNode(NodeType.SQUADRON_BASE, side, [
                    Choice(f"{SQUADRON_REGISTRY[cid].name}: Airbase",
                           value=BandID.US_AIRBASE, kind="pick", card_id=cid,
                           band=BandID.US_AIRBASE),
                    Choice(f"{SQUADRON_REGISTRY[cid].name}: Contingency Location",
                           value=BandID.US_CONTINGENCY_LOCATION, kind="pick", card_id=cid,
                           band=BandID.US_CONTINGENCY_LOCATION)])
                locations[cid] = pick.value

        enablers = yield from self._draft_enablers(side, prof)
        eng.select_posture(side, posture, squads, enablers, squadron_locations=locations)

        # HEDGEHOG grants a free ADA token; the US owner places it on the Airbase
        # or the Contingency Location (chosen before begin_player_turns spawns
        # it). CL is unavailable if a SURGE campaign ban is in force.
        if side == Side.US and prof.ada_bonus_token > 0 and not player.cl_banned_campaign:
            pick = yield DecisionNode(NodeType.SQUADRON_BASE, side, [
                Choice(f"{prof.name} ADA: Airbase", value=BandID.US_AIRBASE,
                       kind="pick", band=BandID.US_AIRBASE),
                Choice(f"{prof.name} ADA: Contingency Location",
                       value=BandID.US_CONTINGENCY_LOCATION, kind="pick",
                       band=BandID.US_CONTINGENCY_LOCATION)])
            eng.state.player(side).posture_bonus_ada_location = pick.value

    def _draft_enablers(self, side: Side, prof) -> Iterator[DecisionNode]:
        """Enabler drafting, used in EVERY cycle (ruling 2026-07-17): the pool
        excludes only cards REMOVED from the campaign (played single-use)."""
        eng = self.engine
        camp = eng.campaign
        player = eng.state.player(side)
        if not camp.enablers_allowed:
            return []
        en_pool = [cid for cid, e in ENABLER_REGISTRY.items()
                   if e.side == side and camp.enabler_allowed(cid)
                   and not (cid in player.enablers
                            and player.enablers[cid].zone == CardZone.REMOVED)]
        if prof.plaaf_only:
            en_pool = [c for c in en_pool if ENABLER_REGISTRY[c].plaaf]
        en_bonus = [(lambda c: bool(ENABLER_REGISTRY[c].plarf), prof.plarf_bonus or 0)]
        return (yield from self._pick_conditional(
            NodeType.ENABLER_PICK, side, en_pool,
            base=prof.enablers, bonus_specs=en_bonus,
            min_required=min(prof.enablers, len(en_pool)),
            label=lambda c: ENABLER_REGISTRY[c].name))

    def _pick_conditional(self, node_type: NodeType, side: Side, pool: list[int],
                          base: int, bonus_specs, min_required: int,
                          label, preselected=None) -> Iterator[DecisionNode]:
        """Sequential card picking with the posture's bonus-slot accounting
        (mirrors the interactive harness's _pick_conditional). `preselected` cards
        (e.g. Campaign 1's forced squadron) start in the draft, count toward the
        base/min, and are not offered again — they can't be dropped."""
        chosen: list[int] = list(preselected or [])
        remaining = [c for c in pool if c not in chosen]

        def used_bonus() -> int:
            return sum(min(sum(1 for c in chosen if pred(c)), cnt)
                       for pred, cnt in bonus_specs)

        def can_add(cid: int) -> bool:
            for pred, cnt in bonus_specs:
                if pred(cid) and sum(1 for c in chosen if pred(c)) < cnt:
                    return True
            return len(chosen) - used_bonus() < base

        while remaining:
            eligible = [c for c in remaining if can_add(c)]
            if not eligible:
                break
            choices: list[Choice] = []
            if len(chosen) >= min_required:
                choices.append(Choice("done", value=None, kind="done"))
            choices += [Choice(label(c), value=c, kind="pick", card_id=c) for c in eligible]
            pick = yield DecisionNode(node_type, side, choices)
            if pick.value is None:
                break
            chosen.append(pick.value)
            remaining.remove(pick.value)
        return chosen

    def _pick_sacrifices(self, side: Side) -> Iterator[DecisionNode]:
        eng = self.engine
        hand = [c.card_id for c in eng.state.player(side).enablers_in_hand()]
        chosen: list[int] = []
        while len(chosen) < len(hand):
            avail = [cid for cid in hand if cid not in chosen]
            choices = [Choice("done bidding", value=None, kind="done")]
            choices += [Choice(f"sacrifice {ENABLER_REGISTRY[cid].name} (+1)",
                               value=cid, kind="pick", card_id=cid) for cid in avail]
            pick = yield DecisionNode(NodeType.BID_SACRIFICE, side, choices)
            if pick.value is None:
                break
            chosen.append(pick.value)
        return chosen

    # -- player-turn sub-actions --------------------------------------------

    def _apply_turn_action(self, side: Side, action: LegalAction) -> Iterator[DecisionNode]:
        eng = self.engine
        opp = board.opponent(side)
        try:
            if action.kind == "pass":
                eng.end_turn(side)   # pass if nothing was done, else concludes the turn
            elif action.kind == "activate":
                squad = eng.state.player(side).squadrons[action.card_id]
                band = yield from self._pick_spawn_band(side, squad.token_type)
                eng.activate_squadron(side, action.card_id, band)
            elif action.kind == "relaunch":
                eng.relaunch_fighter(side, action.token_uid)   # rolls D4 internally
            elif action.kind == "move":
                eng.move(side, action.token_uid, action.dest_band)
            elif action.kind == "acquire":
                acquirer = eng.state.get_token(action.token_uid)
                eng.acquire(side, action.token_uid, action.target_uid)
                # A UAS acquire roll (hit or miss) lets the owner play UAS
                # Proliferation for a re-roll / extra acquisition.
                if acquirer is not None and acquirer.score_type == TokenScoreType.UAS:
                    yield from self._response_window(
                        side, {EnablerTrigger.OWN_UAS_ACQUIRE_ROLL})
            elif action.kind == "shoot_air":
                yield from self._shoot_air(side, action)
            elif action.kind == "shoot_surface":
                yield from self._shoot_surface(side, action)
            elif action.kind == "play_enabler":
                yield from self._play_enabler_turn(side, action.card_id)
        except IllegalAction:
            # A default-parameter play that turned out inapplicable (autoplay
            # parity): end the turn safely so the game always progresses.
            if eng.state.phase == Phase.PLAYER_TURN and not eng.state.game_over \
                    and eng.state.active_side == side:
                eng.end_turn(side)

    def _shoot_air(self, side: Side, action: LegalAction) -> Iterator[DecisionNode]:
        eng = self.engine
        opp = board.opponent(side)
        attacker = eng.state.get_token(action.token_uid)
        target = eng.state.get_token(action.target_uid)
        # Missile Defense (FAQ): the defender may activate a naval MD covering the
        # air-to-air shot's WEZ (ADA coverage is automatic inside the engine); the
        # attacker may then cancel a declared MD (Air Launched Decoy).
        md_uid = yield from self._declare_md(opp, attacker.location, side, target.location)
        if md_uid is not None:
            yield from self._response_window(
                side, {EnablerTrigger.OPP_DECLARES_MISSILE_DEFENSE}, md_context=True)
        r = eng.shoot_air(side, action.token_uid, action.target_uid, missile_defense_uid=md_uid)
        if r.hit:
            triggers = {EnablerTrigger.OPP_ROLLS_HIT, EnablerTrigger.OWN_AIRCRAFT_LOST}
            # Reserves: only when this hit emptied a squadron of tokens.
            if eng.reserves_playable(opp):
                triggers.add(EnablerTrigger.OWN_SQUADRON_LOST_ALL_TOKENS)
            yield from self._response_window(opp, triggers)

    def _shoot_surface(self, side: Side, action: LegalAction) -> Iterator[DecisionNode]:
        eng = self.engine
        opp = board.opponent(side)
        attacker = eng.state.get_token(action.token_uid)
        target_band = action.target_band or eng.state.get_token(action.target_uid).location
        md_uid = yield from self._declare_md(opp, attacker.location, side, target_band)
        if md_uid is not None:
            yield from self._response_window(
                side, {EnablerTrigger.OPP_DECLARES_MISSILE_DEFENSE}, md_context=True)
        yield from self._response_window(side, {EnablerTrigger.OWN_AIR_TO_SURFACE_DECLARED})

        if action.target_band is not None:   # base strike: allocate point-by-point
            lost_before = len(eng.rapid_resupply_targets(opp))
            r = eng.shoot_surface(side, action.token_uid, target_band=action.target_band,
                                  missile_defense_uid=md_uid, defer_allocation=True)
            # HIT phase: Air Launched Decoy cancels the HIT here, BEFORE the
            # attacker allocates any damage — the defender must decide without
            # seeing the damage rolled or how it will be allocated. (ALD is the
            # only OPP_ROLLS_HIT card; a cancel voids the pending allocation.)
            if r.hit:
                yield from self._response_window(opp, {EnablerTrigger.OPP_ROLLS_HIT})
            # DAMAGE phase: allocate (only if the hit stood) and then offer the
            # damage-based responses — base-damage cancels (Red Horse / Resilient
            # Bases), Reserves, Rapid Resupply — which DO see the outcome.
            if eng._pending_allocation is not None:
                yield from self._allocate(side, eng._pending_allocation)
                triggers = {EnablerTrigger.OPP_ATTACKS_BASE}
                if eng.reserves_playable(opp):
                    triggers.add(EnablerTrigger.OWN_SQUADRON_LOST_ALL_TOKENS)
                triggers |= self._card_loss_trigger(opp, lost_before)
                yield from self._response_window(opp, triggers)
        else:
            r = eng.shoot_surface(side, action.token_uid, target_uid=action.target_uid,
                                  missile_defense_uid=md_uid)
            if r.hit:
                yield from self._response_window(opp, {EnablerTrigger.OPP_ROLLS_HIT})

    def _play_enabler_turn(self, side: Side, card_id: int) -> Iterator[DecisionNode]:
        eng = self.engine
        opp = board.opponent(side)
        play = yield from self._build_enabler_play(side, card_id)
        if card_id in MD_ELIGIBLE_STRIKE_CARDS:
            # Strike launched from the attacker's own base area at the enemy base.
            md_uid = yield from self._declare_md(
                opp, board.own_airbase(side), side, board.own_airbase(opp))
            play.missile_defense_uid = md_uid
            if md_uid is not None:
                yield from self._response_window(
                    side, {EnablerTrigger.OPP_DECLARES_MISSILE_DEFENSE}, md_context=True)
        lost_before = len(eng.rapid_resupply_targets(opp))
        r = eng.play_enabler(side, card_id, play)
        if eng._pending_allocation is not None:
            # A base-strike card left its damage unapplied: spend it point by
            # point, exactly like a token base strike.
            yield from self._allocate(side, eng._pending_allocation)
        triggers = card_play_triggers(ENABLER_REGISTRY[card_id])
        if r.base_damage:
            triggers.add(EnablerTrigger.OPP_ATTACKS_BASE)
        if r.destroyed:
            undo = eng._last_attack_undo
            if undo is not None and any(not t.is_naval for t in undo.revived_tokens):
                triggers.add(EnablerTrigger.OWN_AIRCRAFT_LOST)
        if eng.reserves_playable(opp):
            triggers.add(EnablerTrigger.OWN_SQUADRON_LOST_ALL_TOKENS)
        # A strike that killed a Squadron Card, or a cyber card that discarded
        # enablers, lets the victim answer with Rapid Resupply right now.
        triggers |= self._card_loss_trigger(opp, lost_before)
        yield from self._response_window(opp, triggers)

    def _build_enabler_play(self, side: Side, card_id: int,
                            response: bool = False) -> Iterator[DecisionNode]:
        eng = self.engine
        profile = ENABLER_REGISTRY[card_id]
        play = eng._default_enabler_play(side, card_id)
        # "Base"-worded strikes defer so the attacker distributes damage point by
        # point; "Airbase"-worded strikes (HIMARS/Tomahawk/SOF) hit only the VP
        # boxes with no allocation choice, so they never defer.
        play.defer_allocation = (card_id in BASE_STRIKE_CARDS
                                 and card_id not in AIRBASE_ONLY_STRIKE_CARDS)
        if profile.generates_token is not None:
            play.spawn_band = yield from self._pick_spawn_band(side, profile.generates_token)
        if card_id in AERIAL_REFUEL_CARDS:
            # "Place ONE squadron card in addition to the Posture limit" — the
            # owner chooses which card (reuses SQUADRON_PICK so the env's node
            # space is unchanged).
            cands = eng.aerial_refuel_candidates(side)
            if cands:
                pick = yield DecisionNode(NodeType.SQUADRON_PICK, side, [
                    Choice(f"Aerial Refueling: place {SQUADRON_REGISTRY[c].name}",
                           value=c, kind="pick", card_id=c) for c in cands])
                play.target_squadron_id = pick.value
        if card_id in CYBER_ACQUIRE_CARDS:
            # Acquires one token per point of Cyber Rate — the owner picks which.
            play.target_uids = yield from self._pick_tokens(
                side, eng.state.player(side).cyber_rate,
                lambda: eng.acquirable_enemy_tokens(side), "acquire", profile.name)
        if card_id in ROLL_ACQUIRE_CARDS:
            # The count is a D4. Roll it HERE so the owner can choose that many
            # targets; the handler reuses the roll instead of rolling again.
            play.rolled_count = eng._d4()
            play.target_uids = yield from self._pick_tokens(
                side, play.rolled_count,
                lambda: eng.acquirable_enemy_tokens(side), "acquire", profile.name)
        if card_id in RAPID_RESUPPLY_CARDS:
            # "Recover any discarded Squadron or Enabler Card" — the owner picks
            # which of their lost cards comes back.
            cands = eng.rapid_resupply_targets(side)
            if cands:
                pick = yield DecisionNode(NodeType.SQUADRON_PICK, side, [
                    Choice(f"Rapid Resupply: recover "
                           f"{(SQUADRON_REGISTRY.get(c) or ENABLER_REGISTRY[c]).name}",
                           value=c, kind="pick", card_id=c) for c in cands])
                play.target_squadron_id = pick.value
        if card_id in CONSTELLATION_CARDS:
            # "Return any spent PLASSF card to the deck" — the owner picks which.
            cands = eng.reconstitutable_plassf(side)
            if cands:
                pick = yield DecisionNode(NodeType.ENABLER_PICK, side, [
                    Choice(f"{profile.name}: return {ENABLER_REGISTRY[c].name}",
                           value=c, kind="pick", card_id=c) for c in cands])
                play.target_squadron_id = pick.value
        if card_id in INFANTRY_BATTALION_CARDS:
            # "Place on any base or contingency location" — the owner picks.
            bands = eng.infantry_battalion_bases(side)
            if len(bands) > 1:
                pick = yield DecisionNode(NodeType.SQUADRON_BASE, side, [
                    Choice(f"{profile.name}: place on {b.name}", value=b,
                           kind="pick", band=b) for b in bands])
                play.target_band = pick.value
            elif bands:
                play.target_band = bands[0]
        if card_id in FLYING_CREW_CHIEF_CARDS:
            # "Max aircraft for 1 Squadron at a contingency location" — the
            # owner names which CL squadron gets it.
            cands = eng.flying_crew_chief_candidates(side)
            if cands:
                pick = yield DecisionNode(NodeType.SQUADRON_PICK, side, [
                    Choice(f"Flying Crew Chief: {SQUADRON_REGISTRY[c].name} "
                           f"generates max (no roll)", value=c, kind="pick", card_id=c)
                    for c in cands])
                play.target_squadron_id = pick.value
        if card_id in MUNITIONS_UPGRADE_CARDS:
            # "One fighter squadron may shoot air-to-air at range 4" — the owner
            # names which of their fighter squadrons gets it.
            cands = eng.munitions_upgrade_candidates(side)
            if cands:
                pick = yield DecisionNode(NodeType.SQUADRON_PICK, side, [
                    Choice(f"{profile.name}: {SQUADRON_REGISTRY[c].name} shoots "
                           f"air-to-air at range 4", value=c, kind="pick", card_id=c)
                    for c in cands])
                play.target_squadron_id = pick.value
        branches = ENABLER_CHOICE_BRANCHES.get(card_id)
        if card_id in SUBMARINE_STRIKE_CARDS:
            # The branch is fixed by context (user ruling 2026-07-14): CANCEL
            # only as a response to the opponent's submarine card, ATTACK on
            # the owner's own turn — no choice to present.
            play.choice = "cancel" if response else "attack"
        elif card_id in DEFENSIVE_CYBER_CARDS:
            # Same treatment (user ruling 2026-07-17): CANCEL only as a response
            # to the opponent's cyber card, DEGRADE on the owner's own turn.
            play.choice = "cancel" if response else "degrade"
        elif branches:
            pick = yield DecisionNode(NodeType.ENABLER_BRANCH, side, [
                Choice(f"{profile.name}: {b}", value=b, kind="pick",
                       card_id=card_id, branch_idx=i)
                for i, b in enumerate(branches)])
            play.choice = pick.value

        # Targets that depend on the branch just chosen must come AFTER it.
        if card_id in OFFENSIVE_CYBER_CARDS:
            n = eng.state.player(side).cyber_rate
            if play.choice == "discard":
                # The VICTIM decides which of their own cards to give up, so
                # this node belongs to them (their hand is theirs to see).
                victim = board.opponent(side)
                picked: list[int] = []
                for _ in range(n):
                    hand = [c.card_id for c in eng.state.player(victim).enablers_in_hand()
                            if c.card_id not in picked]
                    if not hand:
                        break
                    pick = yield DecisionNode(NodeType.ENABLER_PICK, victim, [
                        Choice(f"Discard {ENABLER_REGISTRY[c].name}", value=c,
                               kind="pick", card_id=c) for c in hand])
                    picked.append(pick.value)
                play.discard_card_ids = picked
            else:
                # The PLAYER picks which enemy tokens to destroy.
                play.target_uids = yield from self._pick_tokens(
                    side, n, lambda: eng.removable_enemy_tokens(side),
                    "destroy", profile.name)
        if card_id in SHIP_KILL_CARDS and play.choice != "cancel":
            # Auto-hit: the owner picks WHICH enemy surface combatant dies.
            # (43/89 only on their attack branch, hence after the branch above.)
            ships = eng.enemy_ship_targets(side)
            if len(ships) > 1:
                pick = yield DecisionNode(NodeType.TURN_ACTION, side, [
                    Choice(f"{profile.name}: sink {self._token_label(u, side)}",
                           value=u, kind="shoot_surface", target_uid=u) for u in ships])
                play.target_uids = [pick.value]
            elif ships:
                play.target_uids = [ships[0]]
        if card_id in JOINT_OFFENSIVE_CYBER_CARDS:
            # Removes 2x Cyber Rate enemy tokens — the owner picks which.
            play.target_uids = yield from self._pick_tokens(
                side, eng.state.player(side).cyber_rate * 2,
                lambda: eng.removable_enemy_tokens(side), "destroy", profile.name)
        if card_id in COUNTER_UAS_CARDS:
            # Destroys Cyber-Rate enemy UAS tokens — the owner picks which UAS.
            play.target_uids = yield from self._pick_tokens(
                side, eng.state.player(side).cyber_rate,
                lambda: eng.removable_enemy_uas(side), "destroy", profile.name)
        if card_id in SPECIAL_MISSION_CARDS and play.choice == "acquire":
            # Roll-then-choose on the acquire branch: roll the D4 here so the
            # owner picks that many targets (the handler reuses the roll). The
            # advantage branch takes no targets.
            play.rolled_count = eng._d4()
            play.target_uids = yield from self._pick_tokens(
                side, play.rolled_count,
                lambda: eng.acquirable_enemy_tokens(side), "acquire", profile.name)
        if card_id in UAS_PROLIFERATION_CARDS:
            # One extra acquisition attempt — the owner picks which enemy token
            # the UAS rolls against (fog-safe; the just-failed target is still in
            # the pool, so a re-roll is a valid choice).
            picks = yield from self._pick_tokens(
                side, 1, lambda: eng.acquirable_enemy_tokens(side), "acquire", profile.name)
            if picks:
                play.target_uids = picks
        return play

    def _pick_spawn_band(self, side: Side, token_type: TokenType) -> Iterator[DecisionNode]:
        eng = self.engine
        player = eng.state.player(side)
        legal = board.valid_spawn_locations(token_type, side,
                                            player.posture_type or PostureType.STANDARD)
        if player.cl_banned_campaign:
            legal = legal - {BandID.US_CONTINGENCY_LOCATION}
        bands = sorted(legal, key=lambda b: b.name)
        if len(bands) <= 1:
            return bands[0] if bands else None
        pick = yield DecisionNode(NodeType.SPAWN_BAND, side, [
            Choice(b.name, value=b, kind="pick", band=b) for b in bands])
        return pick.value

    def _declare_md(self, defender: Side, attacker_loc: BandID, attacker_side: Side,
                    target_band: BandID) -> Iterator[DecisionNode]:
        eng = self.engine
        elig = eng.eligible_missile_defenders(defender, attacker_loc, attacker_side, target_band)
        if not elig:
            return None
        choices = [Choice("no missile defense", value=None, kind="skip")]
        choices += [Choice(f"MD with {t.token_type.value}#{t.uid}", value=t.uid,
                           kind="pick", actor_uid=t.uid, band=t.location) for t in elig]
        pick = yield DecisionNode(NodeType.MD_DECLARE, defender, choices)
        return pick.value

    def _token_label(self, uid: int, viewer: Side) -> str:
        """
        Fog-safe label for a token in a choice: an enemy token that has not been
        acquired shows only what is public (position + acquisition value), never
        its type — the same masking the board uses.
        """
        t = self.engine.state.get_token(uid)
        if t is None:
            return f"#{uid}"
        if t.side == viewer or t.acquired:
            return f"{t.token_type.value}#{uid} in {t.location.name}"
        return f"?#{uid}(av{t.profile.acquisition_value}) in {t.location.name}"

    def _pick_tokens(self, side: Side, n: int, pool, verb: str,
                     card_name: str) -> Iterator[DecisionNode]:
        """
        Let `side` choose `n` enemy tokens, one node each, for a card that
        acquires or destroys that many. `pool` is re-evaluated per pick so a
        token can only be chosen once. Labels go through `_token_label`, so an
        unacquired enemy never leaks its type — fog survives the choice.
        """
        kind = "acquire" if verb == "acquire" else "shoot_air"
        chosen: list[int] = []
        for _ in range(n):
            avail = [u for u in pool() if u not in chosen]
            if not avail:
                break
            pick = yield DecisionNode(NodeType.TURN_ACTION, side, [
                Choice(f"{card_name}: {verb} {self._token_label(u, side)}",
                       value=u, kind=kind, target_uid=u) for u in avail])
            chosen.append(pick.value)
        return chosen

    def _card_loss_trigger(self, side: Side, before: int) -> set:
        """
        OWN_CARD_DISCARDED if `side` lost a Squadron/Enabler Card during the
        action just resolved (`before` = its recoverable count beforehand).
        Comparing counts keeps Rapid Resupply an IMMEDIATE response: a card
        discarded on an earlier turn no longer re-opens the window.
        """
        if len(self.engine.rapid_resupply_targets(side)) > before:
            return {EnablerTrigger.OWN_CARD_DISCARDED}
        return set()

    def _response_window(self, side: Side, triggers: set,
                         md_context: bool = False) -> Iterator[DecisionNode]:
        eng = self.engine
        if eng.state.game_over:
            return
        hand = eng.legal_responses(side, triggers)
        if not hand:
            return
        choices = [Choice("no response", value=None, kind="skip")]
        choices += [Choice(f"respond: {ENABLER_REGISTRY[cid].name}", value=cid,
                           kind="pick", card_id=cid) for cid in hand]
        pick = yield DecisionNode(NodeType.RESPONSE, side, choices)
        if pick.value is None:
            return
        card_id = pick.value
        play = yield from self._build_enabler_play(side, card_id, response=True)
        if md_context:
            play.choice = "cancel_md"   # dual-purpose cards take their cancel-MD branch
        try:
            eng.play_enabler(side, card_id, play, response=True)
        except IllegalAction:
            return   # response turned out inapplicable; window simply closes
        # A response card is itself a card play the OTHER side may answer — e.g.
        # Anti-Access/Area Denial cancels a US mobility/maintenance card that was
        # itself played as a response, and Counter Space answers a space card
        # played in response. Open a follow-up window gated to the just-played
        # card's own play-triggers (empty for most cards, so this usually ends
        # immediately; the hand drains, so the chain always terminates).
        counter = card_play_triggers(ENABLER_REGISTRY[card_id])
        if counter:
            yield from self._response_window(board.opponent(side), counter)

    def _allocate(self, side: Side, pending: PendingBaseAllocation) -> Iterator[DecisionNode]:
        """
        Distribute rolled base-strike damage one point at a time. Targets are
        Squadron CARDS (grounded flights die with their card), cardless tokens
        (e.g. US ADA), the Infantry Battalion, and airbase VP boxes. At a
        Contingency Location there is no spillover: the first pick locks the
        single target and excess damage is lost. Face-down (never-activated)
        enemy cards are presented without their identity.
        """
        eng = self.engine
        defender = eng.state.player(pending.target_side)
        at_cl = pending.target_band == BandID.US_CONTINGENCY_LOCATION
        counts: dict[tuple, int] = {}
        locked: Optional[tuple] = None   # CL: single target, no spillover
        allocation: list[tuple] = []

        def capacity(kind: str, ref) -> int:
            if kind == "squadron":
                squad = defender.squadrons.get(ref)
                return 0 if squad is None else DAMAGE_TO_DESTROY_SQUADRON - squad.damage
            if kind == "vp":
                return AIRBASE_BONUS_DAMAGE_BOXES - defender.airbase_vp_damage
            if kind == "token":
                tok = eng.state.get_token(ref)
                if tok is None or tok.destroyed:
                    return 0
            return 1   # token / infantry die to a single point

        def is_facedown(kind: str, ref) -> bool:
            if kind != "squadron":
                return False
            squad = defender.squadrons.get(ref)
            return squad is not None and not squad.ever_activated and not squad.is_destroyed

        def describe(kind: str, ref) -> str:
            if kind == "squadron":
                if is_facedown(kind, ref):
                    return "face-down Squadron Card"
                return f"squadron {SQUADRON_REGISTRY[ref].name}"
            if kind == "token":
                tok = eng.state.get_token(ref)
                return f"token {tok.token_type.value}#{ref}" if tok else f"token #{ref}"
            return "Infantry Battalion" if kind == "infantry" else "airbase VP boxes"

        for _ in range(pending.amount):
            choices = []
            for kind, ref in pending.targets:
                if at_cl and locked is not None and (kind, ref) != locked:
                    continue
                if counts.get((kind, ref), 0) >= capacity(kind, ref):
                    continue
                hidden = is_facedown(kind, ref)
                choices.append(Choice(describe(kind, ref), value=(kind, ref), kind="pick",
                                      alloc_kind=kind,
                                      card_id=None if hidden else
                                      (ref if kind == "squadron" else None),
                                      target_uid=ref if kind == "token" else None))
            if not choices:
                break   # every target saturated; excess damage is lost
            pick = yield DecisionNode(NodeType.ALLOC_POINT, side, choices)
            kind, ref = pick.value
            counts[(kind, ref)] = counts.get((kind, ref), 0) + 1
            allocation.append((kind, ref, 1))
            if at_cl and locked is None:
                locked = (kind, ref)
        eng.resolve_base_allocation(allocation)
