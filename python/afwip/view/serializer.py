"""
serializer.py — Engine state (+ current decision node) -> view models.

All fog decisions route through afwip.view.fog and are applied HERE, before
any model is built: the returned objects contain only what `viewer` is allowed
to see, so downstream code (HTTP layer, browser) can never leak hidden state.

`viewer=None` produces the PUBLIC view (both sides masked — hotseat handoff
screen); `reveal=True` the omniscient spectator/debug view.
"""

from __future__ import annotations

from typing import Optional

from afwip.core.rules import RulesEngine
from afwip.core.state import CardZone, TokenInstance
from afwip.core.constants import BandID, Side
from afwip.core.cards import (
    ENABLER_REGISTRY, MISSION_REGISTRY, POSTURE_REGISTRY, SQUADRON_REGISTRY,
)
from afwip.script import DecisionNode
from afwip.view import fog
from afwip.view.layout import BOARD_BANDS, choice_category, node_prompt
from afwip.core import board as board_geom
from afwip.view.models import (
    BandView, BoardView, CaptureEntryView, CapturesPanelView, ChoiceView,
    DecisionView, HandCardView, HandView, SquadronPanelView, SquadronView,
    StatusView, TokenView,
)


def token_view(t: TokenInstance, viewer: Optional[Side], reveal: bool) -> TokenView:
    known = fog.token_known(t, viewer, reveal)
    av = t.profile.acquisition_value
    if not known:
        # Position and AV are public; identity and status flags are not.
        return TokenView(uid=t.uid, side=t.side.value, band=t.location.name,
                         fogged=True, av=av, acquired=False,
                         label=f"?#{t.uid}(av{av})")
    flags = ("*" if t.acquired else "") + ("W" if t.is_winchester else "") \
        + ("g" if t.grounded else "")
    return TokenView(uid=t.uid, side=t.side.value, band=t.location.name,
                     fogged=False, av=av, acquired=t.acquired,
                     label=f"{t.token_type.value}#{t.uid}{flags}",
                     type=t.token_type.value,
                     winchester=t.is_winchester, grounded=t.grounded)


def board_view(eng: RulesEngine, viewer: Optional[Side], reveal: bool) -> BoardView:
    gs = eng.state
    bands = []
    for band, header in BOARD_BANDS:
        toks = []
        for side in (Side.US, Side.PRC):
            for t in sorted(gs.player(side).tokens_at(band), key=lambda t: t.uid):
                toks.append(token_view(t, viewer, reveal))
        bands.append(BandView(band=band.name, header=header, tokens=toks))
    return BoardView(bands=bands)


def _squadron_status(s) -> str:
    if s.is_destroyed:
        return "destroyed"
    if s.activated:
        return "active"
    if s.zone == CardZone.SELECTED:
        return "ready"
    return "out"


def squadron_panel(eng: RulesEngine, side: Side, viewer: Optional[Side],
                   reveal: bool) -> SquadronPanelView:
    p = eng.state.player(side)
    rows, hidden = [], 0
    for cid, s in sorted(p.squadrons.items()):
        if not fog.squadron_known(s, viewer, reveal):
            if s.zone == CardZone.SELECTED:
                hidden += 1
            continue
        rows.append(SquadronView(
            card_id=cid, name=SQUADRON_REGISTRY[cid].name,
            token_type=s.token_type.value, status=_squadron_status(s),
            damage=s.damage, tokens_lost=s.tokens_lost,
            at_contingency_location=s.location == BandID.US_CONTINGENCY_LOCATION,
            grounded_tokens=len(s.grounded_token_uids)))
    off = sum(1 for t in p.tokens.values() if t.off_board)
    return SquadronPanelView(side=side.value, squadrons=rows,
                             face_down=hidden, off_board_naval=off)


def hand_view(eng: RulesEngine, side: Side, viewer: Optional[Side],
              reveal: bool) -> HandView:
    p = eng.state.player(side)
    cards, hidden = [], 0
    for c in p.enablers_in_hand():
        if fog.hand_card_known(c, viewer, reveal):
            cards.append(HandCardView(card_id=c.card_id,
                                      name=ENABLER_REGISTRY[c.card_id].name,
                                      revealed=c.revealed_to_opponent,
                                      single_use=ENABLER_REGISTRY[c.card_id].single_use))
        else:
            hidden += 1
    # Spent (played) enablers are public — playing a card is a visible act — so
    # they're shown for both sides regardless of the viewer. PLAYED = one-shot
    # spent this cycle; ACTIVE = an enduring card in effect; REMOVED = single-use
    # burned for the campaign.
    spent = [
        HandCardView(card_id=cid, name=ENABLER_REGISTRY[cid].name,
                     revealed=c.revealed_to_opponent,
                     single_use=ENABLER_REGISTRY[cid].single_use)
        for cid, c in sorted(p.enablers.items())
        if c.zone in (CardZone.PLAYED, CardZone.ACTIVE, CardZone.REMOVED)
    ]
    return HandView(side=side.value, cards=cards, hidden_count=hidden,
                    spent_count=len(spent), spent=spent)


def status_view(eng: RulesEngine) -> StatusView:
    gs = eng.state

    def name(reg, cid):
        return reg[cid].name if cid is not None else None

    return StatusView(
        campaign=gs.campaign, ato_cycle=gs.ato_cycle,
        total_ato_cycles=gs.total_ato_cycles, turn_number=gs.turn_number,
        phase=gs.phase.value, active_side=gs.active_side.value,
        initiative=gs.initiative_holder.value if gs.initiative_holder else None,
        vp={s.value: eng.total_victory_points(s) for s in Side},
        cyber={s.value: gs.player(s).cyber_rate for s in Side},
        # Missions/postures are treated as public (established obs judgment).
        missions={s.value: name(MISSION_REGISTRY, gs.player(s).mission_card_id)
                  for s in Side},
        postures={s.value: name(POSTURE_REGISTRY, gs.player(s).posture_card_id)
                  for s in Side},
        # Base strikes are public acts; the VP damage boxes sit on the map.
        base_damage={s.value: gs.player(s).airbase_vp_damage for s in Side},
        # Board-marker info (public): intel track follows the initiative winner.
        intel={s.value: gs.player(s).intel_track.value for s in Side},
    )


def captures_panel(eng: RulesEngine, side: Side) -> CapturesPanelView:
    """This side's score pile: enemy units it shot down (public info)."""
    p = eng.state.player(side)
    victim = board_geom.opponent(side).value
    entries = []
    for c in p.captures:
        vp = eng.capture_value(side, c)
        if c.is_squadron_card:
            name = SQUADRON_REGISTRY[c.card_id].name if c.card_id else "squadron"
            entries.append(CaptureEntryView(
                kind="squadron", label=name, vp=vp, victim_side=victim,
                on_ground=c.destroyed_on_ground, card_id=c.card_id,
                ato_cycle=c.ato_cycle))
        else:
            tt = c.token_type.value if c.token_type else "token"
            entries.append(CaptureEntryView(
                kind="token", label=f"{tt}#{c.uid}" if c.uid else tt, vp=vp,
                victim_side=victim, on_ground=c.destroyed_on_ground,
                token_type=tt, uid=c.uid, ato_cycle=c.ato_cycle))
    return CapturesPanelView(side=side.value, entries=entries)


def decision_view(node: DecisionNode) -> DecisionView:
    """The current decision, for the DECIDING side's eyes (session routes it)."""
    return DecisionView(
        node_type=node.node_type.name, side=node.side.value,
        prompt=node_prompt(node.node_type),
        choices=[ChoiceView(
            index=i, label=c.label, kind=c.kind or "pick",
            category=choice_category(c.kind),
            card_id=c.card_id,
            band=c.band.name if c.band is not None else None,
            actor_uid=c.actor_uid, target_uid=c.target_uid,
            branch_idx=c.branch_idx, alloc_kind=c.alloc_kind,
        ) for i, c in enumerate(node.choices)])
