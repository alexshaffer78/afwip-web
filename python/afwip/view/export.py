"""
export.py — Rigid, machine-readable text encoding of a fogged GameView.

Turns the exact `GameView` the browser already holds (fog applied by the
serializer) into a deterministic line-oriented block a student can copy-paste
into an LLM chat alongside the `pme_materials/` docs. Because it consumes the
*already-fogged* view, it can never leak hidden state — the same guarantee the
browser has.

The grammar is intentionally rigid so a model (or a parser) can rely on it; it
is specified field-by-field in `pme_materials/04_state_encoding.md`. Summary:

  - Line-oriented. Each meaningful line is `PREFIX: k=v k=v ...` or a
    `=== SECTION ===` banner. Lines beginning `#` are human legends (ignore).
  - Structured fields are `key=value` with no spaces in the value; free text
    (names, labels, prompts) is double-quoted.
  - A null/absent field is the single character `-`. An unknown (fogged) value
    is `?`. Booleans are `1`/`0`.

The block is stable across polls for the same state, so consecutive pastes read
as an ordered play-by-play (`GAME: ... rev=N` increases each decision).
"""

from __future__ import annotations

from afwip.view.models import GameView


SCHEMA_VERSION = "1"


def _q(text: object) -> str:
    """Quote a free-text field, escaping embedded quotes."""
    return '"' + str(text).replace('"', "'") + '"'


def _v(value: object) -> str:
    """A structured (unquoted) field value; None -> '-'."""
    return "-" if value is None else str(value)


def _uid(value: object) -> str:
    return "-" if value is None else str(value)


def game_export(view: GameView, draft: "dict | None" = None) -> str:
    """
    Render a fogged GameView as the rigid AFWIP-STATE text block.

    `draft` (web session only) carries the drafting side's IN-PROGRESS setup
    picks — posture, squadrons (with placement), and enablers chosen so far —
    which aren't committed to engine state until the whole draft finishes, so
    they'd otherwise be invisible while (e.g.) drafting enablers. It is already
    fog-gated to the drafting side by the session.
    """
    viewer = view.viewer                      # "US" / "PRC" / None (spectator)
    st = view.status
    out: list[str] = []

    # -- header --------------------------------------------------------------
    out.append(f"=== AFWIP STATE v{SCHEMA_VERSION} ===")
    out.append(
        f"GAME: id={view.game_id} rev={view.revision} mode={view.mode} "
        f"terminal={1 if view.terminal else 0}"
    )
    out.append(
        f"CAMPAIGN: num={st.campaign} name={_q(_campaign_name(st.campaign))} "
        f"ato={st.ato_cycle}/{st.total_ato_cycles} turn={st.turn_number} "
        f"phase={st.phase} active={st.active_side} initiative={_v(st.initiative)}"
    )
    out.append(f"VIEWER: side={_v(viewer)}")
    for s in ("US", "PRC"):
        # base_vp = VP this side has earned from base strikes on the ENEMY
        # airbase (+1 per damage box) — already included live in vp above.
        enemy = "PRC" if s == "US" else "US"
        base_vp = st.base_damage.get(enemy, 0)
        out.append(
            f"SIDE {s}: mission={_v(st.missions.get(s))} "
            f"posture={_v(st.postures.get(s))} cyber={st.cyber.get(s, 0)} "
            f"vp={st.vp.get(s, 0)} base_vp={base_vp} intel={_v(st.intel.get(s))} "
            f"airbase_dmg={st.base_damage.get(s, 0)}/3"
        )

    # -- board ---------------------------------------------------------------
    out.append("")
    out.append("=== BOARD === (bands left->right: US rear .. front .. PRC rear; "
               "US advances toward BAND_E, PRC toward BAND_A)")
    out.append("# TOKEN fields: band owner uid type av acquired winchester grounded")
    any_token = False
    for band in view.board.bands:
        for t in band.tokens:
            any_token = True
            out.append(
                f"TOKEN: band={t.band} owner={t.side} uid={t.uid} "
                f"type={_q(t.type if t.type is not None else '?')} av={t.av} "
                f"acquired={1 if t.acquired else 0} "
                f"winchester={_bool_or_unknown(t.winchester, t.fogged)} "
                f"grounded={_bool_or_unknown(t.grounded, t.fogged)}"
            )
    if not any_token:
        out.append("# (no tokens on the board)")

    # -- squadrons -----------------------------------------------------------
    out.append("")
    out.append("=== SQUADRONS ===")
    out.append("# SQUADRON fields: owner card name token status damage tokens_lost loc grounded_tokens")
    for panel in view.squadrons:
        for sq in panel.squadrons:
            loc = "CL" if sq.at_contingency_location else "AIRBASE"
            out.append(
                f"SQUADRON: owner={panel.side} card={sq.card_id} name={_q(sq.name)} "
                f"token={_q(sq.token_type)} status={sq.status} damage={sq.damage}/2 "
                f"tokens_lost={sq.tokens_lost} loc={loc} "
                f"grounded_tokens={sq.grounded_tokens}"
            )
        if panel.face_down:
            out.append(f"FACEDOWN_SQUADRONS: side={panel.side} count={panel.face_down}")
        if panel.off_board_naval:
            out.append(f"OFFBOARD_NAVAL: side={panel.side} count={panel.off_board_naval}")

    # -- hands ---------------------------------------------------------------
    out.append("")
    out.append("=== HANDS ===")
    out.append("# HAND fields: owner card name use revealed_to_opponent")
    out.append("# use=single (spent for the campaign once played) or multi "
               "(returns to hand each ATO).")
    out.append("# SPENT = enablers already played (public for both sides)")
    for hand in view.hands:
        for c in hand.cards:
            out.append(
                f"HAND: owner={hand.side} card={c.card_id} name={_q(c.name)} "
                f"use={'single' if c.single_use else 'multi'} "
                f"revealed_to_opponent={1 if c.revealed else 0}"
            )
        if hand.hidden_count or (viewer is not None and hand.side != viewer):
            out.append(
                f"HAND_HIDDEN: side={hand.side} count={hand.hidden_count}"
            )
        for c in hand.spent:
            out.append(
                f"SPENT: owner={hand.side} card={c.card_id} name={_q(c.name)}"
            )

    # -- captures ------------------------------------------------------------
    out.append("")
    out.append("=== CAPTURES === (owner = the side that destroyed the unit; "
               "vp = its value under the owner's mission)")
    out.append("# CAPTURE fields: owner kind label vp on_ground")
    any_capture = False
    for pile in view.captures:
        for e in pile.entries:
            any_capture = True
            out.append(
                f"CAPTURE: owner={pile.side} kind={e.kind} label={_q(e.label)} "
                f"vp={e.vp} on_ground={1 if e.on_ground else 0}"
            )
    if not any_capture:
        out.append("# (nothing destroyed yet)")

    # -- draft in progress ---------------------------------------------------
    if draft is not None:
        out.append("")
        out.append("=== DRAFT (in progress) === (your setup picks so far this ATO; "
                   "not committed until the draft completes)")
        out.append(f"DRAFT: side={draft['side']} "
                   f"posture={_q(draft['posture']) if draft.get('posture') else '-'}")
        for sq in draft.get("squadrons", []):
            out.append(
                f"DRAFT_SQUADRON: card={sq['card']} name={_q(sq['name'])} "
                f"token={_q(sq['token'])} loc={_v(sq.get('loc'))}"
            )
        for en in draft.get("enablers", []):
            out.append(f"DRAFT_ENABLER: card={en['card']} name={_q(en['name'])}")

    # -- decision / result ---------------------------------------------------
    out.append("")
    if view.terminal:
        out.append("=== RESULT ===")
        out.append(f"WINNER: {_v(view.winner)}")
        out.append(f"FINAL_VP: US={st.vp.get('US', 0)} PRC={st.vp.get('PRC', 0)}")
    elif view.decision is not None:
        d = view.decision
        out.append("=== DECISION ===")
        out.append(f"NODE: type={d.node_type} side={d.side} prompt={_q(d.prompt)}")
        out.append("# ACTION fields: index kind actor target band label "
                   "(actor/target are token uids; '-' = not applicable)")
        for c in d.choices:
            out.append(
                f"ACTION: index={c.index} kind={c.kind} "
                f"actor={_uid(c.actor_uid)} target={_uid(c.target_uid)} "
                f"band={_v(c.band)} label={_q(c.label)}"
            )
    else:
        out.append("=== DECISION ===")
        reason = ("game is over" if view.terminal
                  else "no decision for you right now — waiting on the "
                       f"{st.active_side} side / the agent")
        out.append(f"NONE: reason={_q(reason)}")

    out.append("=== END ===")
    return "\n".join(out)


def _bool_or_unknown(flag: object, fogged: bool) -> str:
    """1/0 for a known flag; '?' when the token is fogged (flag is None)."""
    if flag is None:
        return "?" if fogged else "0"
    return "1" if flag else "0"


# Campaign names are static; kept here so the export is self-describing without
# importing the campaigns registry (which would pull in the whole engine).
_CAMPAIGN_NAMES = {
    1: "Meeting Engagement",
    2: "Tournament",
    3: "Prolonged Combat",
    4: "The World Watches",
    5: "Reserves",
}


def _campaign_name(num: int) -> str:
    return _CAMPAIGN_NAMES.get(num, "Free Play")
