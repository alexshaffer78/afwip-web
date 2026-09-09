"""
events.py — Session-local event capture and structured scoring reports.

`EventRecorder` wraps ONE RulesEngine instance: it taps that instance's dice
(instance-attribute patch, nothing global), snapshots state around each
decision, and turns the diff into structured `EventView`s with both a full
message (owner/omniscient) and a fog-safe redaction for the other side. The
diff semantics mirror afwip.tui's event log (destroyed vs recycled vs
off-board — user bug 2026-07-15) without importing or modifying the TUI.

`score_report(...)` is the structured equivalent of the TUI's end-of-ATO
scoring block: per-side captures by name with mission values plus the accrued
`vp_log` audit trail.
"""

from __future__ import annotations

from typing import Optional

from afwip.core.rules import RulesEngine, RollMode
from afwip.core.constants import Side
from afwip.core.cards import MISSION_REGISTRY, SQUADRON_REGISTRY
from afwip.script import DecisionNode, NodeType, Choice
from afwip.view.models import (
    AccruedLineView, CaptureLineView, EventView, ScoreReportView, SideScoreView,
)


# Decision nodes whose full choice label is public information (physical acts
# both players watch: initiative, missile defense, responses, damage allocation)
# or treated as public by the established obs judgment (missions/postures).
PUBLIC_DECISION_NODES = {
    NodeType.MISSION_PICK, NodeType.POSTURE_PICK, NodeType.FIRST_PLAYER,
    NodeType.MD_DECLARE, NodeType.RESPONSE, NodeType.ALLOC_POINT,
}

# TURN_ACTION kinds whose label is public (played openly on the table).
PUBLIC_ACTION_KINDS = {"pass", "activate", "play_enabler"}


def _fmt_roll(entry) -> str:
    """Format one captured dice entry the way the TUI does."""
    if entry[0] == "d4":
        return f"D4={entry[1]}"
    r = entry[1]
    bonus = f"+{r.value - r.natural}" if r.value != r.natural else ""
    if r.mode == RollMode.NORMAL:
        return f"D4={r.natural}{bonus}"
    dice = ",".join(str(d) for d in r.dice)
    return f"D4={r.natural}{bonus} ({r.mode.value.lower()}: {dice})"


def _roll_data(entry) -> dict:
    """Structured form of one captured die, so the UI can draw die faces:
    the raw dice rolled, the chosen `natural`, any `bonus`, and the final
    `value`, plus advantage/disadvantage mode."""
    if entry[0] == "d4":
        v = entry[1]
        return {"value": v, "natural": v, "bonus": 0, "mode": "NORMAL", "dice": [v]}
    r = entry[1]
    data = {"value": r.value, "natural": r.natural, "bonus": r.value - r.natural,
            "mode": r.mode.name, "dice": list(r.dice)}
    role = entry[2] if len(entry) > 2 else None
    if role:                                   # "hit" / "damage" / "acquire"
        data["role"] = role
    return data


def _setup_event(step: int, entry: dict) -> Optional[EventView]:
    """Turn one engine setup-roll record into a clearly-labeled public event.

    Setup dice (initiative bid, the winner's cyber raise, Play Intel) roll
    automatically with no decision node, so without this they surface only as an
    unlabeled `dice: D4=..` blob. Each returned event carries a `data.label` the
    UI shows beside the die faces (and a plain non-dice event when there were no
    dice, e.g. a tournament auto-assigned initiative)."""
    kind = entry.get("kind")
    if kind == "initiative":
        winner = entry["winner"]
        if entry.get("auto") or "us" not in entry:
            msg = f"Initiative → {winner} (Intel Advantage)"
            return EventView(step=step, type="initiative", side=None,
                             message=msg, public=True, data={"label": msg})
        us, prc = entry["us"], entry["prc"]
        label = (f"Initiative bid — US {us['value']} vs PRC {prc['value']} "
                 f"→ {winner} wins (Intel Advantage)")
        return EventView(step=step, type="dice", side=None, message=label,
                         public=True, data={"label": label, "rolls": [us, prc]})
    if kind == "cyber_raise":
        side = entry["side"]
        outcome = (f"Cyber Rate {entry['from']}→{entry['to']}"
                   if entry["success"] else "no raise")
        label = (f"{side} cyber raise — rolled {entry['natural']} "
                 f"(need {entry['needed']}) → {outcome}")
        die = {"natural": entry["natural"], "value": entry["natural"], "bonus": 0,
               "mode": "NORMAL", "dice": [entry["natural"]]}
        return EventView(step=step, type="dice", side=None, message=label,
                         public=True, data={"label": label, "rolls": [die]})
    if kind == "play_intel":
        us, prc = entry["us"], entry["prc"]
        label = (f"Play Intel — US rolled {us['roll']['value']} (sees {us['sees']}), "
                 f"PRC rolled {prc['roll']['value']} (sees {prc['sees']})")
        return EventView(step=step, type="dice", side=None, message=label,
                         public=True,
                         data={"label": label, "rolls": [us["roll"], prc["roll"]]})
    return None


class EventRecorder:
    """Captures dice and state-diff events for one engine instance."""

    def __init__(self, engine: RulesEngine):
        self.engine = engine
        self.events: list[EventView] = []
        self._dice: list = []
        self._tap_dice()

    # -- dice tap (instance-level, session-local) ---------------------------

    def _tap_dice(self) -> None:
        eng = self.engine
        orig_roll, orig_d4 = eng.roll, eng._d4

        def d4():
            v = orig_d4()
            self._dice.append(("d4", v))
            return v

        def roll(mode=RollMode.NORMAL, bonus=0, note=None):
            start = len(self._dice)
            r = orig_roll(mode, bonus)     # routes through the wrapped d4
            del self._dice[start:]         # fold raw dice into one entry
            self._dice.append(("roll", r, note))
            return r

        eng._d4, eng.roll = d4, roll

    # -- snapshots -----------------------------------------------------------

    def snapshot(self) -> dict:
        d = {}
        for side in (Side.US, Side.PRC):
            p = self.engine.state.player(side)
            d[side] = {
                # Keep TokenInstance refs: a departed token still tells us WHY
                # (destroyed vs recycled at cleanup vs moved off-board).
                "toks": {t.uid: t for t in p.living_tokens()},
                "winch": {t.uid for t in p.living_tokens() if t.is_winchester},
                "ground": {t.uid for t in p.living_tokens() if t.grounded},
                "vp": self.engine.total_victory_points(side),
                "cyber": p.cyber_rate,
                "sq_dmg": {cid: s.damage for cid, s in p.squadrons.items()},
                "sq_dead": {cid for cid, s in p.squadrons.items() if s.is_destroyed},
            }
        return d

    # -- event emission -------------------------------------------------------

    def record_decision(self, step: int, node: DecisionNode, choice: Choice) -> None:
        side = node.side
        public = (node.node_type in PUBLIC_DECISION_NODES
                  or (node.node_type == NodeType.TURN_ACTION
                      and choice.kind in PUBLIC_ACTION_KINDS))
        redacted = None if public else f"{side.value} {node.node_type.name}: (hidden)"
        self.events.append(EventView(
            step=step, type="decision", side=side.value,
            message=f"{side.value} {node.node_type.name}: {choice.label}",
            public=public, redacted=redacted,
            data={"node_type": node.node_type.name, "kind": choice.kind or ""},
        ))

    def flush_setup(self, step: int) -> None:
        """Drain the engine's setup-roll log into labeled events. Call BEFORE
        flush_dice: the same dice were captured by the tap, so this clears them
        to stop flush_dice re-emitting an unlabeled blob for the setup rolls."""
        log = getattr(self.engine, "setup_roll_log", None)
        if not log:
            return
        for entry in log:
            ev = _setup_event(step, entry)
            if ev is not None:
                self.events.append(ev)
        log.clear()
        self._dice.clear()

    def flush_dice(self, step: int) -> None:
        if not self._dice:
            return
        text = "  ".join(_fmt_roll(e) for e in self._dice)
        rolls = [_roll_data(e) for e in self._dice]
        self.events.append(EventView(
            step=step, type="dice", side=None, message=f"dice: {text}",
            public=True, data={"rolls": rolls}))
        self._dice.clear()

    def emit_diff(self, step: int, before: dict) -> None:
        """Diff `before` against the current state and append events."""
        after = self.snapshot()
        for side in (Side.US, Side.PRC):
            b, a, tag = before[side], after[side], side.value
            for uid, tok in b["toks"].items():
                if uid not in a["toks"]:
                    self._token_gone(step, tag, tok)
            for uid, tok in a["toks"].items():
                if uid not in b["toks"]:
                    self.events.append(EventView(
                        step=step, type="token_enters", side=tag,
                        message=f"{tag} {tok.token_type.value}#{uid} enters play",
                        public=False,
                        redacted=f"{tag} token #{uid} enters play",
                        data={"uid": uid}))
            for uid in (a["winch"] - b["winch"]) & set(b["toks"]):
                tok = b["toks"][uid]
                self.events.append(EventView(
                    step=step, type="token_winchester", side=tag,
                    message=f"{tag} {tok.token_type.value}#{uid} is Winchester",
                    public=tok.acquired,
                    redacted=f"{tag} token #{uid} status change",
                    data={"uid": uid}))
            for uid in (a["ground"] - b["ground"]) & set(b["toks"]):
                tok = b["toks"][uid]
                self.events.append(EventView(
                    step=step, type="token_grounded", side=tag,
                    message=(f"{tag} {tok.token_type.value}#{uid} returns to base "
                             f"(done this ATO)"),
                    public=tok.acquired,
                    redacted=f"{tag} token #{uid} status change",
                    data={"uid": uid}))
            for cid in a["sq_dead"] - b["sq_dead"]:
                self.events.append(EventView(
                    step=step, type="squadron_destroyed", side=tag,
                    message=(f"{tag} squadron {cid} "
                             f"({SQUADRON_REGISTRY[cid].name}) DESTROYED"),
                    public=True, data={"card_id": cid}))
            for cid, dmg in a["sq_dmg"].items():
                prev = b["sq_dmg"].get(cid, 0)
                if dmg > prev and cid not in a["sq_dead"] - b["sq_dead"]:
                    self.events.append(EventView(
                        step=step, type="squadron_damage", side=tag,
                        message=f"{tag} squadron {cid} damage {prev}->{dmg}",
                        public=True, data={"card_id": cid, "damage": dmg}))
            if a["vp"] != b["vp"]:
                self.events.append(EventView(
                    step=step, type="vp_change", side=tag,
                    message=f"{tag} VP {b['vp']}->{a['vp']}",
                    public=True, data={"vp": a["vp"]}))
            if a["cyber"] != b["cyber"]:
                self.events.append(EventView(
                    step=step, type="cyber_change", side=tag,
                    message=f"{tag} cyber rate {b['cyber']}->{a['cyber']}",
                    public=True, data={"cyber": a["cyber"]}))

    def _token_gone(self, step: int, tag: str, tok) -> None:
        """Why a token left the board: real kill vs recycle vs off-board move."""
        name = f"{tok.token_type.value}#{tok.uid}"
        if tok.destroyed:
            # Kills score openly — identity is public.
            self.events.append(EventView(
                step=step, type="token_destroyed", side=tag,
                message=f"{tag} {name} destroyed", public=True,
                data={"uid": tok.uid, "type": tok.token_type.value}))
        elif tok.off_board:
            self.events.append(EventView(
                step=step, type="token_off_board", side=tag,
                message=f"{tag} {name} moves off-board (Winchester ship; not destroyed)",
                public=tok.acquired, redacted=f"{tag} token #{tok.uid} leaves the board",
                data={"uid": tok.uid}))
        else:
            self.events.append(EventView(
                step=step, type="token_recycled", side=tag,
                message=(f"{tag} {name} recycles to its Squadron Card "
                         f"(ATO cleanup; not destroyed)"),
                public=tok.acquired, redacted=f"{tag} token #{tok.uid} leaves the board",
                data={"uid": tok.uid}))

    def visible_events(self, viewer: Optional[Side], reveal: bool,
                       limit: int = 100) -> list[EventView]:
        """The most recent events as `viewer` may see them (fog-filtered)."""
        out: list[EventView] = []
        for ev in self.events:
            if reveal or ev.public or (viewer is not None and ev.side == viewer.value):
                out.append(ev)
            elif ev.redacted is not None:
                out.append(ev.model_copy(update={"message": ev.redacted}))
        return out[-limit:]


def score_report(engine: RulesEngine, ato: int) -> ScoreReportView:
    """Structured per-side VP attribution for one ATO cycle (public info)."""
    sides: list[SideScoreView] = []
    for side in (Side.US, Side.PRC):
        p = engine.state.player(side)
        mission = MISSION_REGISTRY[p.mission_card_id].name if p.mission_card_id else "-"
        toks: list[CaptureLineView] = []
        sqs: list[CaptureLineView] = []
        cap_vp = 0
        for c in p.captures:
            if c.ato_cycle != ato:
                continue
            v = engine.capture_value(side, c)
            cap_vp += v
            if c.is_squadron_card:
                name = SQUADRON_REGISTRY[c.card_id].name if c.card_id else "squadron"
                sqs.append(CaptureLineView(label=name, vp=v, on_ground=c.destroyed_on_ground))
            else:
                tag = f"{c.token_type.value}#{c.uid}" if c.uid else c.token_type.value
                toks.append(CaptureLineView(label=tag, vp=v,
                                            on_ground=c.destroyed_on_ground))
        accrued: dict[str, int] = {}
        for a, reason, pts in p.vp_log:
            if a == ato:
                accrued[reason] = accrued.get(reason, 0) + pts
        sides.append(SideScoreView(
            side=side.value, mission=mission,
            ato_total=cap_vp + sum(accrued.values()),
            campaign_total=engine.total_victory_points(side),
            token_captures=toks, squadron_captures=sqs,
            accrued=[AccruedLineView(reason=r, points=pt) for r, pt in accrued.items()],
        ))
    return ScoreReportView(ato=ato, sides=sides)
