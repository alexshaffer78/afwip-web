"""
state_text.py — Build the fog-safe copy-state text an LLM plays from.

Assembles the same `GameView` the web app fogs and serializes (via
`afwip.view.serializer`) for one viewer side, then renders it with
`afwip.view.export.game_export`. The result is the `=== AFWIP STATE ===` block
ending in the indexed `ACTION:` list — identical in spirit to the browser's
"Copy state for AI" button, but built straight off the engine (no web session)
and fogged to the deciding side, so the model never sees hidden enemy state.
"""

from __future__ import annotations

from typing import Optional, Union

from afwip.core.constants import Side
from afwip.core.rules import RulesEngine
from afwip.script import DecisionNode
from afwip.view import serializer
from afwip.view.export import game_export
from afwip.view.models import GameView


def llm_state_text(engine: RulesEngine, node: Optional[DecisionNode],
                   viewer: Union[str, Side]) -> str:
    """Render the fogged state + legal-action menu for `viewer` to play from."""
    viewer = viewer if isinstance(viewer, Side) else Side(viewer)
    gs = engine.state
    view = GameView(
        game_id="gen", revision=0, node_id=None,
        viewer=viewer.value, mode="agent_v_agent",
        terminal=node is None,
        winner=gs.winner.value if gs.winner else None,
        status=serializer.status_view(engine),
        board=serializer.board_view(engine, viewer, reveal=False),
        squadrons=[serializer.squadron_panel(engine, s, viewer, reveal=False)
                   for s in Side],
        hands=[serializer.hand_view(engine, s, viewer, reveal=False) for s in Side],
        captures=[serializer.captures_panel(engine, s) for s in Side],
        decision=serializer.decision_view(node) if node is not None else None,
    )
    return game_export(view)
