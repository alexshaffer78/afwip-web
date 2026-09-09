"""
layout.py — Static presentation layout shared by the web view layer.

Pure constants: the board's left-to-right band order with human headers,
per-node prompts, and the choice-category grouping used by the frontend's
choice panel. The band order matches afwip.tui.BOARD_COLUMNS (pinned by a
lockstep test) without importing or modifying the TUI.
"""

from __future__ import annotations

from afwip.core.constants import BandID
from afwip.script import NodeType


# Left-to-right board columns (US rear on the left, PRC rear on the right).
BOARD_BANDS: list[tuple[BandID, str]] = [
    (BandID.US_STANDOFF, "US-STANDOFF"),
    (BandID.US_AIRBASE, "US-AIRBASE"),
    (BandID.US_CONTINGENCY_LOCATION, "US-CL"),
    (BandID.BAND_A, "BAND A"),
    (BandID.BAND_B, "BAND B"),
    (BandID.BAND_C, "BAND C"),
    (BandID.BAND_D, "BAND D"),
    (BandID.BAND_E, "BAND E"),
    (BandID.PRC_AIRBASE, "PRC-AIRBASE"),
    (BandID.PRC_STANDOFF, "PRC-STANDOFF"),
]


# Human-readable prompt for each decision node (shown above the choice panel).
NODE_PROMPTS: dict[NodeType, str] = {
    NodeType.MISSION_PICK: "Choose your Mission Card",
    NodeType.POSTURE_PICK: "Choose your Posture",
    NodeType.SQUADRON_PICK: "Draft Squadron Cards",
    NodeType.SQUADRON_BASE: "Place squadron: Airbase or Contingency Location",
    NodeType.ENABLER_PICK: "Draft Enabler Cards",
    NodeType.BID_SACRIFICE: "Sacrifice enablers to bid for initiative",
    NodeType.FIRST_PLAYER: "Choose who moves first",
    NodeType.INTEL_REVEAL: "Choose an enabler to reveal (opponent Intel)",
    NodeType.TURN_ACTION: "Your turn — choose an action",
    NodeType.SPAWN_BAND: "Choose the generation band",
    NodeType.ENABLER_BRANCH: "Choose a card option",
    NodeType.MD_DECLARE: "Declare missile defense?",
    NodeType.RESPONSE: "Play a response card?",
    NodeType.ALLOC_POINT: "Allocate one point of base-strike damage",
}


# Frontend grouping bucket for a choice, keyed by its Choice `kind`.
CHOICE_CATEGORY: dict[str, str] = {
    "pass": "pass",
    "activate": "activate",
    "move": "move",
    "acquire": "acquire",
    "shoot_air": "attack",
    "shoot_surface": "attack",
    "play_enabler": "enabler",
    "pick": "setup",
    "done": "setup",
    "skip": "setup",
}


def node_prompt(node_type: NodeType) -> str:
    return NODE_PROMPTS.get(node_type, node_type.name)


def choice_category(kind: str) -> str:
    return CHOICE_CATEGORY.get(kind, "other")
