"""
models.py — Pydantic view schemas: the JSON contract between backend and UI.

Everything here is plain JSON-safe data (str/int/float/bool/list/dict) — no
enums, tuples, sets, or numpy types. Sides are "US"/"PRC" strings, bands are
BandID *names* (e.g. "BAND_A"), token types are their display values
(e.g. "F-35A"). Fog of war is applied by the serializer BEFORE these models
are built: a fogged token has `type=None` and status flags omitted, so private
information never reaches the browser.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# -- board -------------------------------------------------------------------

class TokenView(BaseModel):
    uid: int
    side: str                       # "US" / "PRC"
    band: str                       # BandID.name
    fogged: bool
    av: int                         # acquisition value (public)
    acquired: bool                  # public: acquisition flips the token face-up
    label: str                      # ready-made chip text, TUI-compatible
    type: Optional[str] = None      # None when fogged
    winchester: Optional[bool] = None
    grounded: Optional[bool] = None


class BandView(BaseModel):
    band: str                       # BandID.name
    header: str                     # display header, e.g. "BAND A"
    tokens: list[TokenView] = Field(default_factory=list)


class BoardView(BaseModel):
    bands: list[BandView]


# -- side panels ---------------------------------------------------------------

class SquadronView(BaseModel):
    card_id: int
    name: str
    token_type: str
    status: str                     # "ready" / "active" / "out" / "destroyed"
    damage: int
    tokens_lost: int
    at_contingency_location: bool
    grounded_tokens: int


class SquadronPanelView(BaseModel):
    side: str
    squadrons: list[SquadronView]   # only squadrons the viewer may identify
    face_down: int                  # hidden enemy cards, aggregate count
    off_board_naval: int


class HandCardView(BaseModel):
    card_id: int
    name: str
    revealed: bool                  # revealed to the opponent during Intel
    single_use: bool = False        # spent for the campaign once played (vs multi-use)


class HandView(BaseModel):
    side: str
    cards: list[HandCardView]       # only cards the viewer may identify
    hidden_count: int               # unidentified enemy cards in hand
    spent_count: int                # enablers this side has played (== len(spent))
    # Enabler cards this side has PLAYED (public — playing a card is a visible
    # act, so both a side's own and its opponent's spent cards are shown):
    # one-shots played this cycle, enduring cards in effect, and single-use
    # cards burned for the campaign.
    spent: list[HandCardView] = Field(default_factory=list)


# -- status strip ---------------------------------------------------------------

class StatusView(BaseModel):
    campaign: int
    ato_cycle: int
    total_ato_cycles: int
    turn_number: int
    phase: str
    active_side: str
    initiative: Optional[str] = None
    vp: dict[str, int]              # keyed "US"/"PRC"
    cyber: dict[str, int]
    missions: dict[str, Optional[str]]
    postures: dict[str, Optional[str]]
    base_damage: dict[str, int] = {}   # airbase VP damage boxes hit (0..3), public
    intel: dict[str, str] = {}         # intel track per side: NORMAL / ADVANTAGE


class CaptureEntryView(BaseModel):
    """One enemy unit surrendered to this side's score pile (public info)."""
    kind: str                       # "token" | "squadron"
    label: str                      # e.g. "J-10#7", "AEW REGIMENT"
    vp: int                         # current mission value of the capture
    victim_side: str                # side that lost the unit
    on_ground: bool = False
    token_type: Optional[str] = None   # token captures: display type
    uid: Optional[int] = None
    card_id: Optional[int] = None      # squadron captures: for card art
    ato_cycle: int = 0


class CapturesPanelView(BaseModel):
    side: str                       # the capturing side (owner of the pile)
    entries: list[CaptureEntryView]


# -- current decision --------------------------------------------------------

class ChoiceView(BaseModel):
    index: int                      # submit this to take the choice
    label: str
    kind: str
    category: str                   # frontend grouping bucket
    card_id: Optional[int] = None
    band: Optional[str] = None
    actor_uid: Optional[int] = None
    target_uid: Optional[int] = None
    branch_idx: Optional[int] = None
    alloc_kind: Optional[str] = None


class DecisionView(BaseModel):
    node_type: str
    side: str
    prompt: str
    choices: list[ChoiceView]


# -- events / scoring -----------------------------------------------------------

class EventView(BaseModel):
    step: int                       # decision number the event belongs to
    type: str                       # e.g. "decision", "token_destroyed", "dice"
    side: Optional[str] = None
    message: str                    # full detail (owner / omniscient view)
    public: bool = True             # False -> only `side` (or reveal) sees `message`
    redacted: Optional[str] = None  # fog-safe text shown to the other viewer
    data: dict[str, Any] = Field(default_factory=dict)


class CaptureLineView(BaseModel):
    label: str                      # e.g. "J-10#7", "AEW REGIMENT"
    vp: int
    on_ground: bool = False


class AccruedLineView(BaseModel):
    reason: str
    points: int


class SideScoreView(BaseModel):
    side: str
    mission: str
    ato_total: int
    campaign_total: int
    token_captures: list[CaptureLineView]
    squadron_captures: list[CaptureLineView]
    accrued: list[AccruedLineView]


class ScoreReportView(BaseModel):
    ato: int
    sides: list[SideScoreView]


# -- top level --------------------------------------------------------------------

class AgentMoveView(BaseModel):
    """One agent (LLM) move with its reasoning — for the Watch Agents feed."""
    decision_no: int
    side: str
    label: str
    reasoning: str


class GameView(BaseModel):
    game_id: str
    revision: int
    node_id: Optional[str]          # None once the game is over
    viewer: Optional[str]           # side the payload is fogged for; None = public/omniscient
    mode: str
    terminal: bool
    winner: Optional[str] = None
    handoff_required: bool = False  # hotseat: pass the device before showing state
    roll_pending: bool = False      # human_v_agent: reveal the dice, then POST /reveal
    recording: bool = False         # a trajectory is being captured (savable on demand)
    trajectory_saved: bool = False  # the tape has been written to disk this session
    # Recent agent moves + reasoning — populated only in Watch Agents mode.
    agent_log: list[AgentMoveView] = Field(default_factory=list)
    status: StatusView
    board: BoardView
    squadrons: list[SquadronPanelView]
    hands: list[HandView]
    captures: list[CapturesPanelView] = []
    decision: Optional[DecisionView] = None
    event_log: list[EventView] = Field(default_factory=list)
    score_report: Optional[ScoreReportView] = None
    # Rigid machine-readable text encoding of THIS view (fog already applied),
    # for the "Copy state for AI" button. See afwip/view/export.py and
    # pme_materials/04_state_encoding.md. None only if export is unavailable.
    llm_export: Optional[str] = None
