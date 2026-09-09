"""
trajectory.py — Replayable trajectory schema + recorder + materialization.

A trajectory is stored **compactly and replayably**: the campaign, the seed, and
the ordered list of chosen choice-indices (BOTH sides, in decision order) fully
determine the game, because `AFWIPEnv.reset(seed)` fixes the whole thing (dice
included). We therefore do NOT freeze observation vectors on disk — instead
`materialize()` replays the tape through a fresh env and regenerates
`(obs, action, mask, reward, done, outcome)` under the CURRENT observation
encoder. That keeps the dataset tiny and immune to obs-spec changes (the encoder
is still evolving).

Layout on disk: JSONL, one JSON object per game (a file may hold one or many).

    { "schema_version": 1, "campaign": 2, "seed": 12345, "max_turns": 150,
      "env_git_commit": "abc1234", "created_at": "...", "source": "human",
      "outcome": {"winner": "US"|"PRC"|null, "us_vp": 7, "prc_vp": 4,
                  "total_turns": 118, "capped": false},
      "tape": [ {"decision_no": 0, "side": "US", "node_type": "MISSION_PICK",
                 "n_choices": 1, "index": 0, "label": "...",
                 "justification": null}, ... ] }

The **tape carries both sides' choices in order** (required to replay); a side's
training set is the subset of steps where `side == learner_side`. Since the
project trains two separate US / PRC policies, `materialize(record, side)`
extracts one side at a time.

Games are time-limited (`max_turns`, default 150 in the env): at the cap the
leader on VP wins, so every recorded game ends with a real win / loss / draw.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Union

import numpy as np

from afwip.core.constants import Side


SCHEMA_VERSION = 1
DEFAULT_TRAJECTORY_DIR = Path("data/trajectories")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class TapeStep:
    """One applied decision. `index` is the position into the node's choice list
    (identical to the env action and the `ACTION: index=` in the copy-state
    block). `justification` is the optional doctrine rationale for the move.
    `us_vp`/`prc_vp` are the victory points at the moment of the decision (before
    it is applied) — recorded so rewards can be re-derived from the tape alone,
    without replay and independent of any later scoring-rule change."""
    decision_no: int
    side: str                    # "US" / "PRC"
    node_type: str               # NodeType.name
    n_choices: int
    index: int
    label: str
    justification: Optional[str] = None
    us_vp: Optional[int] = None
    prc_vp: Optional[int] = None


@dataclass
class Outcome:
    winner: Optional[str]        # "US" / "PRC" / None (draw)
    us_vp: int
    prc_vp: int
    total_turns: int
    capped: bool                 # game ended by the timed turn cap


@dataclass
class Trajectory:
    campaign: int
    seed: int
    max_turns: Optional[int]
    outcome: Outcome
    tape: list[TapeStep]
    schema_version: int = SCHEMA_VERSION
    env_git_commit: str = "unknown"
    created_at: str = ""
    source: str = "human"        # "human" (hand-curated) / model id, later

    # -- (de)serialization --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign": self.campaign,
            "seed": self.seed,
            "max_turns": self.max_turns,
            "env_git_commit": self.env_git_commit,
            "created_at": self.created_at,
            "source": self.source,
            "outcome": asdict(self.outcome),
            "tape": [asdict(s) for s in self.tape],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trajectory":
        return cls(
            campaign=d["campaign"],
            seed=d["seed"],
            max_turns=d.get("max_turns"),
            outcome=Outcome(**d["outcome"]),
            tape=[TapeStep(**s) for s in d["tape"]],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            env_git_commit=d.get("env_git_commit", "unknown"),
            created_at=d.get("created_at", ""),
            source=d.get("source", "human"),
        )

    def side_steps(self, side: Union[str, Side]) -> list[TapeStep]:
        s = side.value if isinstance(side, Side) else side
        return [t for t in self.tape if t.side == s]

    def outcome_label(self, side: Union[str, Side]) -> str:
        """'win' / 'loss' / 'draw' from `side`'s perspective."""
        s = side.value if isinstance(side, Side) else side
        if self.outcome.winner is None:
            return "draw"
        return "win" if self.outcome.winner == s else "loss"


# ---------------------------------------------------------------------------
# Recorder — used by the web session (and any headless driver)
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:      # pragma: no cover - git absent / not a repo
        return "unknown"


class TrajectoryRecorder:
    """Accumulates the decision tape for one game, then finalizes it against the
    engine into a `Trajectory`. One recorder per game.

    The session appends every applied decision via `record()` at its single
    write seam, then calls `finalize(engine)` when the game ends. Only the tape
    is stored — observations are regenerated by `materialize()` on replay.
    """

    def __init__(self, campaign: int, seed: int, max_turns: Optional[int],
                 source: str = "human"):
        self.campaign = campaign
        self.seed = seed
        self.max_turns = max_turns
        self.source = source
        self.tape: list[TapeStep] = []

    def record(self, decision_no: int, node: Any, index: int,
               justification: Optional[str] = None, engine: Any = None) -> None:
        """Append the decision about to be applied (`node.choices[index]`). Pass
        `engine` to record the current VP (for reward re-derivation)."""
        choice = node.choices[index]
        us_vp = prc_vp = None
        if engine is not None:
            us_vp = engine.total_victory_points(Side.US)
            prc_vp = engine.total_victory_points(Side.PRC)
        self.tape.append(TapeStep(
            decision_no=decision_no,
            side=node.side.value,
            node_type=node.node_type.name,
            n_choices=len(node.choices),
            index=index,
            label=choice.label,
            justification=justification,
            us_vp=us_vp,
            prc_vp=prc_vp,
        ))

    def finalize(self, engine: Any) -> Trajectory:
        gs = engine.state
        capped = (gs.max_turns is not None and gs.total_turns >= gs.max_turns)
        outcome = Outcome(
            winner=gs.winner.value if gs.winner else None,
            us_vp=engine.total_victory_points(Side.US),
            prc_vp=engine.total_victory_points(Side.PRC),
            total_turns=gs.total_turns,
            capped=capped,
        )
        return Trajectory(
            campaign=self.campaign,
            seed=self.seed,
            max_turns=self.max_turns,
            outcome=outcome,
            tape=list(self.tape),
            env_git_commit=_git_commit(),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source=self.source,
        )


# ---------------------------------------------------------------------------
# JSONL IO
# ---------------------------------------------------------------------------

def write_trajectory(traj: Trajectory, path: Union[str, Path]) -> Path:
    """Write a single trajectory as a one-line JSONL file (creating dirs)."""
    return write_trajectories([traj], path)


def write_trajectories(trajs: Iterable[Trajectory], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for t in trajs:
            fh.write(json.dumps(t.to_dict(), separators=(",", ":")))
            fh.write("\n")
    return path


def read_trajectories(path: Union[str, Path]) -> list[Trajectory]:
    out: list[Trajectory] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Trajectory.from_dict(json.loads(line)))
    return out


# ---------------------------------------------------------------------------
# Materialization — replay a tape into training samples
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """One training example for `side`: the regenerated observation, the taken
    action (+ its legal mask), the reward, done flag, and the trajectory-level
    win/loss/draw label (useful for return-conditioning / filtering)."""
    side: str
    observation: dict[str, np.ndarray]
    action_mask: np.ndarray
    action_index: int
    reward: float
    done: bool
    outcome_label: str


REWARDS = ("sparse", "attrition_vp")


def materialize(traj: Trajectory, side: Union[str, Side],
                reward: str = "sparse", vp_scale: float = 1.0) -> list[Sample]:
    """
    Replay `traj` through a fresh env and emit one `Sample` per decision made by
    `side`. Observations are regenerated under the CURRENT encoder.

    reward:
      - "sparse": 0 everywhere except the side's final decision, which gets the
        terminal ±1 (0 on a draw) — the win/loss signal.
      - "attrition_vp": each decision earns `vp_scale * Δ(my_vp - opp_vp)`
        realized between it and the side's next decision (telescopes to the final
        VP differential). Both sides being Attrition in Campaign 2, this is a
        dense, doctrine-neutral score-margin reward.
    """
    from afwip.env import AFWIPEnv          # local import (avoids cycle at load)

    if reward not in REWARDS:
        raise ValueError(f"unknown reward {reward!r} (choose from {REWARDS})")
    s = side.value if isinstance(side, Side) else side
    me = Side(s)
    opp = Side.PRC if me == Side.US else Side.US

    env = AFWIPEnv(campaign=traj.campaign, max_turns=traj.max_turns)
    env.reset(seed=traj.seed)

    def vp_diff() -> int:
        return (env.engine.total_victory_points(me)
                - env.engine.total_victory_points(opp))

    # Pass 1: replay, capturing this side's decisions + the score margin BEFORE
    # each of its actions. Assert the tape matches the live decision stream.
    raw: list[tuple[dict, np.ndarray, int, int]] = []
    for step in traj.tape:
        node = env._node
        assert node is not None, "tape longer than the game (replay desync)"
        acting = env.agent_selection
        assert acting == step.side, (
            f"replay desync at decision {step.decision_no}: "
            f"tape says {step.side}, env says {acting}")
        # Integrity: the stored index is a POSITION into a regenerated choice
        # list, so hard-fail if the engine's choice ordering has drifted (the
        # index would silently mean a different move). node_type + label must
        # match what was recorded.
        assert 0 <= step.index < len(node.choices), (
            f"decision {step.decision_no}: index {step.index} out of range "
            f"0..{len(node.choices) - 1} (choice set changed under this seed)")
        assert (node.node_type.name == step.node_type
                and node.choices[step.index].label == step.label), (
            f"replay drift at decision {step.decision_no}: recorded "
            f"{step.node_type}[{step.index}]={step.label!r} but the env now "
            f"offers {node.node_type.name}[{step.index}]="
            f"{node.choices[step.index].label!r} — engine choice ordering "
            f"changed; this trajectory must be re-recorded")
        if acting == s:
            obs = env.observe(s)
            # VP margin (my_vp - opp_vp) for the attrition reward: prefer the VP
            # RECORDED at capture time (robust to later scoring-rule changes),
            # falling back to the replayed engine for older tapes.
            if step.us_vp is not None and step.prc_vp is not None:
                margin = (step.us_vp - step.prc_vp) if s == "US" else (step.prc_vp - step.us_vp)
            else:
                margin = vp_diff()
            raw.append((obs["observation"], obs["action_mask"], step.index, margin))
        env.step(step.index)

    # Terminal margin from the recorded outcome (self-contained), else replay.
    if traj.outcome.us_vp is not None:
        final_diff = (traj.outcome.us_vp - traj.outcome.prc_vp) if s == "US" \
            else (traj.outcome.prc_vp - traj.outcome.us_vp)
    else:
        final_diff = vp_diff()
    label = traj.outcome_label(s)

    # Pass 2: assign rewards + done, now that we know the ordering and terminal.
    samples: list[Sample] = []
    for i, (obs, mask, idx, before) in enumerate(raw):
        done = (i == len(raw) - 1)
        if reward == "sparse":
            r = 0.0
            if done:
                r = 0.0 if label == "draw" else (1.0 if label == "win" else -1.0)
        else:  # attrition_vp
            nxt = raw[i + 1][3] if i + 1 < len(raw) else final_diff
            r = vp_scale * (nxt - before)
        samples.append(Sample(
            side=s, observation=obs, action_mask=mask, action_index=idx,
            reward=float(r), done=done, outcome_label=label))
    return samples


def materialize_both(traj: Trajectory, reward: str = "sparse",
                     vp_scale: float = 1.0) -> dict[str, list[Sample]]:
    """Convenience: materialize both seats at once -> {"US": [...], "PRC": [...]}."""
    return {side: materialize(traj, side, reward=reward, vp_scale=vp_scale)
            for side in ("US", "PRC")}


# ---------------------------------------------------------------------------
# Move source — interface for automated generation (follow-on; not the hand flow)
# ---------------------------------------------------------------------------

@dataclass
class MoveProposal:
    """A proposed decision: the chosen choice `index`, plus optional reasoning
    (`justification`) and a short forward `plan` (the ~5-move look-ahead)."""
    index: int
    justification: Optional[str] = None
    plan: Optional[str] = None


class MoveSource(Protocol):
    """Produces a move for the current decision node from a serialized state.

    The hand-curation flow does NOT use this (the user plays through the web app
    and its Copy-state button). It is the seam for the LATER automated
    high-volume flow (an OpenAI/other adapter drops in here), kept now so the
    schema/driver need no rework: `state_text` is the fogged copy-state block
    (`afwip.view.export.game_export`), `choices` the current legal choice list.
    """

    def propose(self, state_text: str, choices: list) -> MoveProposal: ...


class InteractiveMoveSource:
    """Minimal reference `MoveSource`: prints the copy-state block and reads a
    choice index from stdin (a headless CLI curation aid). Retained as the
    concrete example the future OpenAI adapter mirrors."""

    def propose(self, state_text: str, choices: list) -> MoveProposal:
        print(state_text)
        n = len(choices)
        while True:
            raw = input(f"choice index [0..{n - 1}]: ").strip()
            if raw.isdigit() and 0 <= int(raw) < n:
                return MoveProposal(index=int(raw))
            print(f"  enter an integer in 0..{n - 1}")
