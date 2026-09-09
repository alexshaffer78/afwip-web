"""
agents.py — Pluggable "answerers" for agent-controlled sides.

Every play mode differs only in who picks the current decision node's choice:
a human (posted index), or an Agent below. `observation` is a ZERO-ARG CALLABLE
returning the PettingZoo observation dict (`env.observe(agent)`) — lazy so
cheap agents never pay for the array encoding; a future PolicyAgent calls it.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from afwip.core.constants import Side
from afwip.script import Choice

# The 7 observation arrays, in the order the ONNX policy expects them (must match
# afwip.rl.obs_encode.OBS_KEYS — hardcoded here so the web runtime stays
# torch-free, loading only onnxruntime + numpy).
_OBS_KEYS = ("scalars", "own_tokens", "enemy_tokens", "own_squadrons",
             "enemy_squadrons", "cards", "choices")


def _models_dir() -> Path:
    """Where the exported <difficulty>_{us,prc}.onnx files live (repo-root
    `models/` by default; override with AFWIP_MODELS_DIR)."""
    env = os.environ.get("AFWIP_MODELS_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "models"


class Agent(Protocol):
    name: str

    def choose(self, observation: Callable[[], Any],
               choices: list[Choice], rng: random.Random,
               context: Optional[dict] = None) -> int:
        """Return the index of the choice to take.

        `context` (optional) carries per-decision extras the session can supply:
          - "side": the deciding side's value ("US"/"PRC")
          - "state_text": a zero-arg callable returning the fogged copy-state
            text (built lazily so cheap agents never pay for it)
        """
        ...


class RandomAgent:
    """Uniform random over the presented choices (the TUI's agent, boxed)."""

    name = "random"

    def choose(self, observation: Callable[[], Any],
               choices: list[Choice], rng: random.Random,
               context: Optional[dict] = None) -> int:
        return rng.randrange(len(choices))


class LLMAgent:
    """Doctrine-aligned GPT player (US uses USAF doctrine, PRC uses PLA doctrine)
    — the same OpenAIMoveSource the batch generator uses, so a watched
    "Watch Agents" game IS a recorded trajectory. Forced nodes skip the API.

    `last_justification` holds the reasoning for the move just chosen; the
    session records it on the trajectory step."""

    name = "gpt"

    def __init__(self, model: Optional[str] = None):
        # Import here so the web app runs without the `openai` package unless
        # this agent is actually used.
        from afwip.rl.openai_agent import OpenAIMoveSource, DEFAULT_MODEL
        self._Source = OpenAIMoveSource
        self.model = model or DEFAULT_MODEL
        self._sources: dict[Side, Any] = {}
        self.last_justification: Optional[str] = None
        self._api_key: Optional[str] = None

    def set_api_key(self, key: Optional[str]) -> None:
        """A UI-supplied OpenAI key; forwarded to each side's move source.
        (Falls back to $OPENAI_API_KEY when unset — see OpenAIMoveSource.client.)"""
        self._api_key = key or None
        self._sources.clear()              # rebuild sources with the new key

    def _source(self, side: Side):
        if side not in self._sources:
            self._sources[side] = self._Source(side, model=self.model,
                                               api_key=self._api_key)
        return self._sources[side]

    def choose(self, observation: Callable[[], Any],
               choices: list[Choice], rng: random.Random,
               context: Optional[dict] = None) -> int:
        self.last_justification = None
        if len(choices) == 1:                    # forced node — no API call
            return 0
        ctx = context or {}
        side = Side(ctx["side"]) if "side" in ctx else Side.US
        state_text = ctx["state_text"]() if "state_text" in ctx else ""
        prop = self._source(side).propose(state_text, choices,
                                          setup=bool(ctx.get("setup")))
        self.last_justification = prop.justification
        return prop.index

    # -- browser split (build request in Python, fetch in JS, parse in Python) --

    def build_request(self, side: Side, choices: list[Choice], state_text: str,
                      setup: bool = False) -> dict:
        return self._source(side).build_request(state_text, choices, setup=setup)

    def parse_reply(self, side: Side, choices: list[Choice], raw: str):
        prop = self._source(side).apply_reply(choices, raw)
        self.last_justification = prop.justification
        return prop.index, prop.justification


class PolicyAgent:
    """Plays via the trained pointer net, exported to ONNX and run with
    onnxruntime (no torch at runtime). One model per side —
    `<difficulty>_{us,prc}.onnx` under the models dir — loaded lazily on first
    use for that side. Forced single-choice nodes skip inference.

    Actions are SAMPLED from the (masked) policy for variety and natural play;
    pass `deterministic=True` for always-best (argmax) moves. The three shipped
    tiers are 'easy' / 'medium' / 'best' (weaker→stronger checkpoints)."""

    def __init__(self, difficulty: str, deterministic: bool = False,
                 models_dir: Optional[Path] = None):
        self.difficulty = difficulty
        self.name = f"policy-{difficulty}"
        self.deterministic = deterministic
        self._dir = Path(models_dir) if models_dir else _models_dir()
        self._sessions: dict[str, Any] = {}

    def _session(self, side: Side):
        key = side.value.lower()               # "us" / "prc"
        if key not in self._sessions:
            import onnxruntime as ort           # lazy: app runs without it otherwise
            path = self._dir / f"{self.difficulty}_{key}.onnx"
            if not path.exists():
                raise FileNotFoundError(f"policy model not found: {path}")
            self._sessions[key] = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"])
        return self._sessions[key]

    def build_feed(self, observation: Callable[[], Any]) -> dict:
        """Stack the env observation into the ONNX input dict (batch dim added).
        Split out of `choose` so the browser can marshal it to onnxruntime-web
        and `await` the (async) inference before picking."""
        import numpy as np
        packed = observation()                  # env.observe(side): obs dict + mask
        obs, mask = packed["observation"], packed["action_mask"]
        feed = {k: obs[k][None].astype(np.float32) for k in _OBS_KEYS}
        feed["action_mask"] = mask[None].astype(bool)
        return feed

    def pick_from_logits(self, logits, choices: list[Choice],
                         rng: random.Random) -> int:
        """Mask to the legal prefix and pick (argmax if deterministic, else
        sample the masked softmax). `logits` is the raw (256,) output row."""
        import numpy as np
        # The legal choices are exactly indices 0..len(choices)-1 (the env fills
        # the mask that way), so score only that prefix.
        z = np.asarray(logits[:len(choices)], dtype=np.float64)
        if self.deterministic:
            return int(z.argmax())
        z = z - z.max()                          # sample from the masked softmax
        p = np.exp(z); p /= p.sum()
        idx = int(np.searchsorted(np.cumsum(p), rng.random()))
        return min(idx, len(choices) - 1)

    def choose(self, observation: Callable[[], Any],
               choices: list[Choice], rng: random.Random,
               context: Optional[dict] = None) -> int:
        if len(choices) == 1:                   # forced node — no inference
            return 0
        side = Side(context["side"]) if context and "side" in context else Side.US
        feed = self.build_feed(observation)
        logits = self._session(side).run(["logits", "value"], feed)[0][0]  # (256,)
        return self.pick_from_logits(logits, choices, rng)


AGENT_REGISTRY: dict[str, Callable[[], Agent]] = {
    "random": RandomAgent,
    "gpt": LLMAgent,               # OpenAI doctrine player (needs OPENAI_API_KEY)
    "policy-easy": lambda: PolicyAgent("easy"),      # trained AI opponents (ONNX)
    "policy-medium": lambda: PolicyAgent("medium"),
    "policy-best": lambda: PolicyAgent("best"),
}


def make_agent(name: str) -> Agent:
    try:
        return AGENT_REGISTRY[name]()
    except KeyError:
        raise ValueError(f"unknown agent '{name}' "
                         f"(available: {', '.join(sorted(AGENT_REGISTRY))})")
