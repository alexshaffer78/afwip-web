"""
_farama_shim.py — minimal stand-ins for `gymnasium` and `pettingzoo` so
`afwip.env` imports and runs inside Pyodide (which ships neither).

`afwip.env` uses an intentionally tiny slice of the Farama API:
  * `gymnasium.spaces.{Discrete, Box, Dict}` — only constructed (stored on the
    env, never sampled) during human/agent play.
  * `pettingzoo.AECEnv` as a base class, plus its reward bookkeeping helpers
    `_clear_rewards` / `_accumulate_rewards` and the dead-agent edge path
    `_was_dead_step` (never reached in normal play — the session guards it).

`session.py` drives the env directly via `.step()` / `.observe()` / `.engine`
and never touches the AEC iteration helpers (`last`, `agent_iter`, …), so this
shim is all the surface the web runtime needs. Call `install()` BEFORE importing
`afwip.env`; on CPython with the real packages present it is a harmless no-op.
"""

from __future__ import annotations

import sys
import types


def _make_spaces_module() -> types.ModuleType:
    mod = types.ModuleType("gymnasium.spaces")

    class Space:
        """Common base; stores its shape/dtype but has no runtime behavior."""
        def __init__(self, shape=None, dtype=None):
            self.shape = shape
            self.dtype = dtype

    class Discrete(Space):
        def __init__(self, n, start=0):
            super().__init__(shape=(), dtype=int)
            self.n = n
            self.start = start

    class Box(Space):
        def __init__(self, low, high, shape=None, dtype=None):
            super().__init__(shape=shape, dtype=dtype)
            self.low = low
            self.high = high

    class Dict(Space):
        def __init__(self, spaces=None, **kwargs):
            super().__init__()
            self.spaces = dict(spaces or {}, **kwargs)

    mod.Space = Space
    mod.Discrete = Discrete
    mod.Box = Box
    mod.Dict = Dict
    return mod


def _make_pettingzoo_module() -> types.ModuleType:
    mod = types.ModuleType("pettingzoo")

    class AECEnv:
        """Just enough of PettingZoo's AECEnv for `afwip.env`: reward
        bookkeeping helpers + a safe dead-step. The env sets `agents`,
        `rewards`, `_cumulative_rewards`, etc. itself in `reset()`."""

        def __init__(self, *args, **kwargs):
            pass

        def _clear_rewards(self) -> None:
            for a in self.rewards:
                self.rewards[a] = 0.0

        def _accumulate_rewards(self) -> None:
            for a, r in self.rewards.items():
                self._cumulative_rewards[a] += r

        def _was_dead_step(self, action) -> None:
            # Reached only if a terminated/truncated agent is stepped; the web
            # session never does this (it stops at terminal). Keep it safe.
            self._node = None
            for a in list(getattr(self, "agents", [])):
                self.terminations[a] = True

    mod.AECEnv = AECEnv
    return mod


def install() -> None:
    """Register the shims in sys.modules if the real packages are absent."""
    if "gymnasium" not in sys.modules:
        try:
            import gymnasium  # noqa: F401  (real package present — use it)
        except ImportError:
            gym = types.ModuleType("gymnasium")
            spaces = _make_spaces_module()
            gym.spaces = spaces
            sys.modules["gymnasium"] = gym
            sys.modules["gymnasium.spaces"] = spaces
    if "pettingzoo" not in sys.modules:
        try:
            import pettingzoo  # noqa: F401
        except ImportError:
            sys.modules["pettingzoo"] = _make_pettingzoo_module()
