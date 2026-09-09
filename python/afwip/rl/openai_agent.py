"""
openai_agent.py — GPT-backed move source for trajectory generation.

`OpenAIMoveSource.propose(state_text, choices)` sends the side's cached system
prompt (rules + doctrine + format) plus the fogged state/action text, asks GPT
for a JSON `{index, reasoning, plan}`, validates the index against the legal
choices, and retries/falls back on a bad reply. Forced (single-choice) nodes are
answered with no API call. The OpenAI client is injectable so tests run without
a network or key.

Cost: only multi-choice nodes cost a call; with a cheap frontier model (default
gpt-5.6-luna) a Campaign-2 game runs a small fraction of a dollar. Usage is
accumulated on the instance for budgeting.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Optional

from afwip.core.constants import Side
from afwip.script import Choice
from afwip.rl.prompts import system_prompt
from afwip.rl.trajectory import MoveProposal

# Default model; override everywhere (script default + web "gpt" agent) with
# `export AFWIP_OPENAI_MODEL=<id>` (e.g. gpt-4o for a chat model).
DEFAULT_MODEL = os.environ.get("AFWIP_OPENAI_MODEL", "gpt-5.6-luna")
DEFAULT_REASONING_EFFORT = os.environ.get("AFWIP_REASONING_EFFORT", "high")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Reasoning models (gpt-5.x / Luna, o-series) take `reasoning_effort` and pin
# temperature to 1; classic chat models (gpt-4o…) take `temperature` +
# `response_format`. Detected by name; override with the `reasoning=` arg.
_REASONING_PREFIXES = ("o1", "o3", "o4", "o5", "gpt-5")


def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return "luna" in m or any(m.startswith(p) for p in _REASONING_PREFIXES)


def _parse(raw: str) -> Optional[dict]:
    """Extract a {index, reasoning, plan} object from a model reply."""
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_RE.search(raw)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict) or "index" not in obj:
        return None
    try:
        obj["index"] = int(obj["index"])
    except (TypeError, ValueError):
        return None
    return obj


def _fallback_index(choices: list[Choice]) -> int:
    """A safe legal move when parsing fails: prefer a non-pass action."""
    for i, c in enumerate(choices):
        if c.kind != "pass":
            return i
    return 0


class OpenAIMoveSource:
    """Doctrine-aligned GPT move source for one side."""

    def __init__(self, side: Side, model: str = DEFAULT_MODEL,
                 client: Any = None, api_key: Optional[str] = None,
                 temperature: float = 0.3, setup_temperature: float = 1.0,
                 reasoning_effort: str = DEFAULT_REASONING_EFFORT,
                 reasoning: Optional[bool] = None, max_retries: int = 1,
                 call_retries: int = 2, retry_backoff: float = 0.6):
        self.side = side if isinstance(side, Side) else Side(side)
        self.model = model
        # Reasoning models (Luna/gpt-5/o-series) use `reasoning_effort` and pin
        # temperature; chat models use temperature. For chat models,
        # `temperature` drives gameplay and `setup_temperature` the pre-turn
        # draft (more variety), chosen per call via propose(setup=…).
        self.reasoning = (reasoning if reasoning is not None
                          else _is_reasoning_model(model))
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.setup_temperature = setup_temperature
        self.max_retries = max_retries
        # Transport robustness: how many EXTRA times to repeat the API call
        # itself if it doesn't go through (raises, or returns an empty reply),
        # with exponential backoff between attempts. Separate from `max_retries`,
        # which re-prompts on a well-formed-but-invalid reply.
        self.call_retries = call_retries
        self.retry_backoff = retry_backoff
        self.system = system_prompt(self.side)
        self._client = client
        self._api_key = api_key
        # Short memory of THIS side's own recent moves + last stated plan, fed
        # back each call so the model stays consistent instead of oscillating
        # (the API is otherwise stateless per decision — the root cause of
        # move-back-and-forth cycling).
        self.history: list[str] = []
        self.last_plan: Optional[str] = None
        self.memory_len = 8
        # Recurrent memory (the API is stateless, so we carry it ourselves):
        #   `strategy` is written ONCE on the first decision — a game-long plan
        #     that is fed back on every subsequent turn (a fixed north star).
        #   `running_summary` is rewritten by the model EVERY turn — a rolling
        #     3-5 sentence digest of the relevant situation so far, passed
        #     forward so each decision has interpreted context, not just the raw
        #     state. Together these give the otherwise-stateless agent a
        #     recurrent working memory.
        self.strategy: Optional[str] = None
        self.running_summary: Optional[str] = None
        # Accumulated usage (for cost/progress logging).
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.parse_failures = 0
        self.call_failures = 0        # transient API failures / empty replies retried

    # -- client ---------------------------------------------------------------

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:      # pragma: no cover - env-dependent
                raise RuntimeError(
                    f"The `openai` package is not importable by THIS interpreter "
                    f"({sys.executable}). Install it into the same environment: "
                    f"python -m pip install -r requirements-rl.txt") from e
            key = self._api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set (export it or pass api_key=).")
            self._client = OpenAI(api_key=key)
        return self._client

    def _complete(self, messages: list[dict], temperature: float) -> str:
        kwargs: dict = {"model": self.model, "messages": messages}
        if self.reasoning:
            # Reasoning models: reasoning_effort, no temperature/response_format
            # (temperature is fixed to 1). Strict JSON is handled by the prompt +
            # the tolerant parser below.
            kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["temperature"] = temperature
            kwargs["response_format"] = {"type": "json_object"}
        # Transport robustness: repeat the call up to `call_retries` extra times
        # if it doesn't go through — either the request raises (network,
        # rate-limit, 5xx, timeout) or it returns an empty completion (a
        # reasoning model can burn its budget and emit nothing) — with
        # exponential backoff. A genuine, repeated failure still raises: the
        # batch/web caller skips that game rather than banking a fallback move as
        # real data. Well-formed-but-invalid replies are handled separately by
        # the parse-retry loop in `propose`.
        last_exc: Optional[Exception] = None
        for attempt in range(self.call_retries + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except Exception as e:                      # transient transport error
                last_exc = e
                self.call_failures += 1
                if attempt < self.call_retries:
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                raise
            self.calls += 1
            usage = getattr(resp, "usage", None)
            if usage is not None:
                self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            content = resp.choices[0].message.content or ""
            if content.strip() or attempt == self.call_retries:
                return content
            # Empty reply: retry the call before handing "" to the parser.
            self.call_failures += 1
            time.sleep(self.retry_backoff * (2 ** attempt))
        # Unreachable: the loop always returns or raises. Guards type-checkers
        # and, defensively, re-raises the last error if it somehow falls through.
        if last_exc is not None:
            raise last_exc
        return ""

    # -- the move -------------------------------------------------------------

    def _user_message(self, state_text: str) -> str:
        """State + this side's recurrent memory: the game-long strategy, the
        rolling situation summary, and recent moves / plan — so each stateless
        API call continues a coherent game instead of restarting cold or undoing
        its own moves."""
        mem = []
        if self.strategy:
            mem.append(f"Your game strategy (set at the start — hold to it "
                       f"unless the situation forces a real change): {self.strategy}")
        if self.running_summary:
            mem.append(f"Situation so far (your running summary): {self.running_summary}")
        if self.history:
            mem.append("Your recent moves (oldest→newest): "
                       + "; ".join(self.history))
        if self.last_plan:
            mem.append(f"Your current plan: {self.last_plan}")
        if not mem:
            return state_text
        mem.append("Advance your strategy and plan and take decisive action. Do "
                   "NOT reverse your own recent move or re-do it without a concrete "
                   "new reason.")
        return state_text + "\n\n" + "\n".join(mem)

    def _remember(self, label: str, plan: Optional[str]) -> None:
        self.history.append(label)
        self.history = self.history[-self.memory_len:]
        if plan:
            self.last_plan = plan

    def propose(self, state_text: str, choices: list[Choice],
                *, setup: bool = False) -> MoveProposal:
        n = len(choices)
        if n == 1:                        # forced node — no API call
            return MoveProposal(index=0, justification=None, plan=None)

        temperature = self.setup_temperature if setup else self.temperature
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self._user_message(state_text)},
        ]
        for _ in range(self.max_retries + 1):
            raw = self._complete(messages, temperature)
            parsed = _parse(raw)
            if parsed is not None and 0 <= parsed["index"] < n:
                reasoning = (str(parsed.get("reasoning"))
                             if parsed.get("reasoning") is not None else None)
                plan = (str(parsed.get("plan"))
                        if parsed.get("plan") is not None else None)
                # Recurrent memory: capture the game-long strategy ONCE (first
                # move that provides one), and refresh the rolling summary every
                # move it is given. Both are fed back by `_user_message`.
                strategy = (str(parsed.get("strategy"))
                            if parsed.get("strategy") is not None else None)
                if strategy and not self.strategy:
                    self.strategy = strategy
                summary = (str(parsed.get("summary"))
                           if parsed.get("summary") is not None else None)
                if summary:
                    self.running_summary = summary
                self._remember(choices[parsed["index"]].label, plan)
                return MoveProposal(index=parsed["index"],
                                    justification=reasoning, plan=plan)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": (
                f"Invalid reply. `index` must be an integer 0..{n - 1} taken "
                f"from the ACTION list. Respond with the JSON object only.")})

        self.parse_failures += 1
        return MoveProposal(index=_fallback_index(choices),
                            justification="fallback: unparseable model reply", plan=None)

    # -- browser split: assemble the request in Python, run the HTTP call in JS,
    #    then parse + update memory in Python (single attempt, no SDK). --------

    def build_request(self, state_text: str, choices: list[Choice],
                      *, setup: bool = False) -> dict:
        """Assemble the OpenAI chat-completions request (same messages/params as
        `_complete`) for the browser fetch bridge. Returns {api_key, body}."""
        temperature = self.setup_temperature if setup else self.temperature
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self._user_message(state_text)},
        ]
        body: dict = {"model": self.model, "messages": messages}
        if self.reasoning:
            body["reasoning_effort"] = self.reasoning_effort
        else:
            body["temperature"] = temperature
            body["response_format"] = {"type": "json_object"}
        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        return {"api_key": key, "body": body}

    def apply_reply(self, choices: list[Choice], raw: str) -> MoveProposal:
        """Parse a raw completion, update recurrent memory, and return the move
        (mirrors `propose`'s success/fallback branches without the SDK loop)."""
        n = len(choices)
        parsed = _parse(raw)
        if parsed is not None and 0 <= parsed["index"] < n:
            reasoning = (str(parsed.get("reasoning"))
                         if parsed.get("reasoning") is not None else None)
            plan = (str(parsed.get("plan"))
                    if parsed.get("plan") is not None else None)
            strategy = (str(parsed.get("strategy"))
                        if parsed.get("strategy") is not None else None)
            if strategy and not self.strategy:
                self.strategy = strategy
            summary = (str(parsed.get("summary"))
                       if parsed.get("summary") is not None else None)
            if summary:
                self.running_summary = summary
            self._remember(choices[parsed["index"]].label, plan)
            return MoveProposal(index=parsed["index"], justification=reasoning,
                                plan=plan)
        self.parse_failures += 1
        return MoveProposal(index=_fallback_index(choices),
                            justification="fallback: unparseable model reply", plan=None)
