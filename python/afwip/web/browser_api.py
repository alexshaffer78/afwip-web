"""
browser_api.py — in-browser replacement for the FastAPI server (app.py).

Runs inside Pyodide. Exposes the same 9 operations the frontend's api.ts calls,
but as plain (async) functions over an in-memory `_sessions` registry instead of
HTTP endpoints. Every function returns a JSON-able **envelope**:

    {"ok": True,  "data": <GameView dict | small result>}
    {"ok": False, "status": 400|404|409|422, "detail": "..."}

so the JS bridge can re-raise ApiError with the SAME status codes the server
used (the frontend relies on 409 for stale submissions).

Agent turns are driven by an async runner: RandomAgent is synchronous, but the
ONNX PolicyAgent and the GPT LLMAgent need async browser I/O (onnxruntime-web /
fetch), so `_advance` awaits JS callbacks registered via `set_onnx_runner` /
`set_openai_runner`. The engine, script, and view layers are reused unchanged.
"""

from __future__ import annotations

# The Farama shim MUST be installed before afwip.env is imported (transitively
# via session below), since Pyodide ships neither gymnasium nor pettingzoo.
from afwip import _farama_shim
_farama_shim.install()

import secrets
from typing import Optional

from afwip.core.constants import CAMPAIGN_ATO_CYCLES, Side
from afwip.web.agents import AGENT_REGISTRY, PolicyAgent, LLMAgent
from afwip.web.session import (
    GameMode, GameSession, SessionError, StaleStateError, MAX_AUTO_STEPS,
)

_sessions: dict[str, GameSession] = {}

# Modes offered in the browser build. Two-device (human_v_human) needs a server
# to coordinate two clients, which a static page has none of — so it's excluded.
_WEB_MODES = [m.value for m in GameMode if m != GameMode.HUMAN_V_HUMAN]

# Agents offered in the browser build. The GPT agent is excluded: OpenAI's API
# blocks direct browser (CORS) calls, so it can't run from a static page without
# a relay. It stays available in the downloadable Python version.
_WEB_AGENTS = sorted(a for a in AGENT_REGISTRY if a != "gpt")

# Async JS callbacks, injected at boot. Signatures:
#   onnx_runner(difficulty:str, side:str, feed:dict) -> awaitable list[float]  (256 logits)
#   openai_runner(payload:dict) -> awaitable str                               (raw completion)
_ONNX_RUN = None
_OPENAI_RUN = None


def set_onnx_runner(fn) -> None:
    global _ONNX_RUN
    _ONNX_RUN = fn


def set_openai_runner(fn) -> None:
    global _OPENAI_RUN
    _OPENAI_RUN = fn


# --------------------------------------------------------------------------- #
# Envelope helpers
# --------------------------------------------------------------------------- #

def _ok(data) -> dict:
    return {"ok": True, "data": data}


def _err(status: int, detail: str) -> dict:
    return {"ok": False, "status": status, "detail": detail}


def _view(session: GameSession, viewer: Optional[Side] = None) -> dict:
    # mode="json" makes enums/nested models JSON-safe, matching the HTTP wire.
    return session.state(viewer_override=viewer).model_dump(mode="json")


def _side(value: Optional[str]) -> Optional[Side]:
    if value is None or value == "":
        return None
    try:
        return Side(value)
    except ValueError:
        raise _ApiError(422, "viewer must be US or PRC")


class _ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _get(game_id: str) -> GameSession:
    s = _sessions.get(game_id)
    if s is None:
        raise _ApiError(404, f"no game '{game_id}'")
    return s


# --------------------------------------------------------------------------- #
# Async agent runner (ONNX / GPT need browser I/O)
# --------------------------------------------------------------------------- #

def _to_js_feed(feed: dict) -> dict:
    """numpy input dict -> plain {name: {data, dims, type}} for onnxruntime-web."""
    out = {}
    for k, arr in feed.items():
        flat = arr.flatten().tolist()
        if k == "action_mask":
            out[k] = {"data": [int(x) for x in flat], "dims": list(arr.shape),
                      "type": "bool"}
        else:
            out[k] = {"data": [float(x) for x in flat], "dims": list(arr.shape),
                      "type": "float32"}
    return out


def _as_list(x):
    return x.to_py() if hasattr(x, "to_py") else list(x)


async def _agent_move(s: GameSession, node, agent) -> None:
    if isinstance(agent, PolicyAgent):
        if len(node.choices) == 1:               # forced node — no inference
            s._step(0)
            return
        if _ONNX_RUN is None:
            raise _ApiError(500, "AI opponent unavailable (onnx runner not set)")
        feed = agent.build_feed(lambda: s.env.observe(node.side.value))
        logits = _as_list(await _ONNX_RUN(agent.difficulty, node.side.value.lower(),
                                          _to_js_feed(feed)))
        idx = agent.pick_from_logits(logits, node.choices, s.agent_rng)
        s._step(idx)
    elif isinstance(agent, LLMAgent):
        if _OPENAI_RUN is None:
            raise _ApiError(500, "GPT opponent unavailable (openai runner not set)")
        idx, justification = await _llm_move(s, node, agent)
        s._step(idx, justification=justification)
    else:                                        # RandomAgent — synchronous
        s._agent_step(node, agent)


async def _advance(s: GameSession) -> None:
    """Run agent-controlled decisions until a human node or game end."""
    for _ in range(MAX_AUTO_STEPS):
        node = s._node
        if node is None:
            return
        agent = s._agent_for(node)
        if agent is None:
            return
        await _agent_move(s, node, agent)
    raise _ApiError(500, "advance exceeded MAX_AUTO_STEPS")


# --------------------------------------------------------------------------- #
# The 9 operations (mirror app.py)
# --------------------------------------------------------------------------- #

def config() -> dict:
    return _ok({
        "campaigns": sorted(CAMPAIGN_ATO_CYCLES),
        "modes": _WEB_MODES,
        "sides": [s.value for s in Side],
        "agents": _WEB_AGENTS,
    })


async def create_game(payload: dict) -> dict:
    try:
        campaign = int(payload.get("campaign", 2))
        if not 1 <= campaign <= 5:
            return _err(422, "campaign must be 1..5")
        try:
            mode = GameMode(payload.get("mode", "human_v_agent"))
        except ValueError:
            return _err(422, "invalid mode")
        try:
            human_side = Side(payload.get("human_side", "US"))
        except ValueError:
            return _err(422, "human_side must be US or PRC")
        agent = payload.get("agent", "random")
        if agent not in AGENT_REGISTRY:
            return _err(422, f"unknown agent '{agent}'")
        seed = payload.get("seed", None)
        game_id = secrets.token_hex(4)
        session = GameSession(
            game_id, campaign=campaign, mode=mode, human_side=human_side,
            seed=seed, agent_name=agent, record=payload.get("record", None),
            openai_api_key=payload.get("openai_api_key", None),
            auto_advance=False)          # agents driven by _advance (async)
        _sessions[game_id] = session
        if mode == GameMode.HUMAN_V_AGENT:
            await _advance(session)      # let the AI move first if it's PRC/US-agent
        return _ok(_view(session))
    except _ApiError as e:
        return _err(e.status, e.detail)
    except (SessionError, ValueError) as e:
        return _err(400, str(e))


def get_game(game_id: str, viewer: Optional[str] = None) -> dict:
    try:
        return _ok(_view(_get(game_id), _side(viewer)))
    except _ApiError as e:
        return _err(e.status, e.detail)


async def post_choice(game_id: str, payload: dict) -> dict:
    try:
        s = _get(game_id)
        actor = _side(payload.get("viewer"))
        s.apply_human(int(payload["index"]),
                      revision=payload.get("revision"),
                      node_id=payload.get("node_id"),
                      justification=payload.get("justification"),
                      actor_side=actor)
        if not s.roll_pending:
            await _advance(s)
        return _ok(_view(s, actor))
    except _ApiError as e:
        return _err(e.status, e.detail)
    except StaleStateError as e:
        return _err(409, str(e))
    except SessionError as e:
        return _err(400, str(e))


async def post_step(game_id: str) -> dict:
    try:
        s = _get(game_id)
        node = s._node
        if node is None:
            return _err(400, "game is over")
        agent = s._agent_for(node)
        if agent is None:
            return _err(400, f"{node.side.value} is human-controlled; POST a choice")
        await _agent_move(s, node, agent)
        return _ok(_view(s))
    except _ApiError as e:
        return _err(e.status, e.detail)
    except SessionError as e:
        return _err(400, str(e))


async def post_reveal(game_id: str, viewer: Optional[str] = None) -> dict:
    try:
        s = _get(game_id)
        was_pending = s.roll_pending
        s.reveal_roll()                  # clears the pending roll (auto_advance off)
        if was_pending:
            await _advance(s)
        return _ok(_view(s, _side(viewer)))
    except _ApiError as e:
        return _err(e.status, e.detail)
    except SessionError as e:
        return _err(400, str(e))


def post_handoff(game_id: str) -> dict:
    try:
        s = _get(game_id)
        s.handoff()
        return _ok(_view(s))
    except _ApiError as e:
        return _err(e.status, e.detail)
    except SessionError as e:
        return _err(400, str(e))


def delete_game(game_id: str) -> dict:
    if _sessions.pop(game_id, None) is None:
        return _err(404, f"no game '{game_id}'")
    return _ok({"deleted": game_id})


async def dispatch(method: str, payload_json: str) -> str:
    """Single JSON-in / JSON-out entry point for the JS bridge — sidesteps
    JS<->Python proxy marshalling. `payload_json` is a JSON object; returns a
    JSON-encoded envelope. `id` in the payload is the game_id where relevant."""
    import json
    p = json.loads(payload_json) if payload_json else {}
    if method == "config":
        return json.dumps(config())
    if method == "create_game":
        return json.dumps(await create_game(p))
    if method == "get_game":
        return json.dumps(get_game(p["id"], p.get("viewer")))
    if method == "post_choice":
        return json.dumps(await post_choice(p["id"], p))
    if method == "post_step":
        return json.dumps(await post_step(p["id"]))
    if method == "post_reveal":
        return json.dumps(await post_reveal(p["id"], p.get("viewer")))
    if method == "post_handoff":
        return json.dumps(post_handoff(p["id"]))
    if method == "delete_game":
        return json.dumps(delete_game(p["id"]))
    return json.dumps(_err(404, f"unknown method '{method}'"))


# --------------------------------------------------------------------------- #
# GPT move: build messages in Python, run the HTTP call in JS, parse in Python.
# (Filled in by the OpenAI bridge step; kept here so the dispatch above is whole.)
# --------------------------------------------------------------------------- #

async def _llm_move(s: GameSession, node, agent):
    from afwip.core.constants import Phase
    from afwip.rl.state_text import llm_state_text
    setup = s.env.engine.state.phase != Phase.PLAYER_TURN
    state_text = llm_state_text(s.env.engine, node, node.side)
    req = agent.build_request(node.side, node.choices, state_text, setup=setup)
    reply = await _OPENAI_RUN(req)
    idx, justification = agent.parse_reply(node.side, node.choices, reply)
    return idx, justification
