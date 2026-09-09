"""
env.py — PettingZoo AEC environment for AFWIP.

Exposes the FULL decision surface of the game as a sequence of decision nodes,
each of which is "choose one item from a presented list":

  MISSION_PICK / POSTURE_PICK / SQUADRON_PICK / SQUADRON_BASE / ENABLER_PICK
  (drafting), BID_SACRIFICE / FIRST_PLAYER (initiative), INTEL_REVEAL,
  TURN_ACTION (the engine's `legal_actions` list — one sub-action per step,
  the turn ends when the agent picks pass or activates a squadron),
  SPAWN_BAND / ENABLER_BRANCH (enabler & activation parameters), MD_DECLARE
  (defender, mid-opponent-turn), RESPONSE (reaction-card windows), and
  ALLOC_POINT (base-strike damage distribution, one point at a time).

The action space is a position-indexed `Discrete(MAX_CHOICES)`: action i means
"take the i-th choice of the current node". The observation is a dict of
fixed-shape arrays — global scalars, per-entity token/squadron/card features
with fog-of-war applied, and a `(MAX_CHOICES, CHOICE_F)` matrix describing each
presented choice — plus the standard `action_mask`.

The encoding aims to be as close to Markov as fog allows: the scalar block
(built from the named `_SCALAR_LAYOUT` spec) carries side identity, ATO/turn
tempo, phase, initiative/first-player, intel track, cyber, VP, per-context net
roll advantage/disadvantage, primed one-shot buffs (auto-hit / air advantage),
postures spent, unrealized airbase VP, enablers played this ATO, Infantry
Battalion placement/damage, and the public scoring piles; per-entity blocks add
token origin/salvos/status and squadron damage/losses/recovery/range overrides.
Only own-side or publicly-visible enemy state is encoded — hidden hand and
unacquired token identities never leak (the same fog guarantee the web view has).

Control flow lives in `afwip.script.GameScript` (shared with the terminal
visualizer): a generator coroutine drives the RulesEngine, yields
DecisionNodes, and receives the picked Choice back through `.send()`.
`agent_selection` follows the deciding side, including windows that open
during the opponent's turn.

Fidelity matches the interactive harness: spawn bands, OR-card branches,
missile defense, responses, and base-damage allocation are player decisions;
other enabler targets use the engine's defaults (`_default_enabler_play`).
Drafting is sequential (US then PRC) as in the hotseat harness, so the PRC
sees the US posture while drafting — the same mild asymmetry the physical
hotseat game has.

Rewards are zero-sum terminal: +1 win / -1 loss / 0 draw (a `vp_shaping`
coefficient optionally adds a dense, symmetric VP-differential signal).
`reveal=True` disables fog-of-war in observations (debugging aid).

Games are time-limited: `max_turns` (default 150) caps the number of turns
across all ATO cycles; on reaching it the engine finalizes the game and the
leader on victory points at that moment wins (a genuine win/loss, not a draw
truncation). `max_steps` is a separate hard safety net on decision nodes.

    env = AFWIPEnv(campaign=3)
    env.reset(seed=0)
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        action = policy(obs) if not (term or trunc) else None
        env.step(action)
"""

from __future__ import annotations

import random
from typing import Any, Iterator, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from afwip.script import GameScript, NodeType, Choice, DecisionNode  # noqa: F401 (re-exported)
from afwip.core.rules import RulesEngine, RollContext, RollMode, combine_modes
from afwip.core.state import GameState, CardZone
from afwip.core.constants import (
    Side, BandID, TokenType, TokenScoreType, TokenOrigin, Phase, IntelTrack,
    DAMAGE_TO_DESTROY_SQUADRON, AIRBASE_BONUS_DAMAGE_BOXES, INFANTRY_DAMAGE_BOXES,
)
from afwip.core.cards import (
    MISSION_REGISTRY, SQUADRON_REGISTRY, ENABLER_REGISTRY, POSTURE_REGISTRY,
)
from afwip.core.tokens import TOKEN_REGISTRY


# ---------------------------------------------------------------------------
# Encoding tables
# ---------------------------------------------------------------------------

# Hard cap on presented choices. TURN_ACTION is the widest node (moves +
# acquires + shoots scale with living tokens); random full-surface play was
# observed to reach 129 in Campaign 3, so 256 gives ~2x headroom. The env
# raises rather than truncates when exceeded — truncation would bias play.
MAX_CHOICES = 256
MAX_TOKENS = 24            # per-side token slots in the observation

BAND_LIST = list(BandID)
BAND_IDX = {b: i for i, b in enumerate(BAND_LIST)}
TOKEN_TYPES = list(TokenType)
TT_IDX = {t: i for i, t in enumerate(TOKEN_TYPES)}
ALL_CARD_IDS = sorted(set(MISSION_REGISTRY) | set(POSTURE_REGISTRY)
                      | set(SQUADRON_REGISTRY) | set(ENABLER_REGISTRY))
CARD_IDX = {cid: i for i, cid in enumerate(ALL_CARD_IDS)}
KINDS = ["pass", "activate", "move", "acquire", "shoot_air", "shoot_surface",
         "play_enabler", "other"]
KIND_IDX = {k: i for i, k in enumerate(KINDS)}
ALLOC_KINDS = ["squadron", "token", "infantry", "vp"]
SQUADRON_SLOTS = {s: sorted(cid for cid, p in SQUADRON_REGISTRY.items() if p.side == s)
                  for s in Side}
POSTURE_SLOTS = {s: sorted(cid for cid, p in POSTURE_REGISTRY.items() if p.side == s)
                 for s in Side}
MISSION_SLOTS = {s: sorted(cid for cid, p in MISSION_REGISTRY.items() if p.side == s)
                 for s in Side}

N_BANDS = len(BAND_LIST)
N_TT = len(TOKEN_TYPES)
N_CARDS = len(ALL_CARD_IDS)
N_NODES = len(NodeType)

SCORE_TYPES = list(TokenScoreType)
N_SCORE = len(SCORE_TYPES)
SCORE_IDX = {t: i for i, t in enumerate(SCORE_TYPES)}

PHASES = list(Phase)
PHASE_IDX = {p: i for i, p in enumerate(PHASES)}
N_PHASE = len(PHASES)

# Contexts summarised in the derived net-roll-mode block (advantage / neutral /
# disadvantage per side). Missile defense is situational (declared per shot) and
# is left to the choice/token features rather than a standing summary.
ROLL_SUMMARY_CONTEXTS = [RollContext.ACQUIRE, RollContext.AIR_ATTACK,
                         RollContext.SURF_ATTACK, RollContext.BASE_ATTACK]
N_ROLL_CTX = len(ROLL_SUMMARY_CONTEXTS)

# The only bands a Squadron Card (and hence an Infantry Battalion protecting it)
# can occupy — the two airbases and the US Contingency Location.
BASE_BANDS = [BandID.US_AIRBASE, BandID.US_CONTINGENCY_LOCATION, BandID.PRC_AIRBASE]
BASE_BAND_IDX = {b: i for i, b in enumerate(BASE_BANDS)}
N_BASE_BANDS = len(BASE_BANDS)

POSTURE_SLOT_W = 6            # width reserved per side in the posture blocks

TOKEN_F = N_TT + N_BANDS + 8  # type, band, [present, acquired, winch, grounded,
                              #   av, air_salvos, surf_salvos, squadron_origin]
SQUADRON_F = N_TT + 10        # type, [on_board, activated, destroyed, at_cl,
                              #   damage, grounded_toks, tokens_lost, recovered,
                              #   air_range_override, surviving_tokens]
# Card-status rows in the "cards" observation block:
#   0 own hand   1 own enduring   2 own out-of-play (spent single-use)
#   3 own recyclable but not in hand (played/discarded/sitting out this cycle)
#   4 enemy revealed in hand   5 enemy publicly played/spent (seen when played)
CARD_ROWS = 6
CHOICE_F = (N_NODES + len(KINDS) + 1 + N_CARDS + (N_BANDS + 1)
            + (N_TT + 1) + (N_BANDS + 1)                 # actor type/band
            + (N_TT + 1) + (N_BANDS + 1) + 2             # target type/band/acquired/av
            + 3 + (len(ALLOC_KINDS) + 1))                # branch, alloc kind

# ---------------------------------------------------------------------------
# Scalar observation layout — single source of truth.
#
# Each entry is (name, width); offsets and SCALAR_F are derived from it, so a
# field can be inserted without hand-recomputing indices. Fog invariant: every
# per-side field below is either own-side or PUBLIC state (posture, mission,
# intel, cyber, airbase damage, scoring piles, openly-played enabler effects) —
# genuinely hidden enemy state (unrevealed hand identities, unacquired token
# types) is never encoded here. Two ordering constraints the tests depend on:
# the "captures" block MUST stay last, and the 3-wide "attrition" block MUST sit
# immediately before it (ACTIVE_FLAG is keyed off CAPTURES_SCALAR_OFFSET - 3).
_SCALAR_LAYOUT: list[tuple[str, int]] = [
    ("campaign", 5),                       # campaign one-hot (1..5)
    ("tempo", 4),                          # ato frac, total/5, turn/200, passes/2
    ("node_type", N_NODES),                # current decision-node one-hot
    ("initiative", 1),                     # I hold initiative this ATO
    ("vp", 2),                             # my / opp total victory points
    ("cyber", 2),                          # my / opp cyber rate
    ("posture", POSTURE_SLOT_W * 2),       # posture one-hot, mine then opp
    ("mission", POSTURE_SLOT_W * 2),       # mission one-hot, mine then opp
    ("mas", 8),                            # move/acquire/shoot/enabler flags x2
    ("hidden_agg", 3),                     # fog-safe hidden enemy aggregates
    ("own_hand_airbase", 2),               # my hand size, my airbase damage
    ("opp_airbase_clban", 2),              # opp airbase damage, my CL ban
    ("final_ato", 1),                      # this is the last ATO cycle
    # --- completeness block (Markov-state additions) --------------------------
    ("timed_progress", 1),                 # total_turns / max_turns (deadline proximity)
    ("side_is_us", 1),                     # literal seat identity (asymmetric campaigns)
    ("intel", 2),                          # intel-track ADVANTAGE, mine / opp
    ("pending_air_adv", 2),                # primed air-attack advantage, mine / opp
    ("pending_auto_hit", 2),               # primed auto-hit strike, mine / opp
    ("roll_summary", N_ROLL_CTX * 2),      # net roll mode per context, mine / opp
    ("postures_used", POSTURE_SLOT_W * 2), # postures already spent this campaign
    ("airbase_unscored", 2),              # airbase damage not yet scored, mine / opp
    ("enablers_played", 2),                # enabler cards played this ATO, mine / opp
    ("infantry", (N_BASE_BANDS + 1) * 2),  # battalion presence-by-band + damage x2
    ("opp_cl_banned", 1),                  # opponent's CL ban (mine is above)
    ("phase", N_PHASE),                    # engine phase one-hot
    ("first_player", 1),                   # I am first player this ATO
    ("rule_of_law", 4),                    # squad activations / enabler tokens x2
    ("air_units_killed", 2),               # C4 air-unit kill credits, mine / opp
    ("extra_slots", 2),                    # bonus squadron slots, mine / opp
    ("cl_max", 2),                         # Flying Crew Chief active, mine / opp
    # --- tail (must remain last, in this order) -------------------------------
    ("attrition", 3),                      # active-side flag, off-board naval x2
    ("captures", (N_SCORE + 1) * 2),       # scoring piles by score type + card x2
]
SCALAR_OFF: dict[str, int] = {}
_acc = 0
for _name, _w in _SCALAR_LAYOUT:
    SCALAR_OFF[_name] = _acc
    _acc += _w
SCALAR_F = _acc

# Offsets into the scalar vector (for trainers/tests): posture and mission
# blocks are 2 sides x 6 slots (own first, then opponent).
POSTURE_SCALAR_OFFSET = SCALAR_OFF["posture"]
MISSION_SCALAR_OFFSET = SCALAR_OFF["mission"]
CAPTURES_SCALAR_OFFSET = SCALAR_OFF["captures"]


def _one_hot(vec: np.ndarray, offset: int, idx: Optional[int]) -> int:
    """Set vec[offset + idx] when idx is not None; return the block's end."""
    if idx is not None:
        vec[offset + idx] = 1.0
    return offset


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AFWIPEnv(AECEnv):
    """PettingZoo AEC environment over the AFWIP rules engine (full game)."""

    metadata = {"name": "afwip_v0", "render_modes": ["ansi"], "is_parallelizable": False}

    def __init__(self, campaign: int = 3, reveal: bool = False,
                 vp_shaping: float = 0.0, max_turns: int = 150,
                 max_steps: int = 20000, render_mode: Optional[str] = None):
        super().__init__()
        self.campaign = campaign
        self.reveal = reveal
        self.vp_shaping = vp_shaping
        # Timed-game cap (real games are time-limited): the game ends by VP once
        # this many turns have been played across all ATO cycles. C2 games run a
        # median ~109 turns, so 150 lightly trims only the long tail. `max_steps`
        # is a separate hard safety net on decision NODES (not turns).
        self.max_turns = max_turns
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.possible_agents = [Side.US.value, Side.PRC.value]
        self._action_spaces = {a: spaces.Discrete(MAX_CHOICES) for a in self.possible_agents}
        obs_dict = spaces.Dict({
            "scalars": spaces.Box(-np.inf, np.inf, (SCALAR_F,), np.float32),
            "own_tokens": spaces.Box(0.0, np.inf, (MAX_TOKENS, TOKEN_F), np.float32),
            "enemy_tokens": spaces.Box(0.0, np.inf, (MAX_TOKENS, TOKEN_F), np.float32),
            "own_squadrons": spaces.Box(0.0, np.inf, (len(SQUADRON_SLOTS[Side.US]), SQUADRON_F), np.float32),
            "enemy_squadrons": spaces.Box(0.0, np.inf, (len(SQUADRON_SLOTS[Side.US]), SQUADRON_F), np.float32),
            "cards": spaces.Box(0.0, 1.0, (CARD_ROWS, N_CARDS), np.float32),
            "choices": spaces.Box(-np.inf, np.inf, (MAX_CHOICES, CHOICE_F), np.float32),
        })
        self._observation_spaces = {
            a: spaces.Dict({"observation": obs_dict,
                            "action_mask": spaces.Box(0, 1, (MAX_CHOICES,), np.int8)})
            for a in self.possible_agents
        }

        self.engine: Optional[RulesEngine] = None
        self._node: Optional[DecisionNode] = None
        self._gen: Optional[Iterator[DecisionNode]] = None

    # -- spaces ---------------------------------------------------------------

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    # -- PettingZoo lifecycle ---------------------------------------------------

    def reset(self, seed: Optional[int] = None, options=None):
        rng = random.Random(seed) if seed is not None else random.Random()
        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self.engine = RulesEngine(
            GameState.new_game(campaign=self.campaign, max_turns=self.max_turns),
            rng=rng)
        self._steps = 0
        self._last_vp_diff = 0.0
        self._gen = GameScript(self.engine).run()
        try:
            self._node = next(self._gen)
        except StopIteration:   # pragma: no cover - a game cannot end with no decisions
            self._node = None
            self._finish()
            return
        self.agent_selection = self._node.side.value
        self._set_infos()

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        node = self._node
        idx = int(action)
        if not 0 <= idx < len(node.choices):
            raise ValueError(
                f"Action {idx} out of range: node {node.node_type.name} has "
                f"{len(node.choices)} choices (use the action mask)")
        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()
        self._steps += 1

        try:
            self._node = self._gen.send(node.choices[idx])
            self.agent_selection = self._node.side.value
        except StopIteration:
            self._node = None
            self._finish()

        if self.vp_shaping and self.engine is not None:
            diff = (self.engine.total_victory_points(Side.US)
                    - self.engine.total_victory_points(Side.PRC))
            delta = diff - self._last_vp_diff
            self._last_vp_diff = diff
            if delta:
                self.rewards[Side.US.value] += self.vp_shaping * delta
                self.rewards[Side.PRC.value] -= self.vp_shaping * delta

        if self._node is not None and self._steps >= self.max_steps:
            self._node = None
            for a in self.agents:
                self.truncations[a] = True

        self._set_infos()
        self._accumulate_rewards()

    def observe(self, agent):
        side = Side(agent)
        obs = {
            "scalars": self._scalar_features(side),
            "own_tokens": self._token_features(side, own=True),
            "enemy_tokens": self._token_features(side, own=False),
            "own_squadrons": self._squadron_features(side, own=True),
            "enemy_squadrons": self._squadron_features(side, own=False),
            "cards": self._card_features(side),
            "choices": np.zeros((MAX_CHOICES, CHOICE_F), np.float32),
        }
        mask = np.zeros(MAX_CHOICES, np.int8)
        node = self._node
        if node is not None and node.side == side:
            if len(node.choices) > MAX_CHOICES:
                raise RuntimeError(
                    f"Node {node.node_type.name} has {len(node.choices)} choices "
                    f"(> MAX_CHOICES={MAX_CHOICES}); raise the cap")
            for i, c in enumerate(node.choices):
                obs["choices"][i] = self._choice_features(node.node_type, c, side)
                mask[i] = 1
        return {"observation": obs, "action_mask": mask}

    def render(self):
        if self.engine is None:
            return ""
        gs = self.engine.state
        node = self._node.node_type.name if self._node else "GAME_OVER"
        out = (f"ATO {gs.ato_cycle}/{gs.total_ato_cycles} turn {gs.turn_number} "
               f"node={node} agent={getattr(self, 'agent_selection', '-')} "
               f"VP US={self.engine.total_victory_points(Side.US)} "
               f"PRC={self.engine.total_victory_points(Side.PRC)}")
        if self.render_mode == "ansi":
            return out
        print(out)

    def close(self):
        pass

    # -- termination -----------------------------------------------------------

    def _finish(self) -> None:
        eng = self.engine
        us_vp = eng.total_victory_points(Side.US)
        prc_vp = eng.total_victory_points(Side.PRC)
        if eng.state.winner is not None:
            winner = eng.state.winner
        elif us_vp != prc_vp:
            winner = Side.US if us_vp > prc_vp else Side.PRC
        else:
            winner = None
        for a in self.agents:
            self.terminations[a] = True
            if winner is not None:
                self.rewards[a] += 1.0 if a == winner.value else -1.0

    def _set_infos(self) -> None:
        node = self._node
        info: dict[str, Any] = {}
        if node is not None:
            info = {"node": node.node_type.name, "side": node.side.value,
                    "num_choices": len(node.choices),
                    "choice_labels": [c.label for c in node.choices]}
        elif self.engine is not None:
            info = {"us_vp": self.engine.total_victory_points(Side.US),
                    "prc_vp": self.engine.total_victory_points(Side.PRC),
                    "winner": self.engine.state.winner.value if self.engine.state.winner else None}
        for a in self.agents:
            self.infos[a] = dict(info)

    # =====================================================================
    # Observation encoding
    # =====================================================================

    def _scalar_features(self, side: Side) -> np.ndarray:
        eng = self.engine
        gs = eng.state
        me, opp = gs.player(side), gs.opponent(side)
        v = np.zeros(SCALAR_F, np.float32)
        off = SCALAR_OFF

        v[off["campaign"] + gs.campaign - 1] = 1.0
        b = off["tempo"]
        v[b] = gs.ato_cycle / gs.total_ato_cycles
        v[b + 1] = gs.total_ato_cycles / 5.0
        v[b + 2] = gs.turn_number / 200.0
        v[b + 3] = gs.consecutive_passes / 2.0
        if self._node is not None:
            v[off["node_type"] + self._node.node_type.value] = 1.0
        v[off["initiative"]] = 1.0 if gs.initiative_holder == side else 0.0
        v[off["vp"]] = eng.total_victory_points(side) / 50.0
        v[off["vp"] + 1] = eng.total_victory_points(opp.side) / 50.0
        v[off["cyber"]] = me.cyber_rate / 4.0
        v[off["cyber"] + 1] = opp.cyber_rate / 4.0
        for j, player in enumerate((me, opp)):    # posture / mission one-hots (public)
            if player.posture_card_id is not None:
                v[off["posture"] + j * POSTURE_SLOT_W
                  + POSTURE_SLOTS[player.side].index(player.posture_card_id)] = 1.0
            if player.mission_card_id is not None:
                v[off["mission"] + j * POSTURE_SLOT_W
                  + MISSION_SLOTS[player.side].index(player.mission_card_id)] = 1.0
        b = off["mas"]
        for j, player in enumerate((me, opp)):    # MAS flags (public actions)
            v[b + j * 4 + 0] = float(player.has_moved)
            v[b + j * 4 + 1] = float(player.has_acquired)
            v[b + j * 4 + 2] = float(player.has_shot)
            v[b + j * 4 + 3] = float(player.has_played_enabler)
        # Hidden enemy squadron/hand counts (fog-safe aggregates).
        b = off["hidden_agg"]
        hidden_sq = [s for s in opp.squadrons.values()
                     if not s.activated and not s.is_destroyed and s.zone == CardZone.SELECTED]
        v[b] = sum(1 for s in hidden_sq if s.location != BandID.US_CONTINGENCY_LOCATION) / 5.0
        v[b + 1] = sum(1 for s in hidden_sq if s.location == BandID.US_CONTINGENCY_LOCATION) / 5.0
        v[b + 2] = sum(1 for c in opp.enablers_in_hand() if not c.revealed_to_opponent) / 6.0
        b = off["own_hand_airbase"]
        v[b] = len(me.enablers_in_hand()) / 6.0
        v[b + 1] = me.airbase_vp_damage / AIRBASE_BONUS_DAMAGE_BOXES
        b = off["opp_airbase_clban"]
        v[b] = opp.airbase_vp_damage / AIRBASE_BONUS_DAMAGE_BOXES
        v[b + 1] = float(me.cl_banned_campaign)
        v[off["final_ato"]] = float(gs.is_final_ato())

        # -- completeness block (Markov-state additions) ----------------------
        # Timed-game deadline proximity: fraction of the turn cap consumed. The
        # game ends by VP at the cap, so play near the deadline differs.
        cap = gs.max_turns if gs.max_turns else 200
        v[off["timed_progress"]] = min(1.0, gs.total_turns / cap)
        # Seat identity — obs is otherwise side-relative, but C2/C4/C5 are
        # asymmetric (bans, scoring), so the net must know which side it plays.
        v[off["side_is_us"]] = 1.0 if side == Side.US else 0.0
        # Intel track (public) — ADVANTAGE biases acquire/intel rolls.
        b = off["intel"]
        v[b] = float(me.intel_track == IntelTrack.ADVANTAGE)
        v[b + 1] = float(opp.intel_track == IntelTrack.ADVANTAGE)
        # Primed one-shot buffs from openly-played enablers.
        b = off["pending_air_adv"]
        v[b] = float(me.pending_air_advantage)
        v[b + 1] = float(opp.pending_air_advantage)
        b = off["pending_auto_hit"]
        v[b] = float(me.pending_auto_hit)
        v[b + 1] = float(opp.pending_auto_hit)
        # Derived net roll mode per context, per side (+1 adv / 0 / -1 disadv),
        # folding standing enduring-enabler and posture biases so the policy need
        # not re-derive the modifier table. Token- and band-specific biases
        # (Badger Surge H-6K, EC-130 co-location) stay in the token/choice
        # features (token=None here avoids leaking any hidden identity).
        b = off["roll_summary"]
        for j, roller in enumerate((me, opp)):
            defender = opp if roller is me else me
            for k, ctx in enumerate(ROLL_SUMMARY_CONTEXTS):
                modes = (eng._enduring_modes(roller.side, ctx, None)
                         + eng._posture_modes(roller.side, ctx, defender.side, None))
                net = combine_modes(modes)
                v[b + j * N_ROLL_CTX + k] = (
                    1.0 if net == RollMode.ADVANTAGE
                    else -1.0 if net == RollMode.DISADVANTAGE else 0.0)
        # Postures already spent this campaign (public; constrains future drafts).
        b = off["postures_used"]
        for j, player in enumerate((me, opp)):
            for slot, cid in enumerate(POSTURE_SLOTS[player.side]):
                if POSTURE_REGISTRY[cid].posture_type in player.postures_used:
                    v[b + j * POSTURE_SLOT_W + slot] = 1.0
        # Airbase damage not yet paid out as VP (the future-scoring delta).
        b = off["airbase_unscored"]
        v[b] = max(0, me.airbase_vp_damage - me.airbase_vp_scored) / AIRBASE_BONUS_DAMAGE_BOXES
        v[b + 1] = max(0, opp.airbase_vp_damage - opp.airbase_vp_scored) / AIRBASE_BONUS_DAMAGE_BOXES
        # Enablers played so far this ATO (N-K / Three Dominances / Counter-Int.).
        b = off["enablers_played"]
        v[b] = len(me.enablers_played_log) / 6.0
        v[b + 1] = len(opp.enablers_played_log) / 6.0
        # Infantry Battalion(s): presence over base bands + summed damage, per side.
        b = off["infantry"]
        for j, player in enumerate((me, opp)):
            base = b + j * (N_BASE_BANDS + 1)
            dmg = 0
            for band, taken in player.infantry_battalions.items():
                if band in BASE_BAND_IDX:
                    v[base + BASE_BAND_IDX[band]] = 1.0
                dmg += taken
            v[base + N_BASE_BANDS] = dmg / INFANTRY_DAMAGE_BOXES
        v[off["opp_cl_banned"]] = float(opp.cl_banned_campaign)
        v[off["phase"] + PHASE_IDX[gs.phase]] = 1.0
        v[off["first_player"]] = 1.0 if gs.first_player == side else 0.0
        # Rule of Law per-turn accumulators (public).
        b = off["rule_of_law"]
        for j, player in enumerate((me, opp)):
            v[b + j * 2] = len(player.squadrons_activated_this_turn) / 4.0
            v[b + j * 2 + 1] = player.enabler_tokens_generated_this_turn / 4.0
        # Campaign-4 air-unit kill credits (cap 2), bonus slots, Flying Crew Chief.
        b = off["air_units_killed"]
        v[b] = len(me.air_units_killed) / 2.0
        v[b + 1] = len(opp.air_units_killed) / 2.0
        b = off["extra_slots"]
        v[b] = me.extra_squadron_slots / 4.0
        v[b + 1] = opp.extra_squadron_slots / 4.0
        b = off["cl_max"]
        v[b] = float(me.cl_max_squadron_id is not None)
        v[b + 1] = float(opp.cl_max_squadron_id is not None)

        # -- tail: attrition + capture (scoring) piles ------------------------
        b = off["attrition"]
        v[b] = 1.0 if gs.active_side == side else 0.0
        v[b + 1] = sum(1 for t in me.tokens.values() if t.off_board) / 4.0
        v[b + 2] = sum(1 for t in opp.tokens.values() if t.off_board) / 4.0
        b = off["captures"]
        for j, player in enumerate((me, opp)):
            base = b + j * (N_SCORE + 1)
            for c in player.captures:
                if c.is_squadron_card:
                    v[base + N_SCORE] += 0.25
                elif c.score_type is not None:
                    v[base + SCORE_IDX[c.score_type]] += 0.25
        return v

    def _token_features(self, side: Side, own: bool) -> np.ndarray:
        gs = self.engine.state
        viewer_own = gs.player(side) if own else gs.opponent(side)
        arr = np.zeros((MAX_TOKENS, TOKEN_F), np.float32)
        toks = sorted(viewer_own.living_tokens(), key=lambda t: t.uid)[:MAX_TOKENS]
        for i, t in enumerate(toks):
            known = own or self.reveal or t.acquired
            row = arr[i]
            if known:
                row[TT_IDX[t.token_type]] = 1.0
            base = N_TT
            row[base + BAND_IDX[t.location]] = 1.0
            base += N_BANDS
            row[base] = 1.0                                     # present
            row[base + 1] = float(t.acquired)
            row[base + 4] = t.profile.acquisition_value / 4.0   # av is public info
            if known:
                row[base + 2] = float(t.is_winchester)
                row[base + 3] = float(t.grounded)
                row[base + 5] = (t.air_salvos_remaining or 0) / 4.0
                row[base + 6] = (t.surf_salvos_remaining or 0) / 4.0
                row[base + 7] = float(t.origin == TokenOrigin.SQUADRON_CARD)
        return arr

    def _squadron_features(self, side: Side, own: bool) -> np.ndarray:
        gs = self.engine.state
        player = gs.player(side) if own else gs.opponent(side)
        slots = SQUADRON_SLOTS[player.side]
        arr = np.zeros((len(slots), SQUADRON_F), np.float32)
        for i, cid in enumerate(slots):
            squad = player.squadrons.get(cid)
            if squad is None:
                continue
            # Enemy face-down squadrons are hidden until first activated (a
            # public act — identity stays known in later cycles) or destroyed;
            # face-down counts appear in scalars. Own cards are always shown,
            # including roster cards sitting a cycle out (zone DECK).
            known = own or self.reveal or squad.ever_activated or squad.is_destroyed
            if not known:
                continue
            row = arr[i]
            row[TT_IDX[squad.token_type]] = 1.0
            base = N_TT
            row[base] = float(squad.zone in (CardZone.SELECTED, CardZone.ACTIVE))
            row[base + 1] = float(squad.activated)
            row[base + 2] = float(squad.is_destroyed)
            row[base + 3] = float(squad.location == BandID.US_CONTINGENCY_LOCATION)
            row[base + 4] = squad.damage / DAMAGE_TO_DESTROY_SQUADRON
            row[base + 5] = len(squad.grounded_token_uids) / 4.0
            # Permanent token losses (destroyed tokens do not regenerate) —
            # public once the squadron is known: kills happen on the board.
            row[base + 6] = squad.tokens_lost / 4.0
            # Rapid Resupply recovery flag (risky reactivation) and any
            # Munitions Upgrade air-range override on this squadron.
            row[base + 7] = float(squad.recovered)
            override = player.air_range_override.get(cid)
            row[base + 8] = (override / 4.0) if override is not None else 0.0
            # Surviving / re-fieldable tokens = full complement minus permanent
            # losses (0 once the card is destroyed). Drives later-ATO drafting:
            # a gutted-but-alive squadron re-fields fewer tokens next cycle.
            full = TOKEN_REGISTRY[squad.token_type].token_count
            surviving = 0 if squad.is_destroyed else max(0, full - squad.tokens_lost)
            row[base + 9] = surviving / 4.0
        return arr

    def _card_features(self, side: Side) -> np.ndarray:
        gs = self.engine.state
        me, opp = gs.player(side), gs.opponent(side)
        arr = np.zeros((CARD_ROWS, N_CARDS), np.float32)
        for c in me.enablers.values():
            idx = CARD_IDX[c.card_id]
            if c.zone == CardZone.SELECTED:
                arr[0, idx] = 1.0                    # in hand
            if c.enduring:
                arr[1, idx] = 1.0                    # active effect this cycle
            if c.zone == CardZone.REMOVED:
                arr[2, idx] = 1.0                    # spent for the campaign
            elif c.zone in (CardZone.PLAYED, CardZone.ACTIVE, CardZone.DISCARDED,
                            CardZone.DECK):
                arr[3, idx] = 1.0                    # returns next cycle
        for c in opp.enablers.values():
            idx = CARD_IDX[c.card_id]
            if c.zone == CardZone.SELECTED and (c.revealed_to_opponent or self.reveal):
                arr[4, idx] = 1.0                    # revealed in enemy hand
            elif c.zone in (CardZone.PLAYED, CardZone.ACTIVE, CardZone.REMOVED,
                            CardZone.DISCARDED):
                arr[5, idx] = 1.0                    # played/discarded openly, or spent
        return arr

    def _choice_features(self, node_type: NodeType, c: Choice, side: Side) -> np.ndarray:
        gs = self.engine.state
        v = np.zeros(CHOICE_F, np.float32)
        i = 0
        v[i + node_type.value] = 1.0; i += N_NODES
        v[i + KIND_IDX.get(c.kind, KIND_IDX["other"])] = 1.0; i += len(KINDS)
        v[i] = float(c.kind in ("done", "skip") or c.value is None); i += 1
        if c.card_id is not None and c.card_id in CARD_IDX:
            v[i + CARD_IDX[c.card_id]] = 1.0
        i += N_CARDS
        v[i + (BAND_IDX[c.band] if c.band is not None else N_BANDS)] = 1.0
        i += N_BANDS + 1
        actor = gs.get_token(c.actor_uid) if c.actor_uid is not None else None
        v[i + (TT_IDX[actor.token_type] if actor else N_TT)] = 1.0
        i += N_TT + 1
        v[i + (BAND_IDX[actor.location] if actor else N_BANDS)] = 1.0
        i += N_BANDS + 1
        target = gs.get_token(c.target_uid) if c.target_uid is not None else None
        t_known = target is not None and (self.reveal or target.acquired
                                          or target.side == side)
        v[i + (TT_IDX[target.token_type] if t_known else N_TT)] = 1.0
        i += N_TT + 1
        v[i + (BAND_IDX[target.location] if target else N_BANDS)] = 1.0
        i += N_BANDS + 1
        if target is not None:
            v[i] = float(target.acquired)
            v[i + 1] = target.profile.acquisition_value / 4.0
        i += 2
        v[i + (c.branch_idx + 1 if c.branch_idx is not None else 0)] = 1.0
        i += 3
        alloc = ALLOC_KINDS.index(c.alloc_kind) + 1 if c.alloc_kind in ALLOC_KINDS else 0
        v[i + alloc] = 1.0
        return v


def env(**kwargs) -> AFWIPEnv:
    """PettingZoo-conventional constructor."""
    return AFWIPEnv(**kwargs)
