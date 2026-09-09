"""
afwip.rl — reinforcement-learning package.

Two layers:

  * **Data generation** — `trajectory` (replayable schema + JSONL IO, the
    `TrajectoryRecorder` the web session hangs off its hotseat/copy-state flow,
    and `materialize()` which replays a record through a fresh env to regenerate
    `(obs, action, mask, reward, done, outcome)` per side under the CURRENT
    encoder) plus the LLM move sources (`openai_agent`, `prompts`, `state_text`).

  * **Learner** — a small entity-encoder + pointer policy (`nets.AFWIPPolicy`)
    scoring each candidate action `z(s,a)`, BC-pretrained on the trajectories
    (`dataset`, `bc`) then improved by masked PPO self-play (`selfplay`, `ppo`)
    with separate US/PRC nets. Config in `config`; obs batching in `obs_encode`.

The learner deps (torch, gymnasium, pettingzoo, tensorboard) are already in
`requirements.txt`; the heavy `nets`/`bc`/`ppo` imports are NOT pulled in here so
importing `afwip.rl` for data-gen alone stays torch-free.
"""

from afwip.rl.trajectory import (  # noqa: F401
    SCHEMA_VERSION, DEFAULT_TRAJECTORY_DIR,
    TapeStep, Outcome, Trajectory, Sample,
    TrajectoryRecorder,
    write_trajectory, write_trajectories, read_trajectories,
    materialize, materialize_both,
    MoveProposal, MoveSource, InteractiveMoveSource,
)
