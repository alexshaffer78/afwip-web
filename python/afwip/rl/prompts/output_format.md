# Output format

For each state, choose exactly ONE action and reply with **only** a JSON object:

```json
{"index": <int>, "reasoning": "<= 3 sentences>", "plan": "<next ~5 moves>", "summary": "<3-5 sentence running situation>", "strategy": "<3-5 sentence game plan — first move only>"}
```

You are called once per decision and remember nothing between calls on your own,
so `strategy` and `summary` ARE your memory: they are fed back to you next turn.
When you are given a prior "game strategy" and "running summary", build on them —
do not restart from scratch.

Rules:
- `index` MUST be one of the `ACTION: index=` values in the pasted state. Never
  invent an index or a move that is not listed. Choose for the decision the
  `NODE: type=…` is asking.
- `reasoning`: at most 3 sentences, grounded in your side's doctrine and the
  tactical situation, for THIS move.
- `plan`: **think ~5 moves ahead.** Write the concrete sequence you intend —
  this move, then the next several — including how you expect the OPPONENT to
  respond and how this move SETS UP the ones after it. Pick the current `index`
  as the first step of that plan (only it is played now). Plan toward a concrete
  scoring objective (which enemy unit you will kill, which card you will finish,
  what your force will look like in 5 moves), not a vague intention. A good move
  advances a multi-move plan; a move that fits no plan is usually a wasted turn.
- `strategy`: **only on your FIRST move of the game** — a 3-5 sentence overall
  game plan (how you intend to win this campaign: your force concept, the
  enemy nodes you will target, how you will spend your advantages). It is
  remembered and shown back to you every later turn, so you need not (and should
  not) repeat it; omit the field on later moves, or echo it unchanged. Hold to it
  unless the situation genuinely forces a change.
- `summary`: **every move** — rewrite the running summary you were given into an
  updated 3-5 sentence digest of the situation that matters going forward: what
  has happened, where the forces are, the score, and what you are in the middle
  of executing. Write it for your future self — it is the only per-turn memory
  you carry, so keep it current and drop stale detail.
- Output the JSON object and nothing else.
