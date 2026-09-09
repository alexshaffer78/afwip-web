// Typed client for the in-browser engine. Same surface the server version
// exposed (so App.tsx/components are unchanged), but every call runs the real
// Python engine locally via Pyodide (see pyodide/boot.ts) instead of hitting a
// server. Errors carry the same status codes — the app relies on 409 for stale
// submissions.

import type { ConfigView, GameView } from "./types";
import { bootEngine, type BrowserApi } from "./pyodide/boot";

export class ApiError extends Error {
  status: number;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
  }
}

export interface CreateGameOptions {
  campaign: number;
  mode: string;
  human_side?: string;
  seed?: number | null;
  agent?: string;
  openai_api_key?: string;   // only sent for the "gpt" agent
}

let _engine: Promise<BrowserApi> | null = null;

// Boot (or reuse) the engine. Progress messages feed the loading overlay; only
// the first caller's callback is used (boot is memoized).
export function warmupEngine(onProgress?: (msg: string) => void): Promise<BrowserApi> {
  if (!_engine) _engine = bootEngine(onProgress);
  return _engine;
}

async function wrap<T>(fn: (e: BrowserApi) => Promise<T>): Promise<T> {
  const engine = await warmupEngine();
  try {
    return await fn(engine);
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(err?.status ?? 500, err?.message ?? String(err));
  }
}

export const api = {
  config: () => wrap((e) => e.config() as Promise<ConfigView>),
  createGame: (opts: CreateGameOptions) =>
    wrap((e) => e.createGame(opts) as Promise<GameView>),
  getGame: (id: string, viewer?: string | null) =>
    wrap((e) => e.getGame(id, viewer) as Promise<GameView>),
  postChoice: (id: string, index: number, revision: number, nodeId: string | null,
               viewer?: string | null) =>
    wrap((e) => e.postChoice(id, index, revision, nodeId, viewer) as Promise<GameView>),
  step: (id: string) => wrap((e) => e.step(id) as Promise<GameView>),
  reveal: (id: string, viewer?: string | null) =>
    wrap((e) => e.reveal(id, viewer) as Promise<GameView>),
  handoff: (id: string) => wrap((e) => e.handoff(id) as Promise<GameView>),
  saveTrajectory: (id: string) =>
    wrap((e) => e.saveTrajectory(id) as Promise<
      { game_id: string; trajectory: unknown; decisions: number }>),
  deleteGame: (id: string) =>
    wrap((e) => e.deleteGame(id) as Promise<{ deleted: string }>),
};
