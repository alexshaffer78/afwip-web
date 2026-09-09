// boot.ts — boots the in-browser game engine: Pyodide (running the real AFWIP
// Python) + onnxruntime-web (the trained AI opponents) + a fetch bridge for the
// GPT opponent. Exposes `browserApi`, which mirrors the server's api.ts surface
// so the React app is unchanged. Everything runs client-side; nothing is hosted.

const BASE = import.meta.env.BASE_URL;                 // "/afwip-web/" on Pages
const PYODIDE_VER = "0.26.4";
const ORT_VER = "1.19.2";
const PYODIDE_JS = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VER}/full/pyodide.js`;
const ORT_JS = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/ort.min.js`;
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;

type Progress = (msg: string) => void;

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`failed to load ${src}`));
    document.head.appendChild(s);
  });
}

// --- onnxruntime-web: one InferenceSession per difficulty_side, async run. ---
const _ortSessions: Record<string, any> = {};
async function onnxRunner(difficulty: string, side: string, feed: any) {
  const key = `${difficulty}_${side}`;
  const ort = (window as any).ort;
  if (!_ortSessions[key]) {
    _ortSessions[key] = await ort.InferenceSession.create(
      `${BASE}models/${key}.onnx`, { executionProviders: ["wasm"] });
  }
  const f = feed.toJs ? feed.toJs({ dict_converter: Object.fromEntries }) : feed;
  const feeds: Record<string, any> = {};
  for (const name of Object.keys(f)) {
    const spec = f[name];
    feeds[name] = spec.type === "bool"
      ? new ort.Tensor("bool", Uint8Array.from(spec.data), spec.dims)
      : new ort.Tensor("float32", Float32Array.from(spec.data), spec.dims);
  }
  const out = await (_ortSessions[key]).run(feeds);
  return Array.from(out.logits.data as Float32Array);
}

// --- GPT opponent: the player's key calls OpenAI directly from the browser. ---
// `payload` is assembled in Python (model + messages + params); we return the
// raw assistant text for Python to parse. The key never leaves this device
// except in the request to OpenAI itself.
async function openaiRunner(payload: any) {
  const p = payload.toJs ? payload.toJs({ dict_converter: Object.fromEntries }) : payload;
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${p.api_key}`,
    },
    body: JSON.stringify(p.body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json())?.error?.message ?? detail; } catch { /* */ }
    throw new Error(`OpenAI ${res.status}: ${detail}`);
  }
  const data = await res.json();
  return data.choices?.[0]?.message?.content ?? "";
}

export interface BrowserApi {
  config: () => Promise<any>;
  createGame: (opts: any) => Promise<any>;
  getGame: (id: string, viewer?: string | null) => Promise<any>;
  postChoice: (id: string, index: number, revision: number,
               nodeId: string | null, viewer?: string | null) => Promise<any>;
  step: (id: string) => Promise<any>;
  reveal: (id: string, viewer?: string | null) => Promise<any>;
  handoff: (id: string) => Promise<any>;
  saveTrajectory: (id: string) => Promise<any>;
  deleteGame: (id: string) => Promise<any>;
}

let _booting: Promise<BrowserApi> | null = null;

export function bootEngine(onProgress: Progress = () => {}): Promise<BrowserApi> {
  if (_booting) return _booting;
  _booting = (async () => {
    onProgress("Loading Python runtime…");
    await loadScript(PYODIDE_JS);
    await loadScript(ORT_JS);
    (window as any).ort.env.wasm.wasmPaths = ORT_BASE;

    const pyodide = await (window as any).loadPyodide();
    onProgress("Loading numerics…");
    await pyodide.loadPackage(["numpy", "pydantic"]);

    onProgress("Loading the game…");
    const buf = await (await fetch(`${BASE}afwip.zip`)).arrayBuffer();
    pyodide.unpackArchive(buf, "zip");                // -> afwip/ on sys.path

    const bapi = pyodide.pyimport("afwip.web.browser_api");
    bapi.set_onnx_runner(onnxRunner);
    bapi.set_openai_runner(openaiRunner);

    async function call(method: string, payload?: any): Promise<any> {
      const json = await bapi.dispatch(method, JSON.stringify(payload ?? {}));
      const env = JSON.parse(json);
      if (!env.ok) {
        const e = new Error(env.detail || "request failed") as any;
        e.status = env.status;
        throw e;
      }
      return env.data;
    }

    onProgress("Ready.");
    return {
      config: () => call("config"),
      createGame: (opts) => call("create_game", opts),
      getGame: (id, viewer) => call("get_game", { id, viewer: viewer ?? null }),
      postChoice: (id, index, revision, nodeId, viewer) =>
        call("post_choice", { id, index, revision, node_id: nodeId,
                              viewer: viewer ?? null }),
      step: (id) => call("post_step", { id }),
      reveal: (id, viewer) => call("post_reveal", { id, viewer: viewer ?? null }),
      handoff: (id) => call("post_handoff", { id }),
      saveTrajectory: (id) => call("save_trajectory", { id }),
      deleteGame: (id) => call("delete_game", { id }),
    };
  })();
  return _booting;
}
