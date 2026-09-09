# AFWIP — Web Edition

Play the AFWIP classroom wargame **in your browser** — no install, no account,
no server. Open the link and play.

> **Live:** https://alexshaffer78.github.io/afwip-web/

The entire game — the verified rules engine and the trained AI opponents — runs
**client-side** in your browser via [Pyodide](https://pyodide.org) (the real
Python engine compiled to WebAssembly) and
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) (the AI models).
Nothing you do is sent anywhere; there is no backend.

## Play modes

- **Practice vs Agent** — play one side against the computer.
- **Watch Agents** — spectate an AI-vs-AI game.
- **Two Players (one device)** — hotseat, passing the screen between turns.

Opponents: **Random** and three trained difficulties — **Easy / Medium / Best**
(the same ONNX models as the desktop version).

### Not in the web edition
- **Two-device play** — needs a server to coordinate two clients; a static page
  has none. Use the downloadable Python version for LAN play.
- **GPT opponent** — OpenAI's API blocks direct browser calls (CORS), so it
  can't run from a static page. It's available in the Python version.

First load downloads the engine (~20 MB, cached afterward), then it starts in a
second or two.

## How it's built

```
React + TS (src/)  ──►  api.ts  ──►  Pyodide bridge (src/pyodide/boot.ts)
                                          │
                                          ├─ afwip/ (Python engine, from public/afwip.zip)
                                          └─ onnxruntime-web (models/*.onnx)  ← AI opponents
```

The frontend talks to a single seam (`src/api.ts`); in this build that seam runs
the engine in-browser instead of calling a server. `python/afwip/` is a vendored,
trimmed copy of the game engine (rules, env, view, session) plus a browser
adapter (`web/browser_api.py`); the build zips it into `public/afwip.zip`, which
Pyodide unpacks at runtime.

## Develop / build locally

Requires Node 18+ (and `zip`, used by the prebuild step).

```bash
npm install
npm run dev        # local dev server
npm run build      # -> dist/ (static site)
npm run preview    # serve the production build
```

`npm run build` first runs `scripts/pack-python.sh` to bundle `python/afwip`
into `public/afwip.zip`, then builds the static site into `dist/`.

## Deploy

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds and
publishes `dist/` to GitHub Pages. (Repo → Settings → Pages → Source: **GitHub
Actions**.) The Vite `base` is `/afwip-web/`; change it in `vite.config.ts` if
you rename the repo or use a custom domain.
