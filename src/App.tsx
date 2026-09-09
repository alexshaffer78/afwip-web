// App shell: menu -> game screen. Owns the GameView, submits choices with
// revision/node_id (refreshing on 409), runs the spectator autoplay loop, and
// gates hotseat play behind the pass-the-device overlay.

import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, warmupEngine, type CreateGameOptions } from "./api";
import type { ConfigView, GameView, Highlight } from "./types";
import Board from "./components/Board";
import ChoicePanel from "./components/ChoicePanel";
import CopyForAI from "./components/CopyForAI";
import SaveTrajectory from "./components/SaveTrajectory";
import AgentReasoning from "./components/AgentReasoning";
import EventLog from "./components/EventLog";
import LastRoll from "./components/LastRoll";
import RollPrompt from "./components/RollPrompt";
import { latestRolls } from "./components/Dice";
import NewGame, { type TimerConfig } from "./components/NewGame";
import Timer from "./components/Timer";
import ScorePiles from "./components/ScorePiles";
import ScoreReport from "./components/ScoreReport";
import SidePanels from "./components/SidePanels";
import StatusBar from "./components/StatusBar";

const NO_HIGHLIGHT: Highlight = { actorUid: null, targetUid: null, band: null };
const AUTO_SPEEDS = [
  { label: "slow", ms: 1200 },
  { label: "normal", ms: 500 },
  { label: "fast", ms: 150 },
];

export default function App() {
  const [config, setConfig] = useState<ConfigView | null>(null);
  const [view, setView] = useState<GameView | null>(null);
  const [pending, setPending] = useState<GameView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [highlight, setHighlight] = useState<Highlight>(NO_HIGHLIGHT);
  const [auto, setAuto] = useState(false);
  const [speedIdx, setSpeedIdx] = useState(1);
  const [timers, setTimers] = useState<TimerConfig>({ turn: 0, ato: 0 });
  // When the ATO clock expires in a two-player game we drive passes/handoffs
  // until this ATO cycle ends; this holds the cycle being force-ended.
  const [endAtoCycle, setEndAtoCycle] = useState<number | null>(null);
  // Two-device play: which side THIS device is (null for single-client modes).
  const [myViewer, setMyViewer] = useState<string | null>(null);
  const busyRef = useRef(false);
  // Latest view/pending, for the two-device poll (avoids a stale closure so the
  // interval keeps running and always compares against the current revision).
  const viewRef = useRef<GameView | null>(null);
  const pendingRef = useRef<GameView | null>(null);
  useEffect(() => { viewRef.current = view; }, [view]);
  useEffect(() => { pendingRef.current = pending; }, [pending]);

  // First load boots the in-browser engine (Pyodide + AI runtime, a one-time
  // download); show progress until it's ready, then fetch the config.
  const [bootMsg, setBootMsg] = useState<string | null>("Starting…");
  useEffect(() => {
    warmupEngine((m) => setBootMsg(m))
      .then(() => api.config())
      .then((cfg) => { setConfig(cfg); setBootMsg(null); })
      .catch((e) => { setBootMsg(null); setError(String(e.message ?? e)); });
  }, []);

  const run = useCallback(async (fn: () => Promise<GameView>) => {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    try {
      setView(await fn());
      setError(null);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && view) {
        // Stale submission (double-click / race): refresh instead of erroring.
        setView(await api.getGame(view.game_id, myViewer));
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [view, myViewer]);

  const start = (opts: CreateGameOptions, timerCfg: TimerConfig) => {
    setTimers(timerCfg);
    setEndAtoCycle(null);
    setAuto(false);
    setHighlight(NO_HIGHLIGHT);
    if (opts.mode === "human_v_human") {
      // Creator hosts as their chosen side; fetch that side's fogged view.
      const mine = opts.human_side ?? "US";
      setMyViewer(mine);
      run(async () => {
        const created = await api.createGame(opts);
        return api.getGame(created.game_id, mine);
      });
    } else {
      setMyViewer(null);
      run(() => api.createGame(opts));
    }
  };

  // Join an existing two-device game by code, as a chosen side.
  // A human action that rolls dice comes back with roll_pending: hold the
  // outcome (keep showing the current `view`) and let RollPrompt reveal it.
  const choose = async (index: number) => {
    if (!view || busyRef.current) return;
    setHighlight(NO_HIGHLIGHT);
    busyRef.current = true;
    setBusy(true);
    try {
      const resp = await api.postChoice(view.game_id, index, view.revision, view.node_id,
                                        myViewer);
      setError(null);
      if (resp.roll_pending) setPending(resp);
      else setView(resp);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setView(await api.getGame(view.game_id, myViewer));
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  };

  // The player clicked "Roll" and the dice have settled: reveal the outcome,
  // then let the AI take its turn.
  const revealRoll = async () => {
    if (!pending || busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    try {
      setView(pending);
      setView(await api.reveal(pending.game_id, myViewer));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(null);
      busyRef.current = false;
      setBusy(false);
    }
  };

  // --- tournament clock enforcement -------------------------------------------
  // Submit the current player's "Pass / end turn" if it's an available choice.
  const autoPass = () => {
    if (!view || view.terminal || view.handoff_required || !view.decision || busyRef.current) {
      return;
    }
    const idx = view.decision.choices.findIndex((c) => c.kind === "pass");
    if (idx >= 0) choose(idx);
  };

  // Turn clock hit 0 -> pass this player's turn.
  const onTurnExpire = () => autoPass();

  // ATO clock hit 0 -> end the ATO. In a two-player game we can drive both
  // sides to pass (below); otherwise fall back to passing the current turn.
  const onAtoExpire = () => {
    if (view && view.mode === "hotseat") setEndAtoCycle(view.status.ato_cycle);
    else autoPass();
  };

  // Force-end loop: while an ATO is timed out, auto-confirm handoffs and pass
  // until the cycle advances (both players passing ends the ATO) or the game ends.
  useEffect(() => {
    if (endAtoCycle === null || !view) return;
    if (view.terminal || view.status.ato_cycle !== endAtoCycle) {
      setEndAtoCycle(null);        // the ATO has ended (or the game has)
      return;
    }
    if (busyRef.current) return;
    const t = setTimeout(() => {
      if (busyRef.current) return;
      if (view.handoff_required) {
        run(() => api.handoff(view.game_id));
      } else if (view.decision) {
        const idx = view.decision.choices.findIndex((c) => c.kind === "pass");
        if (idx >= 0) choose(idx);
        else setEndAtoCycle(null);  // no pass available here — stop rather than loop
      }
    }, 250);
    return () => clearTimeout(t);
  }, [view, endAtoCycle, run]);

  // Spectator autoplay loop.
  useEffect(() => {
    if (!auto || !view || view.terminal || view.mode !== "agent_v_agent") return;
    const t = setTimeout(() => run(() => api.step(view.game_id)), AUTO_SPEEDS[speedIdx].ms);
    return () => clearTimeout(t);
  }, [auto, view, speedIdx, run]);

  // Two-device play: poll for the other device's move and refresh on change.
  // A self-sustaining interval (reading live state from refs) — a setTimeout
  // keyed on `view` would stop re-arming once a poll returns the same revision,
  // which stalls sync while it's the other side's turn.
  useEffect(() => {
    if (!view || view.mode !== "human_v_human") return;
    const gameId = view.game_id;
    const id = setInterval(async () => {
      const cur = viewRef.current;
      if (!cur || cur.terminal || pendingRef.current || busyRef.current) return;
      try {
        const fresh = await api.getGame(gameId, myViewer);
        if (fresh.revision !== (viewRef.current?.revision ?? -1)) setView(fresh);
      } catch { /* transient network hiccup — retry next tick */ }
    }, 1500);
    return () => clearInterval(id);
  }, [view?.game_id, view?.mode, myViewer]);

  const quit = () => {
    // In two-device play, leaving shouldn't delete the shared game for the other
    // player — just drop back to the menu.
    if (view && view.mode !== "human_v_human") {
      api.deleteGame(view.game_id).catch(() => undefined);
    }
    setView(null);
    setAuto(false);
    setEndAtoCycle(null);
    setMyViewer(null);
    setError(null);
  };

  if (bootMsg) {
    return (
      <div className="app">
        <div className="boot-screen">
          <h1>AFWIP</h1>
          <div className="subtitle">Air Force Wargame: Indo-Pacific</div>
          <div className="boot-spinner" />
          <div className="boot-msg">{bootMsg}</div>
          <div className="boot-note">
            First load downloads the game engine (~20 MB) — this happens once,
            then it's cached.
          </div>
        </div>
      </div>
    );
  }

  if (!view) {
    return (
      <div className="app">
        <NewGame config={config} onStart={start} error={error} />
      </div>
    );
  }

  const spectate = view.mode === "agent_v_agent";
  return (
    <div className="app">
      <div className="topbar">
        <h1>AFWIP</h1>
        {spectate && !view.terminal && (
          <>
            <button disabled={busy || auto} onClick={() => run(() => api.step(view.game_id))}>
              Next ▸
            </button>
            <button onClick={() => setAuto(!auto)}>{auto ? "Pause ⏸" : "Auto ▶"}</button>
            <select value={speedIdx} onChange={(e) => setSpeedIdx(Number(e.target.value))}>
              {AUTO_SPEEDS.map((s, i) => (
                <option key={s.label} value={i}>{s.label}</option>
              ))}
            </select>
          </>
        )}
        {view.viewer && (
          <span>
            viewing as{" "}
            <span className={view.viewer === "US" ? "side-us" : "side-prc"}>{view.viewer}</span>
          </span>
        )}
        <div className="spacer" />
        <span style={{ color: "var(--text-dim)", fontFamily: "var(--font-mono)", fontSize: 12 }}>
          game {view.game_id} · rev {view.revision}
        </span>
        <button onClick={quit}>New game</button>
      </div>

      {error && <div className="error-banner" style={{ margin: "8px 10px 0" }}>{error}</div>}

      <div className="main">
        <div className="left-col">
          <StatusBar status={view.status} />
          <div className="panel board-panel">
            <Board board={view.board} highlight={highlight} status={view.status}
              turnSide={view.decision?.side ?? view.status.active_side} />
          </div>
          <ScorePiles captures={view.captures} />
          <SidePanels squadrons={view.squadrons} hands={view.hands} />
          {view.score_report && <ScoreReport report={view.score_report} />}
        </div>
        <div className="right-col">
          <Timer
            key={view.game_id}
            turnSecs={timers.turn}
            atoSecs={timers.ato}
            turnKey={view.status.turn_number}
            atoKey={view.status.ato_cycle}
            running={!view.terminal && view.status.phase === "PLAYER_TURN"
                     && !view.handoff_required}
            onTurnExpire={onTurnExpire}
            onAtoExpire={onAtoExpire}
          />
          <LastRoll events={view.event_log} />
          {view.llm_export && <CopyForAI text={view.llm_export} />}
          {pending ? (
            <RollPrompt
              rolls={latestRolls(pending.event_log)}
              onReveal={revealRoll}
              busy={busy}
            />
          ) : (
            view.decision &&
            !view.terminal && (
              <ChoicePanel
                decision={view.decision}
                busy={busy || (spectate && auto)}
                onChoose={spectate ? () => undefined : choose}
                onHover={setHighlight}
              />
            )
          )}
          {view.mode === "human_v_human" && !view.decision && !view.terminal && !pending && (
            <div className="panel" style={{ textAlign: "center", color: "var(--text-dim)" }}>
              Waiting for the other player to move…
            </div>
          )}
          {view.agent_log && view.agent_log.length > 0 && (
            <AgentReasoning moves={view.agent_log} />
          )}
          <EventLog events={view.event_log} />
        </div>
      </div>

      {view.handoff_required && (
        <div className="overlay">
          <div className="card">
            <h2>Pass the device</h2>
            <p>
              Hand the device to the{" "}
              <span className={view.status.active_side === "US" ? "side-us" : "side-prc"}>
                next player
              </span>
              . Their hidden information stays off-screen until they continue.
            </p>
            <button className="primary" onClick={() => run(() => api.handoff(view.game_id))}>
              I'm ready — show my view
            </button>
          </div>
        </div>
      )}

      {view.terminal && (
        <div className="overlay">
          <div className="card">
            <h2>
              {view.winner ? (
                <>
                  <span className={view.winner === "US" ? "side-us" : "side-prc"}>
                    {view.winner}
                  </span>{" "}
                  wins
                </>
              ) : (
                "Draw"
              )}
            </h2>
            <p>
              Final score — US {view.status.vp["US"]} · PRC {view.status.vp["PRC"]}
            </p>
            {view.score_report && <ScoreReport report={view.score_report} />}
            {view.recording && (
              <SaveTrajectory gameId={view.game_id} saved={view.trajectory_saved} />
            )}
            <button className="primary" onClick={quit}>Back to menu</button>
          </div>
        </div>
      )}
    </div>
  );
}
