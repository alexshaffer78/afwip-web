// Landing menu: pick a mode, campaign, side, seed, and optional per-turn /
// per-ATO time limits. "Tournament Play" is a one-click preset: Campaign 2,
// two players, a 2-minute turn clock and a 48-minute ATO clock.

import { useState } from "react";
import type { ConfigView } from "../types";
import type { CreateGameOptions } from "../api";

export interface TimerConfig {
  turn: number;   // per-turn seconds; 0 = off
  ato: number;    // per-ATO seconds; 0 = off
}

// Friendly labels for the opponent dropdown (keys come from the backend
// AGENT_REGISTRY). Unlisted agents fall back to their raw key.
const AGENT_LABELS: Record<string, string> = {
  "policy-easy": "AI — Easy",
  "policy-medium": "AI — Medium",
  "policy-best": "AI — Best",
  gpt: "GPT (OpenAI)",
  random: "Random",
};

const MODES = [
  {
    id: "human_v_agent",
    title: "Practice vs Agent",
    desc: "Play one side against the computer — the classroom practice mode.",
  },
  {
    id: "agent_v_agent",
    title: "Watch Agents",
    desc: "Spectate an agent-vs-agent game, one decision at a time or on autoplay.",
  },
  {
    id: "hotseat",
    title: "Two Players (one device)",
    desc: "Human vs human on this device, passing it between turns.",
  },
  {
    id: "human_v_human",
    title: "Two Players (network)",
    desc: "Play on two devices on the same network — share the game code below with the other player.",
  },
];

// Time-limit dropdown options (label, seconds). 0 = "Off".
const TURN_OPTS: [string, number][] = [
  ["Off", 0], ["1 min", 60], ["2 min", 120], ["3 min", 180], ["5 min", 300],
];
const ATO_OPTS: [string, number][] = [
  ["Off", 0], ["24 min", 1440], ["36 min", 2160], ["48 min", 2880], ["60 min", 3600],
];

// Tournament preset.
const TOURNEY = {
  opts: { campaign: 2, mode: "hotseat", human_side: "US", seed: null } as CreateGameOptions,
  timers: { turn: 120, ato: 2880 } as TimerConfig,
};

export default function NewGame({ config, onStart, error }: {
  config: ConfigView | null;
  onStart: (opts: CreateGameOptions, timers: TimerConfig) => void;
  error: string | null;
}) {
  const [mode, setMode] = useState("human_v_agent");
  const [campaign, setCampaign] = useState(2);
  const [side, setSide] = useState("US");
  const [agent, setAgent] = useState("random");
  const [seed, setSeed] = useState("");
  const [turnSecs, setTurnSecs] = useState(0);
  const [atoSecs, setAtoSecs] = useState(0);
  // Remembered per-browser so the user doesn't re-paste it each game (it stays
  // in localStorage on this device; the server keeps it only in session memory).
  const [apiKey, setApiKey] = useState(() => {
    try { return localStorage.getItem("afwip_openai_key") ?? ""; } catch { return ""; }
  });

  return (
    <div className="menu">
      <h1>AFWIP</h1>
      <div className="subtitle">Air Force Wargame: Indo-Pacific</div>
      {error && <div className="error-banner">{error}</div>}

      <button
        className="tourney"
        title="Campaign 2, two players, 2-minute turns, 48-minute ATO"
        onClick={() => onStart(TOURNEY.opts, TOURNEY.timers)}
      >
        Tournament Play
        <span className="tourney-sub">Campaign 2 · Two Players · 2 min / turn · 48 min / ATO</span>
      </button>
      <div className="or-divider">or set up a game</div>

      <div className="mode-cards">
        {MODES.filter((m) => (config?.modes
            ?? ["human_v_agent", "agent_v_agent", "hotseat"]).includes(m.id))
          .map((m) => (
          <button
            key={m.id}
            className={`mode-card ${mode === m.id ? "selected" : ""}`}
            onClick={() => setMode(m.id)}
          >
            <div className="title">{m.title}</div>
            <div className="desc">{m.desc}</div>
          </button>
        ))}
      </div>
      <div className="opts">
        <label>
          Campaign{" "}
          <select value={campaign} onChange={(e) => setCampaign(Number(e.target.value))}>
            {(config?.campaigns ?? [1, 2, 3, 4, 5]).map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </label>
        {(mode === "human_v_agent" || mode === "human_v_human") && (
          <label>
            Play as{" "}
            <select value={side} onChange={(e) => setSide(e.target.value)}>
              <option value="US">US</option>
              <option value="PRC">PRC</option>
            </select>
          </label>
        )}
        {mode !== "hotseat" && mode !== "human_v_human" && (
          <label>
            Agent{" "}
            <select value={agent} onChange={(e) => setAgent(e.target.value)}>
              {(config?.agents ?? ["random"]).map((a) => (
                <option key={a} value={a}>{AGENT_LABELS[a] ?? a}</option>
              ))}
            </select>
          </label>
        )}
        {mode !== "hotseat" && agent === "gpt" && (
          <label>
            OpenAI key{" "}
            <input
              type="password"
              style={{ width: 200 }}
              placeholder="sk-..."
              autoComplete="off"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value.trim())}
            />
          </label>
        )}
        <label>
          Seed{" "}
          <input
            style={{ width: 90 }}
            placeholder="random"
            value={seed}
            onChange={(e) => setSeed(e.target.value.replace(/\D/g, ""))}
          />
        </label>
      </div>
      <div className="opts">
        <label>
          Turn time{" "}
          <select value={turnSecs} onChange={(e) => setTurnSecs(Number(e.target.value))}>
            {TURN_OPTS.map(([label, secs]) => (
              <option key={secs} value={secs}>{label}</option>
            ))}
          </select>
        </label>
        <label>
          ATO time{" "}
          <select value={atoSecs} onChange={(e) => setAtoSecs(Number(e.target.value))}>
            {ATO_OPTS.map(([label, secs]) => (
              <option key={secs} value={secs}>{label}</option>
            ))}
          </select>
        </label>
      </div>
      <button
        className="start"
        onClick={() => {
          if (agent === "gpt") {
            try { localStorage.setItem("afwip_openai_key", apiKey); } catch { /* ignore */ }
          }
          onStart(
            {
              campaign,
              mode,
              human_side: side,
              agent,
              seed: seed === "" ? null : Number(seed),
              ...(agent === "gpt" && apiKey ? { openai_api_key: apiKey } : {}),
            },
            { turn: turnSecs, ato: atoSecs },
          );
        }}
      >
        Start Game
      </button>
    </div>
  );
}
