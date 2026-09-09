// Tournament clock: an optional per-turn countdown and a per-ATO countdown,
// shown during play. Purely a client-side pacing aid — the engine does not
// enforce it. The turn clock resets each turn (turnKey = turn_number); the ATO
// clock resets each ATO (atoKey = ato_cycle). Both tick only while `running`
// (a live player turn, not setup / handoff / game over) and are allowed to go
// into overtime (shown in red, counting up past 0).

import { useEffect, useRef, useState } from "react";

function fmt(secs: number): string {
  const over = secs < 0;
  const s = Math.abs(secs);
  const mm = Math.floor(s / 60);
  const ss = String(s % 60).padStart(2, "0");
  return `${over ? "-" : ""}${mm}:${ss}`;
}

function rowClass(left: number, warnAt: number): string {
  if (left < 0) return "timer-row over";
  if (left <= warnAt) return "timer-row warn";
  return "timer-row";
}

export default function Timer({
  turnSecs, atoSecs, turnKey, atoKey, running, onTurnExpire, onAtoExpire,
}: {
  turnSecs: number;   // per-turn limit in seconds; 0 = no turn clock
  atoSecs: number;    // per-ATO limit in seconds; 0 = no ATO clock
  turnKey: number;    // resets the turn clock whenever it changes (turn_number)
  atoKey: number;     // resets the ATO clock whenever it changes (ato_cycle)
  running: boolean;   // tick only during a live player turn
  onTurnExpire?: () => void;   // fired once when the turn clock reaches 0
  onAtoExpire?: () => void;    // fired once when the ATO clock reaches 0
}) {
  const [turnLeft, setTurnLeft] = useState(turnSecs);
  const [atoLeft, setAtoLeft] = useState(atoSecs);

  // Restart the turn clock at the top of each new turn.
  useEffect(() => { if (turnSecs > 0) setTurnLeft(turnSecs); }, [turnKey, turnSecs]);
  // Restart the ATO clock at the top of each new ATO.
  useEffect(() => { if (atoSecs > 0) setAtoLeft(atoSecs); }, [atoKey, atoSecs]);

  // Fire the expiry callbacks exactly once per turn / ATO, via refs so the
  // callback's changing identity doesn't re-trigger the effect.
  const turnCb = useRef(onTurnExpire); turnCb.current = onTurnExpire;
  const atoCb = useRef(onAtoExpire); atoCb.current = onAtoExpire;
  const turnFired = useRef(false);
  const atoFired = useRef(false);
  useEffect(() => { turnFired.current = false; }, [turnKey]);
  useEffect(() => { atoFired.current = false; }, [atoKey]);
  useEffect(() => {
    if (turnSecs > 0 && turnLeft <= 0 && !turnFired.current) {
      turnFired.current = true;
      turnCb.current?.();
    }
  }, [turnLeft, turnSecs]);
  useEffect(() => {
    if (atoSecs > 0 && atoLeft <= 0 && !atoFired.current) {
      atoFired.current = true;
      atoCb.current?.();
    }
  }, [atoLeft, atoSecs]);

  // One ticker; reads `running` through a ref so the interval isn't torn down
  // and rebuilt on every pause/resume.
  const runningRef = useRef(running);
  runningRef.current = running;
  useEffect(() => {
    if (turnSecs <= 0 && atoSecs <= 0) return;
    const id = setInterval(() => {
      if (!runningRef.current) return;
      if (turnSecs > 0) setTurnLeft((t) => t - 1);
      if (atoSecs > 0) setAtoLeft((a) => a - 1);
    }, 1000);
    return () => clearInterval(id);
  }, [turnSecs, atoSecs]);

  if (turnSecs <= 0 && atoSecs <= 0) return null;
  return (
    <div className="panel timer-panel">
      <h2>Clock{!running && <span className="timer-paused"> paused</span>}</h2>
      {turnSecs > 0 && (
        <div className={rowClass(turnLeft, 15)}>
          <span className="timer-label">Turn</span>
          <span className="timer-clock">{fmt(turnLeft)}</span>
        </div>
      )}
      {atoSecs > 0 && (
        <div className={rowClass(atoLeft, 60)}>
          <span className="timer-label">ATO</span>
          <span className="timer-clock">{fmt(atoLeft)}</span>
        </div>
      )}
    </div>
  );
}
