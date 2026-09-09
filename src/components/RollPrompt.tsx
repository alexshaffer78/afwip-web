// Human click-to-roll: shows a "Roll" button; clicking it spins the dice (1 or 2
// per advantage/disadvantage) and settles on the server-authoritative result,
// then reveals the outcome (onReveal). AI agents never see this — they auto-roll.

import { useState } from "react";
import type { DieRoll } from "../types";
import Dice from "./Dice";

const rnd = () => 1 + Math.floor(Math.random() * 4);

export default function RollPrompt({
  rolls,
  onReveal,
  busy,
}: {
  rolls: DieRoll[];
  onReveal: () => void;
  busy: boolean;
}) {
  const [phase, setPhase] = useState<"idle" | "rolling" | "done">("idle");
  const [display, setDisplay] = useState<DieRoll[]>(rolls);

  const hasAdv = rolls.some((r) => r.mode !== "NORMAL");

  const roll = () => {
    if (phase !== "idle" || busy) return;
    setPhase("rolling");
    let ticks = 0;
    const iv = setInterval(() => {
      setDisplay(rolls.map((r) => ({ ...r, dice: r.dice.map(rnd) })));
      if (++ticks >= 8) {
        clearInterval(iv);
        setDisplay(rolls); // settle on the real result
        setPhase("done");
        setTimeout(onReveal, 600); // let the player read it, then reveal the outcome
      }
    }, 70);
  };

  return (
    <div className="panel roll-prompt">
      <h2>Your roll</h2>
      {phase === "idle" ? (
        <button className="roll-btn" onClick={roll} disabled={busy}>
          🎲 Roll
        </button>
      ) : (
        <div className={`roll-stage ${phase}`}>
          <Dice rolls={display} size="lg" />
        </div>
      )}
      <p className="roll-hint">
        {phase === "idle"
          ? hasAdv
            ? "Rolling at advantage/disadvantage — two dice, keep one."
            : "Click to roll the die."
          : phase === "rolling"
          ? "Rolling…"
          : "Locked in."}
      </p>
    </div>
  );
}
