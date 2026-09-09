// Visual D4 die faces. Advantage/disadvantage shows both rolled dice with the
// chosen one highlighted; a bonus shows the "+N = value" resolution.

import type { DieRoll, EventView } from "../types";

// The most recent roll in a (fog-filtered) event log, or [] if none yet.
export function latestRolls(events: EventView[]): DieRoll[] {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    const rolls = e.type === "dice" ? (e.data?.rolls as DieRoll[] | undefined) : undefined;
    if (Array.isArray(rolls) && rolls.length) return rolls;
  }
  return [];
}

function Die({ n, used }: { n: number; used: boolean }) {
  return <span className={`die ${used ? "die-used" : "die-unused"}`}>{n}</span>;
}

function RollView({ r }: { r: DieRoll }) {
  const badge =
    r.mode === "ADVANTAGE" ? "adv" : r.mode === "DISADVANTAGE" ? "dis" : "";
  // The "used" die is the one equal to the chosen natural value.
  let markedUsed = false;
  return (
    <span className="roll" title={`rolled ${r.dice.join(", ")} → ${r.value}`}>
      {r.role && <span className="roll-role">{r.role}</span>}
      {r.dice.map((d, i) => {
        const used = d === r.natural && !(markedUsed && r.dice.length > 1);
        if (used) markedUsed = true;
        return <Die key={i} n={d} used={used} />;
      })}
      {badge && <span className="roll-mode">{badge}</span>}
      {r.bonus !== 0 && (
        <span className="roll-bonus">
          {r.bonus > 0 ? `+${r.bonus}` : r.bonus} = {r.value}
        </span>
      )}
    </span>
  );
}

export default function Dice({
  rolls,
  size,
}: {
  rolls: DieRoll[];
  size?: "lg";
}) {
  return (
    <span className={`dice-faces${size === "lg" ? " dice-lg" : ""}`}>
      {rolls.map((r, i) => (
        <RollView key={i} r={r} />
      ))}
    </span>
  );
}
