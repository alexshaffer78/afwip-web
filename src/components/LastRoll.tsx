// Prominent display of the most recent dice roll(s), so they don't scroll away
// in the event log. Reads the newest "dice" events from the fog-filtered log —
// and shows EVERY dice event from that latest step, because a single step can
// produce more than one roll (e.g. the initiative bid AND the winner's cyber
// raise both resolve on the Bid-for-Initiative step). Showing only the last one
// hid the bid behind the cyber roll.

import type { EventView, DieRoll } from "../types";
import Dice from "./Dice";

interface RollItem {
  rolls: DieRoll[];
  label: string | null;
}

function rollsOf(e: EventView): DieRoll[] | undefined {
  return e.type === "dice" ? (e.data?.rolls as DieRoll[] | undefined) : undefined;
}

function latestRollGroup(events: EventView[]): { step: number; items: RollItem[] } | null {
  let maxStep: number | null = null;
  for (const e of events) {
    const rolls = rollsOf(e);
    if (Array.isArray(rolls) && rolls.length && (maxStep === null || e.step >= maxStep)) {
      maxStep = e.step;
    }
  }
  if (maxStep === null) return null;
  const items: RollItem[] = [];
  for (const e of events) {
    const rolls = rollsOf(e);
    if (e.step === maxStep && Array.isArray(rolls) && rolls.length) {
      items.push({ rolls, label: typeof e.data?.label === "string" ? e.data.label : null });
    }
  }
  return { step: maxStep, items };
}

export default function LastRoll({ events }: { events: EventView[] }) {
  const latest = latestRollGroup(events);
  if (!latest) return null;
  return (
    <div className="panel last-roll">
      <h2>
        Last roll <span className="last-roll-step">[{latest.step}]</span>
      </h2>
      {latest.items.map((it, i) => (
        <div key={i} className="last-roll-item">
          {it.label && <div className="last-roll-label">{it.label}</div>}
          <Dice rolls={it.rolls} size="lg" />
        </div>
      ))}
    </div>
  );
}
