// Scrolling event log (fog-filtered server-side), newest pinned into view.
// Auto-scroll adjusts ONLY this box's scrollTop — never scrollIntoView, which
// would also scroll every ancestor and yank the whole page to the bottom on
// each action.

import { useEffect, useRef } from "react";
import type { EventView, DieRoll } from "../types";
import Dice from "./Dice";

export default function EventLog({ events }: { events: EventView[] }) {
  const boxRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = boxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [events.length]);

  return (
    <div className="panel log-panel">
      <h2>Event log</h2>
      <div className="event-log" ref={boxRef}>
        {events.map((e, i) => {
          const rolls =
            e.type === "dice" ? (e.data?.rolls as DieRoll[] | undefined) : undefined;
          const label = typeof e.data?.label === "string" ? e.data.label : null;
          return (
            <div key={i} className={`ev t-${e.type} public-${e.public}`}>
              <span style={{ opacity: 0.5 }}>[{e.step}]</span>{" "}
              {e.side && (
                <span className={e.side === "US" ? "side-us" : "side-prc"}>{e.side} </span>
              )}
              {rolls && rolls.length ? (
                <>
                  {label && <span className="dice-label">{label} </span>}
                  <Dice rolls={rolls} />
                </>
              ) : (
                e.message.replace(/^(US|PRC) /, "")
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
