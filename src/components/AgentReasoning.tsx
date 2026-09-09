// Watch Agents: the LLM's reasoning for its recent moves. Populated only in
// agent-vs-agent mode (view.agent_log); newest pinned into view. Auto-scroll
// touches only this box's scrollTop (see EventLog for why).

import { useEffect, useRef } from "react";
import type { AgentMoveView } from "../types";

export default function AgentReasoning({ moves }: { moves: AgentMoveView[] }) {
  const boxRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = boxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [moves.length]);

  return (
    <div className="panel log-panel">
      <h2>Agent reasoning</h2>
      <div className="event-log" ref={boxRef}>
        {moves.map((m) => (
          <div key={m.decision_no} className="ev">
            <span className={m.side === "US" ? "side-us" : "side-prc"}>{m.side}</span>{" "}
            <span style={{ opacity: 0.75 }}>{m.label}</span>
            <div style={{ opacity: 0.85, marginTop: 2 }}>{m.reasoning}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
