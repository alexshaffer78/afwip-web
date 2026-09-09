// End-of-ATO scoring attribution: captures by name + accrued VP entries.

import type { ScoreReportView } from "../types";

export default function ScoreReport({ report }: { report: ScoreReportView }) {
  return (
    <div className="panel score-report">
      <h2>ATO {report.ato} scoring</h2>
      {report.sides.map((s) => (
        <div className="side-block" key={s.side}>
          <div>
            <span className={s.side === "US" ? "side-us" : "side-prc"}>{s.side}</span>
            {" "}[{s.mission}]: +{s.ato_total} this ATO — campaign total {s.campaign_total}
          </div>
          {s.token_captures.length > 0 && (
            <div className="line">
              tokens: {s.token_captures.map((c) => `${c.label} (+${c.vp})`).join(", ")}
            </div>
          )}
          {s.squadron_captures.length > 0 && (
            <div className="line">
              squadrons: {s.squadron_captures.map((c) => `${c.label} (+${c.vp})`).join(", ")}
            </div>
          )}
          {s.accrued.map((a) => (
            <div className="line" key={a.reason}>{a.reason}: +{a.points}</div>
          ))}
          {s.token_captures.length === 0 && s.squadron_captures.length === 0 &&
            s.accrued.length === 0 && <div className="line">no points scored</div>}
        </div>
      ))}
    </div>
  );
}
