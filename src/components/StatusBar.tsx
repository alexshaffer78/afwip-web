// Campaign / ATO / turn / initiative / VP / cyber / mission / posture strip.

import type { StatusView } from "../types";

function SideStat({ label, status, field }: {
  label: string;
  status: StatusView;
  field: "vp" | "cyber";
}) {
  return (
    <span>
      <span className="k">{label}</span>
      <span className="side-us">US {status[field]["US"]}</span>
      {" / "}
      <span className="side-prc">PRC {status[field]["PRC"]}</span>
    </span>
  );
}

export default function StatusBar({ status }: { status: StatusView }) {
  return (
    <div className="panel">
      <div className="status-strip">
        <span>
          <span className="k">CAMPAIGN</span>{status.campaign}
        </span>
        <span>
          <span className="k">ATO</span>
          {status.ato_cycle}/{status.total_ato_cycles}
        </span>
        <span>
          <span className="k">TURN</span>{status.turn_number}
        </span>
        <span>
          <span className="k">INITIATIVE</span>
          {status.initiative ? (
            <span className={status.initiative === "US" ? "side-us" : "side-prc"}>
              {status.initiative}
            </span>
          ) : "—"}
        </span>
        <SideStat label="VP" status={status} field="vp" />
        <SideStat label="CYBER" status={status} field="cyber" />
        <span>
          <span className="k">MISSION</span>
          <span className="side-us">{status.missions["US"] ?? "—"}</span>
          {" / "}
          <span className="side-prc">{status.missions["PRC"] ?? "—"}</span>
        </span>
        <span>
          <span className="k">POSTURE</span>
          <span className="side-us">{status.postures["US"] ?? "—"}</span>
          {" / "}
          <span className="side-prc">{status.postures["PRC"] ?? "—"}</span>
        </span>
      </div>
    </div>
  );
}
