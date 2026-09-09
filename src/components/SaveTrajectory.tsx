// "Save trajectory": explicitly write the recorded hotseat decision tape to
// disk (data/trajectories/{game_id}.jsonl). Recording happens automatically as
// you play; saving is a deliberate choice so you only keep the games worth
// keeping. Re-saving overwrites the same file.

import { useState } from "react";
import { api, ApiError } from "../api";

export default function SaveTrajectory({
  gameId,
  saved,
}: {
  gameId: string;
  saved: boolean;
}) {
  const [status, setStatus] = useState<"idle" | "saving" | "done" | "error">(
    "idle",
  );
  const [detail, setDetail] = useState("");

  const save = async () => {
    setStatus("saving");
    try {
      const res = await api.saveTrajectory(gameId);
      // No server disk in the browser build — download the tape as a file.
      const blob = new Blob([JSON.stringify(res.trajectory, null, 2)],
                            { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `afwip-${gameId}.json`;
      a.click();
      URL.revokeObjectURL(url);
      setDetail(`${res.decisions} decisions downloaded`);
      setStatus("done");
      setTimeout(() => setStatus("idle"), 2500);
    } catch (e) {
      setDetail(e instanceof ApiError ? e.message : "save failed");
      setStatus("error");
    }
  };

  const label =
    status === "saving"
      ? "Saving…"
      : status === "done"
        ? "Saved ✓"
        : saved
          ? "Save trajectory (re-save)"
          : "Save trajectory";

  return (
    <div className="panel copy-ai">
      <button
        className="copy-ai-btn"
        onClick={save}
        disabled={status === "saving"}
        title="Download this game's recorded decision tape as a JSON file"
      >
        {label}
      </button>
      <span className="copy-ai-hint">
        {status === "error"
          ? detail
          : status === "done"
            ? detail
            : "Keep this game as training data."}
      </span>
    </div>
  );
}
