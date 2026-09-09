// "Copy state for AI": copies the server-rendered machine-readable state +
// action encoding (view.llm_export) to the clipboard, for pasting into an
// LLM chat alongside the pme_materials docs. The text is fog-safe (built
// server-side from the same fogged view the browser sees).

import { useState } from "react";

export default function CopyForAI({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Fallback for non-secure contexts / older browsers.
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy");
      } finally {
        document.body.removeChild(ta);
      }
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="panel copy-ai">
      <button className="copy-ai-btn" onClick={copy} title="Copy this state and your legal actions in a machine-readable format">
        {copied ? "Copied" : "Copy state for AI"}
      </button>
      <span className="copy-ai-hint">
        Paste into your AFWIP chat for play-by-play advice.
      </span>
    </div>
  );
}
