// Grouped choice buttons for the current decision. Hovering a choice raises a
// board highlight (actor / target / destination) and, when the choice names a
// card, a preview of the real card face; clicking submits its index.
// Legality is entirely backend-authoritative — we only render what was sent.

import { useState } from "react";
import type { ChoiceView, DecisionView, Highlight } from "../types";

const CATEGORY_ORDER = ["pass", "activate", "move", "acquire", "attack", "enabler", "setup", "other"];
const CATEGORY_LABEL: Record<string, string> = {
  pass: "End / Pass",
  activate: "Activate Squadron",
  move: "Move",
  acquire: "Acquire Target",
  attack: "Attack",
  enabler: "Play Enabler",
  setup: "Options",
  other: "Other",
};

interface Props {
  decision: DecisionView;
  busy: boolean;
  onChoose: (index: number) => void;
  onHover: (h: Highlight) => void;
}

function toHighlight(c: ChoiceView): Highlight {
  return { actorUid: c.actor_uid, targetUid: c.target_uid, band: c.band };
}

const NO_HIGHLIGHT: Highlight = { actorUid: null, targetUid: null, band: null };

export default function ChoicePanel({ decision, busy, onChoose, onHover }: Props) {
  // Hovering a choice that names a card previews the real card face —
  // drafting and enabler play become visual card picking.
  const [previewCard, setPreviewCard] = useState<number | null>(null);
  const groups = new Map<string, ChoiceView[]>();
  for (const c of decision.choices) {
    const cat = groups.get(c.category) ?? [];
    cat.push(c);
    groups.set(c.category, cat);
  }
  const ordered = CATEGORY_ORDER.filter((c) => groups.has(c));
  const sideClass = decision.side === "US" ? "side-us" : "side-prc";

  return (
    <div className="panel choices-scroll">
      <div className="prompt">
        {decision.prompt}
        <span className={`who ${sideClass}`}>{decision.side}</span>
      </div>
      {previewCard != null && (
        <div className="choice-card-preview">
          <img src={`${import.meta.env.BASE_URL}art/cards/${previewCard}.jpg`} alt="card preview" />
        </div>
      )}
      {ordered.map((cat) => (
        <div className="choice-group" key={cat}>
          {ordered.length > 1 && <div className="cat">{CATEGORY_LABEL[cat]}</div>}
          <div className="choice-list">
            {groups.get(cat)!.map((c) => (
              <button
                key={c.index}
                className={`choice-btn cat-${cat}`}
                disabled={busy}
                onClick={() => { setPreviewCard(null); onChoose(c.index); }}
                onMouseEnter={() => { onHover(toHighlight(c)); setPreviewCard(c.card_id); }}
                onMouseLeave={() => { onHover(NO_HIGHLIGHT); setPreviewCard(null); }}
                onFocus={() => { onHover(toHighlight(c)); setPreviewCard(c.card_id); }}
                onBlur={() => { onHover(NO_HIGHLIGHT); setPreviewCard(null); }}
              >
                {c.label}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
