// Squadron rosters and enabler hands using the real scanned card faces
// (fog already applied server-side — anything hidden arrives only as counts,
// rendered as card backs). Click a card to inspect it full-size.

import { useState } from "react";
import type { HandView, SquadronPanelView } from "../types";
import CardModal, { CardBack } from "./CardModal";

type Inspect = { cardId: number; title: string } | null;

function SquadronPanel({ panel, onInspect }: {
  panel: SquadronPanelView;
  onInspect: (i: Inspect) => void;
}) {
  const sideClass = panel.side === "US" ? "side-us" : "side-prc";
  return (
    <div className="panel">
      <h2><span className={sideClass}>{panel.side}</span> squadrons</h2>
      <div className="card-row">
        {panel.squadrons.map((s) => (
          <div key={s.card_id}
            className={`card-cell ${s.status === "destroyed" ? "destroyed" : ""}`}>
            <button className="card-thumb"
              onClick={() => onInspect({ cardId: s.card_id, title: s.name })}>
              <img src={`${import.meta.env.BASE_URL}art/cards/${s.card_id}.jpg`} alt={s.name} />
            </button>
            <div className="cell-badges">
              {s.status === "destroyed" && <span className="badge bad">DESTROYED</span>}
              {s.status === "active" && <span className="badge good">active</span>}
              {s.damage > 0 && <span className="badge bad">dmg {s.damage}</span>}
              {s.tokens_lost > 0 && <span className="badge">-{s.tokens_lost}</span>}
              {s.at_contingency_location && <span className="badge">CL</span>}
              {s.grounded_tokens > 0 && <span className="badge">{s.grounded_tokens}g</span>}
            </div>
          </div>
        ))}
        {Array.from({ length: panel.face_down }, (_, i) => (
          <CardBack key={`fd${i}`} side={panel.side} />
        ))}
      </div>
      {panel.off_board_naval > 0 && (
        <div className="roster st-ready">{panel.off_board_naval}× naval off-board</div>
      )}
      {panel.squadrons.length === 0 && panel.face_down === 0 && (
        <div className="roster st-out">—</div>
      )}
    </div>
  );
}

function HandPanel({ hand, onInspect }: {
  hand: HandView;
  onInspect: (i: Inspect) => void;
}) {
  const sideClass = hand.side === "US" ? "side-us" : "side-prc";
  return (
    <div className="panel">
      <h2><span className={sideClass}>{hand.side}</span> hand</h2>
      <div className="card-row">
        {hand.cards.map((c) => (
          <div key={c.card_id} className="card-cell">
            <button className="card-thumb"
              onClick={() => onInspect({ cardId: c.card_id, title: c.name })}>
              <img src={`${import.meta.env.BASE_URL}art/cards/${c.card_id}.jpg`} alt={c.name} />
            </button>
            {c.revealed && <div className="cell-badges"><span className="badge">revealed</span></div>}
          </div>
        ))}
        {hand.hidden_count > 0 && <CardBack side={hand.side} count={hand.hidden_count} />}
      </div>
      {hand.cards.length === 0 && hand.hidden_count === 0 && (
        <div className="roster st-out">—</div>
      )}
      {hand.spent.length > 0 && (
        <div className="spent-block">
          <div className="spent-label">Spent ({hand.spent.length})</div>
          <div className="card-row">
            {hand.spent.map((c) => (
              <div key={`sp${c.card_id}`} className="card-cell spent">
                <button className="card-thumb"
                  onClick={() => onInspect({ cardId: c.card_id, title: c.name })}>
                  <img src={`${import.meta.env.BASE_URL}art/cards/${c.card_id}.jpg`} alt={c.name} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function SidePanels({ squadrons, hands }: {
  squadrons: SquadronPanelView[];
  hands: HandView[];
}) {
  const [inspect, setInspect] = useState<Inspect>(null);
  return (
    <>
      <div className="panels-2col">
        {squadrons.map((p) => (
          <SquadronPanel key={p.side} panel={p} onInspect={setInspect} />
        ))}
      </div>
      <div className="panels-2col">
        {hands.map((h) => (
          <HandPanel key={h.side} hand={h} onInspect={setInspect} />
        ))}
      </div>
      {inspect && (
        <CardModal cardId={inspect.cardId} title={inspect.title}
          onClose={() => setInspect(null)} />
      )}
    </>
  );
}
