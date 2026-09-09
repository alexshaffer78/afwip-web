// Each side's score pile: enemy tokens and Squadron Cards it has shot down,
// shown as surrendered pieces (mini counters / card thumbs with a red X) with
// their current mission VP value. Public information, like the physical piles.

import type { CaptureEntryView, CapturesPanelView } from "../types";
import { FAMILY_GLYPH, SIDE_FILL, tokenArt } from "../art/tokenStyle";

function TokenTrophy({ e }: { e: CaptureEntryView }) {
  const art = tokenArt(e.token_type);
  const side = SIDE_FILL[e.victim_side] ?? "#666";
  return (
    <div className="trophy" title={`${e.label}${e.on_ground ? " (destroyed on ground)" : ""} — +${e.vp} VP (ATO ${e.ato_cycle})`}>
      <svg viewBox="0 0 40 40">
        <rect x={1} y={1} width={38} height={38} rx={4} fill="#f8f7f2"
          stroke={side} strokeWidth={2.5} />
        {art && (
          <path d={FAMILY_GLYPH[art.family]} fill="#8a8794" stroke="#4a4752"
            strokeWidth={0.8} transform="translate(4.5, 2) scale(0.78)" />
        )}
        <line x1={6} y1={6} x2={34} y2={34} stroke="#c8281e" strokeWidth={3.5}
          strokeLinecap="round" />
        <line x1={34} y1={6} x2={6} y2={34} stroke="#c8281e" strokeWidth={3.5}
          strokeLinecap="round" />
      </svg>
      <span className="vp">+{e.vp}</span>
    </div>
  );
}

function SquadronTrophy({ e }: { e: CaptureEntryView }) {
  return (
    <div className="trophy squadron" title={`${e.label} (Squadron Card) — +${e.vp} VP (ATO ${e.ato_cycle})`}>
      {e.card_id != null ? (
        <div className="mini-card">
          <img src={`${import.meta.env.BASE_URL}art/cards/${e.card_id}.jpg`} alt={e.label} />
          <svg viewBox="0 0 40 56" className="x-overlay">
            <line x1={6} y1={8} x2={34} y2={48} stroke="#c8281e" strokeWidth={4}
              strokeLinecap="round" />
            <line x1={34} y1={8} x2={6} y2={48} stroke="#c8281e" strokeWidth={4}
              strokeLinecap="round" />
          </svg>
        </div>
      ) : (
        <span>{e.label}</span>
      )}
      <span className="vp">+{e.vp}</span>
    </div>
  );
}

export default function ScorePiles({ captures }: { captures: CapturesPanelView[] }) {
  return (
    <div className="panels-2col">
      {captures.map((pile) => {
        const sideClass = pile.side === "US" ? "side-us" : "side-prc";
        const total = pile.entries.reduce((a, e) => a + e.vp, 0);
        return (
          <div className="panel" key={pile.side}>
            <h2>
              <span className={sideClass}>{pile.side}</span> shot down
              {pile.entries.length > 0 && (
                <span className="badge" style={{ marginLeft: 8 }}>+{total} VP</span>
              )}
            </h2>
            <div className="trophy-row">
              {pile.entries.map((e, i) =>
                e.kind === "squadron"
                  ? <SquadronTrophy key={i} e={e} />
                  : <TokenTrophy key={i} e={e} />)}
              {pile.entries.length === 0 && (
                <span className="roster st-out">no kills yet</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
