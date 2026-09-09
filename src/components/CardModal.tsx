// Full-size card inspector: click any card thumbnail to read the real card.

export default function CardModal({ cardId, title, onClose }: {
  cardId: number;
  title: string;
  onClose: () => void;
}) {
  return (
    <div className="overlay" onClick={onClose}>
      <div className="card-modal" onClick={(e) => e.stopPropagation()}>
        <img src={`${import.meta.env.BASE_URL}art/cards/${cardId}.jpg`} alt={title} />
        <div className="cap">
          {title}
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}

// Face-down card back (SVG, side-colored) for hidden enemy cards.
export function CardBack({ side, count }: { side: string; count?: number }) {
  const fill = side === "US" ? "#3b52ae" : "#a4322a";
  return (
    <div className="card-thumb card-back-thumb" title={`face-down ${side} card`}>
      <svg viewBox="0 0 60 84">
        <rect x={1} y={1} width={58} height={82} rx={6} fill={fill}
          stroke="rgba(0,0,0,0.4)" />
        <circle cx={30} cy={38} r={14} fill="none"
          stroke="rgba(255,255,255,0.55)" strokeWidth={3} />
        <text x={30} y={44} textAnchor="middle" fontSize={16} fontWeight={800}
          fill="rgba(255,255,255,0.8)">?</text>
        {count != null && count > 1 && (
          <text x={30} y={74} textAnchor="middle" fontSize={12} fontWeight={700}
            fill="#fff">×{count}</text>
        )}
      </svg>
    </div>
  );
}
