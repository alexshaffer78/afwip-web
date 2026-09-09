// One square counter, drawn like the physical piece.
//
// Visible: white spec-sheet face, side-color border, gray top-down silhouette,
// type label, status marks (W = winchester, dimmed = grounded).
// Fogged: the physical "not acquired" back — solid side color, national
// roundel, black diamond with the acquisition value. Rendered from side+AV
// only, so no DOM inspection can reveal the hidden type.

import type { Highlight, TokenView } from "../types";
import { FAMILY_GLYPH, SIDE_FILL, tokenArt } from "../art/tokenStyle";

export const CHIP = 50; // counter edge length in board units

function USRoundel({ cx, cy, r }: { cx: number; cy: number; r: number }) {
  const star = (n: number) => {
    // 5-point star path around (cx, cy) with outer radius n.
    const pts: string[] = [];
    for (let i = 0; i < 10; i++) {
      const rad = i % 2 === 0 ? n : n * 0.42;
      const a = -Math.PI / 2 + (i * Math.PI) / 5;
      pts.push(`${cx + rad * Math.cos(a)},${cy + rad * Math.sin(a)}`);
    }
    return `M${pts.join("L")}Z`;
  };
  return (
    <g>
      <rect x={cx - r * 2.1} y={cy - r * 0.42} width={r * 4.2} height={r * 0.84}
        fill="#fff" stroke="#111" strokeWidth={0.8} />
      <rect x={cx - r * 2.1} y={cy - r * 0.1} width={r * 4.2} height={r * 0.2} fill="#b03a2e" />
      <circle cx={cx} cy={cy} r={r} fill="#111" />
      <path d={star(r * 0.78)} fill="#fff" />
    </g>
  );
}

function PRCRoundel({ cx, cy, r }: { cx: number; cy: number; r: number }) {
  const star = (n: number) => {
    const pts: string[] = [];
    for (let i = 0; i < 10; i++) {
      const rad = i % 2 === 0 ? n : n * 0.42;
      const a = -Math.PI / 2 + (i * Math.PI) / 5;
      pts.push(`${cx + rad * Math.cos(a)},${cy + rad * Math.sin(a)}`);
    }
    return `M${pts.join("L")}Z`;
  };
  return (
    <g>
      <rect x={cx - r * 2.1} y={cy - r * 0.32} width={r * 4.2} height={r * 0.64}
        fill="#c8281e" stroke="#f2c94c" strokeWidth={1} />
      <path d={star(r)} fill="#c8281e" stroke="#f2c94c" strokeWidth={1.2} />
    </g>
  );
}

interface Props {
  t: TokenView;
  x: number; // top-left in board units
  y: number;
  highlight: Highlight;
  onHover: (t: TokenView, e: React.MouseEvent) => void;
  onLeave: () => void;
}

export default function TokenChip({ t, x, y, highlight, onHover, onLeave }: Props) {
  const isActor = highlight.actorUid === t.uid;
  const isTarget = highlight.targetUid === t.uid;
  const side = SIDE_FILL[t.side] ?? "#666";
  const art = tokenArt(t.type);
  const cx = x + CHIP / 2;

  return (
    <g
      opacity={t.grounded ? 0.55 : 1}
      onMouseMove={(e) => onHover(t, e)}
      onMouseLeave={onLeave}
      style={{ cursor: t.fogged ? "default" : "pointer" }}
    >
      {(isActor || isTarget) && (
        <rect x={x - 3.5} y={y - 3.5} width={CHIP + 7} height={CHIP + 7} rx={7}
          fill="none" strokeWidth={3} stroke={isActor ? "#ffd166" : "#e5484d"}>
          <animate attributeName="opacity" values="1;0.35;1" dur="1.2s" repeatCount="indefinite" />
        </rect>
      )}

      {t.fogged ? (
        // Physical back: side color + roundel + AV diamond. side/AV only.
        <g>
          <rect x={x} y={y} width={CHIP} height={CHIP} rx={4} fill={side}
            stroke="#20242e" strokeWidth={1} />
          {t.side === "US"
            ? <USRoundel cx={cx} cy={y + 20} r={9} />
            : <PRCRoundel cx={cx} cy={y + 20} r={9} />}
          <g transform={`translate(${cx}, ${y + 39}) rotate(45)`}>
            <rect x={-6} y={-6} width={12} height={12} fill="#111" />
          </g>
          <text x={cx} y={y + 42.5} textAnchor="middle" fontSize={9}
            fontWeight={700} fill="#fff" fontFamily="var(--font-ui)">
            {t.av}
          </text>
          <text x={x + CHIP - 4} y={y + 10} textAnchor="end" fontSize={7}
        fill="rgba(255,255,255,0.85)" fontFamily="var(--font-mono)">
            #{t.uid}
          </text>
        </g>
      ) : (
        <g>
          <rect x={x} y={y} width={CHIP} height={CHIP} rx={4} fill="#f8f7f2"
            stroke={side} strokeWidth={3} />
          {art && (
            <path d={FAMILY_GLYPH[art.family]} fill="#8a8794" stroke="#4a4752"
              strokeWidth={0.8}
              transform={`translate(${x + 8}, ${y + 5}) scale(0.85)`} />
          )}
          <text x={cx} y={y + CHIP - 4.5} textAnchor="middle" fontSize={8.5}
            fontWeight={700} fill="#1c1c24" fontFamily="var(--font-ui)">
            {art?.label ?? t.type}
          </text>
          <text x={x + 4} y={y + 11} fontSize={7} fill="#6b6875"
            fontFamily="var(--font-mono)">
            #{t.uid}
          </text>
          {t.acquired && (
            <circle cx={x + CHIP - 8} cy={y + 8} r={3.6} fill="#e5484d"
              stroke="#fff" strokeWidth={1} />
          )}
          {t.winchester && (
            <text x={x + CHIP - 4} y={y + 22} textAnchor="end" fontSize={9}
              fontWeight={800} fill="#b03a2e" fontFamily="var(--font-ui)">
              W
            </text>
          )}
        </g>
      )}
      <title>
        {t.fogged
          ? `Unidentified ${t.side} token #${t.uid} — roll ${t.av} to acquire`
          : `${t.type}#${t.uid} (${t.side})${t.acquired ? " — acquired" : ""}` +
            `${t.winchester ? " — Winchester" : ""}${t.grounded ? " — grounded" : ""}`}
      </title>
    </g>
  );
}
