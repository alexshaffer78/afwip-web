// SVG game board laid out like the physical AFWIP map:
//
//   [PRC edge: Stand Off / Airbase] [band 5..1] [US edge: CL / Airbase / Stand Off]
//
// PRC edge on the LEFT, US on the RIGHT, with each side's off-map zones
// stacked vertically in a single edge column exactly as printed. Middle range
// bands are teal lanes with mirrored blue/red numerals. Hovering a choice
// highlights actor / target / destination and draws the targeting arrow;
// hovering a visible counter shows the real scanned token face.

import { useState } from "react";
import type { BandView, BoardView, Highlight, StatusView, TokenView } from "../types";
import TokenChip, { CHIP } from "./TokenChip";
import { MAP, tokenArt } from "../art/tokenStyle";

const BAND_W = 122;
const EDGE_W = 134;
const GAP = 5;
const ROW_H = CHIP + GAP;
const MIN_H = 520;

// US band number for each middle band (PRC's is 6 - US's, printed mirrored).
const BAND_NUM: Record<string, number> = {
  BAND_A: 1, BAND_B: 2, BAND_C: 3, BAND_D: 4, BAND_E: 5,
};

const ZONE_LABEL: Record<string, string> = {
  US_STANDOFF: "STAND OFF",
  US_AIRBASE: "US AIRBASE",
  US_CONTINGENCY_LOCATION: "CONTINGENCY LOC.",
  PRC_AIRBASE: "PRC AIRBASE",
  PRC_STANDOFF: "STAND OFF",
};

// Vertical stacking of each side's edge zones, top to bottom, as printed.
const PRC_STACK = ["PRC_STANDOFF", "PRC_AIRBASE"];
const US_STACK = ["US_CONTINGENCY_LOCATION", "US_AIRBASE", "US_STANDOFF"];

// Vertical space inside a zone taken by its label + icon, before tokens.
function zoneHeaderSpace(band: string): number {
  if (band.endsWith("AIRBASE")) return 138;   // roundel + damage boxes + tracks
  if (band === "US_CONTINGENCY_LOCATION") return 72;    // CL disc
  return 26;                                            // standoff: label only
}

function zoneFill(band: string): string {
  return band.endsWith("STANDOFF") ? MAP.sky : MAP.edge;
}

function AirbaseIcon({ cx, cy, side, damage }: {
  cx: number; cy: number; side: "US" | "PRC"; damage: number;
}) {
  const fill = side === "US" ? "#24408f" : "#a4322a";
  return (
    <g>
      <circle cx={cx} cy={cy} r={17} fill={fill} />
      <rect x={cx - 16} y={cy - 3} width={32} height={6} rx={1} fill="#b9bcc4"
        transform={`rotate(18 ${cx} ${cy})`} />
      <rect x={cx - 14} y={cy - 2.5} width={28} height={5} rx={1} fill="#cdd0d6"
        transform={`rotate(-55 ${cx} ${cy})`} />
      {/* VP damage boxes: hollow until hit, filled + marked when damaged */}
      {[0, 1, 2].map((i) => {
        const hit = i < damage;
        const bx = cx - 15 + i * 11;
        return (
          <g key={i}>
            <rect x={bx} y={cy + 21} width={8} height={8}
              fill={hit ? "#e5484d" : "#fdf3f3"}
              stroke="#7f1d1d" strokeWidth={1} />
            {hit && (
              <>
                <line x1={bx + 1.5} y1={cy + 22.5} x2={bx + 6.5} y2={cy + 27.5}
                  stroke="#20242e" strokeWidth={1.4} strokeLinecap="round" />
                <line x1={bx + 6.5} y1={cy + 22.5} x2={bx + 1.5} y2={cy + 27.5}
                  stroke="#20242e" strokeWidth={1.4} strokeLinecap="round" />
              </>
            )}
          </g>
        );
      })}
      <title>{`${side} airbase — ${damage}/3 VP damage boxes hit this ATO`}</title>
    </g>
  );
}

// Cyber and intel tracks as printed in the edge sidebars: gray numbered boxes
// with the current cyber rate marked, and a NORMAL / ADV pair for intel.
function Tracks({ cx, y, cyber, intel }: {
  cx: number; y: number; cyber: number; intel: string;
}) {
  const BOX = 15;
  const x0 = cx - (5 * (BOX + 2) - 2) / 2;
  const adv = intel === "ADVANTAGE";
  return (
    <g fontFamily="var(--font-ui)">
      <text x={cx} y={y} textAnchor="middle" fontSize={7.5} fontWeight={700}
        letterSpacing="0.08em" fill="#5a5f6e">CYBER TRACK</text>
      {[0, 1, 2, 3, 4].map((n) => {
        const bx = x0 + n * (BOX + 2);
        const cur = n === cyber;
        return (
          <g key={n}>
            <rect x={bx} y={y + 3} width={BOX} height={BOX} rx={2}
              fill={cur ? "#ffd166" : "#c9ccd2"}
              stroke={cur ? "#8a6a12" : "#7d818a"} strokeWidth={cur ? 1.6 : 1} />
            <text x={bx + BOX / 2} y={y + 3 + BOX / 2 + 3.5} textAnchor="middle"
              fontSize={9} fontWeight={800}
              fill={cur ? "#20242e" : "#4d5058"}>{n}</text>
          </g>
        );
      })}
      <text x={cx} y={y + 29} textAnchor="middle" fontSize={7.5} fontWeight={700}
        letterSpacing="0.08em" fill="#5a5f6e">INTEL TRACK</text>
      {[["NORMAL", !adv], ["ADV", adv]].map(([label, cur], i) => {
        const w = 38;
        const bx = cx - w - 2 + i * (w + 4);
        return (
          <g key={String(label)}>
            <rect x={bx} y={y + 32} width={w} height={13} rx={2}
              fill={cur ? "#ffd166" : "#c9ccd2"}
              stroke={cur ? "#8a6a12" : "#7d818a"} strokeWidth={cur ? 1.6 : 1} />
            <text x={bx + w / 2} y={y + 41.5} textAnchor="middle" fontSize={8}
              fontWeight={800} fill={cur ? "#20242e" : "#4d5058"}>
              {String(label)}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function CLIcon({ cx, cy }: { cx: number; cy: number }) {
  return (
    <g>
      <circle cx={cx} cy={cy} r={15} fill="#24408f" />
      <rect x={cx - 14} y={cy - 3} width={28} height={6} rx={1} fill="#9aa0ab"
        transform={`rotate(-30 ${cx} ${cy})`} />
    </g>
  );
}

interface Pop {
  slug: string;
  label: string;
  x: number;
  y: number;
}

interface ZoneRect {
  band: BandView;
  x: number;
  y: number;
  w: number;
  h: number;
  headerSpace: number;
}

export default function Board({ board, highlight, status, turnSide }: {
  board: BoardView;
  highlight: Highlight;
  status: StatusView;
  turnSide: string;   // side currently being asked to act (decision side, else active_side)
}) {
  const [pop, setPop] = useState<Pop | null>(null);

  const byId = new Map(board.bands.map((b) => [b.band, b]));
  // Middle bands rendered 5..1 left-to-right (PRC side first, as printed).
  const middle = ["BAND_E", "BAND_D", "BAND_C", "BAND_B", "BAND_A"]
    .map((id) => byId.get(id)!)
    .filter(Boolean);

  const rows = (b: BandView) => Math.ceil(b.tokens.length / 2);

  // Height: whichever needs more — the middle lanes or either edge stack.
  const middleNeed = 74 + Math.max(3, ...middle.map(rows)) * ROW_H + 44;
  const stackNeed = (stack: string[]) =>
    stack.reduce((acc, id) => {
      const b = byId.get(id);
      return acc + zoneHeaderSpace(id) + (b ? rows(b) : 0) * ROW_H + 14;
    }, 0) + 8;
  const height = Math.max(MIN_H, middleNeed, stackNeed(PRC_STACK), stackNeed(US_STACK));
  const width = 2 * EDGE_W + middle.length * BAND_W;

  // Edge stacks: give every zone its minimum, split leftover evenly.
  function layoutStack(stack: string[], x: number): ZoneRect[] {
    const mins = stack.map((id) => {
      const b = byId.get(id);
      return zoneHeaderSpace(id) + (b ? rows(b) : 0) * ROW_H + 14;
    });
    const extra = (height - 8 - mins.reduce((a, m) => a + m, 0)) / stack.length;
    const out: ZoneRect[] = [];
    let y = 4;
    stack.forEach((id, i) => {
      const h = mins[i] + Math.max(0, extra);
      out.push({ band: byId.get(id)!, x, y, w: EDGE_W - 4, h: h - 4,
                 headerSpace: zoneHeaderSpace(id) });
      y += h;
    });
    return out;
  }

  const zones: ZoneRect[] = [
    ...layoutStack(PRC_STACK, 2),
    ...layoutStack(US_STACK, EDGE_W + middle.length * BAND_W + 2),
  ];

  // Token counter positions (top-left) and centers, by uid.
  const positions = new Map<number, { x: number; y: number }>();
  const centers = new Map<number, { x: number; y: number }>();
  const place = (t: TokenView, x: number, y: number) => {
    positions.set(t.uid, { x, y });
    centers.set(t.uid, { x: x + CHIP / 2, y: y + CHIP / 2 });
  };
  middle.forEach((band, col) => {
    const x0 = EDGE_W + col * BAND_W + (BAND_W - 2 * CHIP - GAP) / 2;
    band.tokens.forEach((t, i) =>
      place(t, x0 + (i % 2) * (CHIP + GAP), 74 + Math.floor(i / 2) * ROW_H));
  });
  zones.forEach((z) => {
    const x0 = z.x + (z.w - 2 * CHIP - GAP) / 2;
    z.band.tokens.forEach((t, i) =>
      place(t, x0 + (i % 2) * (CHIP + GAP), z.y + z.headerSpace + Math.floor(i / 2) * ROW_H));
  });

  // Highlight targets: band id -> its lane/zone center.
  const bandCenter = (id: string): { x: number; y: number } | undefined => {
    const mid = middle.findIndex((b) => b.band === id);
    if (mid >= 0) return { x: EDGE_W + mid * BAND_W + BAND_W / 2, y: 46 };
    const z = zones.find((zz) => zz.band.band === id);
    return z ? { x: z.x + z.w / 2, y: z.y + 16 } : undefined;
  };

  const from = highlight.actorUid != null ? centers.get(highlight.actorUid) : undefined;
  const to = highlight.targetUid != null
    ? centers.get(highlight.targetUid)
    : highlight.band != null ? bandCenter(highlight.band) : undefined;

  const onHover = (t: TokenView, e: React.MouseEvent) => {
    if (t.fogged || !t.type) return setPop(null);
    const art = tokenArt(t.type);
    if (!art) return;
    const wrap = (e.currentTarget as SVGElement).closest(".board-wrap") as HTMLElement;
    const r = wrap.getBoundingClientRect();
    setPop({
      slug: art.slug,
      label: `${t.type}#${t.uid}`,
      x: Math.min(e.clientX - r.left + 16, r.width - 240),
      y: Math.min(e.clientY - r.top + 14, r.height - 240),
    });
  };

  const chip = (t: TokenView) => {
    const p = positions.get(t.uid)!;
    return (
      <TokenChip key={t.uid} t={t} x={p.x} y={p.y} highlight={highlight}
        onHover={onHover} onLeave={() => setPop(null)} />
    );
  };

  // Whose turn it is: frames the board and highlights that side's airbase.
  const activeSide = turnSide === "PRC" ? "PRC" : "US";
  const activeAirbase = activeSide === "US" ? "US_AIRBASE" : "PRC_AIRBASE";
  const activeColor = activeSide === "US" ? "#4cc3ff" : "#ff5f56";

  return (
    <div className={`board-wrap turn-${activeSide.toLowerCase()}`}>
      <svg
        className="board-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="AFWIP game board"
      >
        <defs>
          <marker id="arrowhead" markerWidth="8" markerHeight="7" refX="7" refY="3.5" orient="auto">
            <polygon points="0 0, 8 3.5, 0 7" fill="#ffd166" />
          </marker>
          {/* Colored glow around the active side's airbase (whose turn it is). */}
          <filter id="airbase-glow-us" x="-40%" y="-40%" width="180%" height="180%">
            <feDropShadow dx="0" dy="0" stdDeviation="6" floodColor="#4cc3ff" floodOpacity="0.9" />
          </filter>
          <filter id="airbase-glow-prc" x="-40%" y="-40%" width="180%" height="180%">
            <feDropShadow dx="0" dy="0" stdDeviation="6" floodColor="#ff5f56" floodOpacity="0.9" />
          </filter>
        </defs>

        <rect x={0} y={0} width={width} height={height} rx={10} fill="#eef0ee" />

        {/* middle range bands */}
        {middle.map((band, col) => {
          const x = EDGE_W + col * BAND_W;
          const isDest = highlight.band === band.band;
          const num = BAND_NUM[band.band];
          return (
            <g key={band.band}>
              <rect x={x + 2} y={2} width={BAND_W - 4} height={height - 4} rx={6}
                fill={num % 2 === 0 ? MAP.oceanAlt : MAP.ocean}
                stroke={isDest ? "#ffd166" : "rgba(0,0,0,0.18)"}
                strokeWidth={isDest ? 3 : 1} />
              <text x={x + BAND_W / 2} y={38} textAnchor="middle" fontSize={30}
                fontWeight={800} fill={MAP.numeralUS} opacity={0.85}
                fontFamily="var(--font-ui)">{num}</text>
              <text x={x + BAND_W / 2} y={height - 14} textAnchor="middle" fontSize={30}
                fontWeight={800} fill={MAP.numeralPRC} opacity={0.85}
                fontFamily="var(--font-ui)"
                transform={`rotate(180 ${x + BAND_W / 2} ${height - 24})`}>{6 - num}</text>

              {band.band === "BAND_C" && (
                <path d={`M${x + 24} ${height * 0.6}
                          q16 -20 30 -7 q18 13 9 29 q-11 18 -27 9 q-18 -9 -12 -31 Z`}
                  fill={MAP.land} opacity={0.9} />
              )}
              {band.band === "BAND_D" && (
                <g fill={MAP.land} opacity={0.9}>
                  <ellipse cx={x + 44} cy={height * 0.54} rx={10} ry={5.5} />
                  <ellipse cx={x + 72} cy={height * 0.64} rx={7} ry={4} />
                  <ellipse cx={x + 56} cy={height * 0.74} rx={4.5} ry={3} />
                </g>
              )}

              {band.tokens.map(chip)}
            </g>
          );
        })}

        {/* edge zones, stacked as printed */}
        {zones.map((z) => {
          const id = z.band.band;
          const isDest = highlight.band === id;
          const isActiveAirbase = id === activeAirbase;
          const cx = z.x + z.w / 2;
          return (
            <g key={id}>
              <rect x={z.x} y={z.y} width={z.w} height={z.h} rx={6}
                fill={zoneFill(id)}
                stroke={isDest ? "#ffd166" : isActiveAirbase ? activeColor : "rgba(0,0,0,0.25)"}
                strokeWidth={isDest ? 3 : isActiveAirbase ? 3.5 : 1}
                filter={isActiveAirbase && !isDest
                  ? `url(#airbase-glow-${activeSide.toLowerCase()})` : undefined} />
              <text x={cx} y={z.y + 15} textAnchor="middle" fontSize={9.5}
                fontWeight={700} letterSpacing="0.1em" fill="#5a5f6e"
                fontFamily="var(--font-ui)">{ZONE_LABEL[id]}</text>
              {id.endsWith("AIRBASE") && (() => {
                const s = id.startsWith("US") ? "US" : "PRC";
                return (
                  <>
                    <AirbaseIcon cx={cx} cy={z.y + 44} side={s}
                      damage={status.base_damage[s] ?? 0} />
                    <Tracks cx={cx} y={z.y + 86}
                      cyber={status.cyber[s] ?? 0}
                      intel={status.intel[s] ?? "NORMAL"} />
                  </>
                );
              })()}
              {id === "US_CONTINGENCY_LOCATION" && <CLIcon cx={cx} cy={z.y + 42} />}
              {z.band.tokens.map(chip)}
            </g>
          );
        })}

        {/* faint center watermark ring, like the printed logo */}
        <g opacity={0.1} pointerEvents="none">
          <circle cx={width / 2} cy={height / 2} r={Math.min(84, height / 3)}
            fill="none" stroke="#fff" strokeWidth={15} />
        </g>

        {from && to && (
          <line x1={from.x} y1={from.y} x2={to.x} y2={to.y}
            stroke="#ffd166" strokeWidth={3} strokeDasharray="7 5"
            markerEnd="url(#arrowhead)" opacity={0.95} />
        )}
      </svg>

      {pop && (
        <div className="token-pop" style={{ left: pop.x, top: pop.y }}>
          <img src={`${import.meta.env.BASE_URL}art/tokens/${pop.slug}.jpg`} alt={pop.label} />
          <div className="cap">{pop.label}</div>
        </div>
      )}
    </div>
  );
}
