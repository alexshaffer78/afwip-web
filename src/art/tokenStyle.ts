// Visual language for token counters, extracted from the physical components:
// square counters; cobalt-blue US / brick-red PRC; white spec-sheet face with a
// gray top-down silhouette when visible; solid side-color back with the
// national roundel and a black "roll to acquire" diamond when fogged.
//
// Keyed by the backend's TokenView.type display string. `slug` names the real
// scanned face at /art/tokens/<slug>.jpg (shown in the hover inspector).

export type Family = "fighter" | "bomber" | "uas" | "aew" | "ship" | "ada" | "transport";

export interface TokenArt {
  slug: string;
  family: Family;
  label: string; // short text that fits on a counter
}

export const TOKEN_ART: Record<string, TokenArt> = {
  "F-22": { slug: "f-22", family: "fighter", label: "F-22" },
  "F-35A": { slug: "f-35a", family: "fighter", label: "F-35A" },
  "F-15C": { slug: "f-15c", family: "fighter", label: "F-15C" },
  "F-15E": { slug: "f-15e", family: "fighter", label: "F-15E" },
  "F-16C": { slug: "f-16c", family: "fighter", label: "F-16C" },
  "B-52": { slug: "b-52", family: "bomber", label: "B-52" },
  "EC-130": { slug: "ec-130", family: "transport", label: "EC-130" },
  "Attack UAS (US)": { slug: "attack-uas-us", family: "uas", label: "ATK UAS" },
  "Recon UAS (US)": { slug: "recon-uas-us", family: "uas", label: "RCN UAS" },
  "AEW (US)": { slug: "aew-us", family: "aew", label: "AEW" },
  "DDG 81": { slug: "ddg-81", family: "ship", label: "DDG 81" },
  "DDG 115": { slug: "ddg-115", family: "ship", label: "DDG 115" },
  "ADA (US)": { slug: "ada-us", family: "ada", label: "ADA" },
  "J-20B": { slug: "j-20b", family: "fighter", label: "J-20B" },
  "JH-7": { slug: "jh-7", family: "fighter", label: "JH-7" },
  "J-10": { slug: "j-10", family: "fighter", label: "J-10" },
  "J-15": { slug: "j-15", family: "fighter", label: "J-15" },
  "J-16": { slug: "j-16", family: "fighter", label: "J-16" },
  "H-6K": { slug: "h-6k", family: "bomber", label: "H-6K" },
  "Attack UAS (PRC)": { slug: "attack-uas-prc", family: "uas", label: "ATK UAS" },
  "Recon UAS (PRC)": { slug: "recon-uas-prc", family: "uas", label: "RCN UAS" },
  "AEW (PRC)": { slug: "aew-prc", family: "aew", label: "AEW" },
  "Dalian #105": { slug: "dalian-105", family: "ship", label: "DALIAN" },
  "Nanning #162": { slug: "nanning-162", family: "ship", label: "NANNING" },
  "Missile Boat Flotilla": { slug: "flotilla", family: "ship", label: "FLOTILLA" },
  "Mid Range ADA (PRC)": { slug: "mid-range-ada-prc", family: "ada", label: "ADA-M" },
  "Long Range ADA (PRC)": { slug: "long-range-ada-prc", family: "ada", label: "ADA-L" },
};

export function tokenArt(type: string | null): TokenArt | null {
  return type ? TOKEN_ART[type] ?? null : null;
}

// Physical component palette.
export const SIDE_FILL: Record<string, string> = { US: "#3b52ae", PRC: "#a4322a" };
export const MAP = {
  ocean: "#3f9cb7",
  oceanAlt: "#3793ad",
  land: "#3f9e4e",
  edge: "#f4f1ea",     // white sidebar zones (bases, standoff strips)
  sky: "#cfe8f5",      // standoff sky boxes
  numeralUS: "#1d3f8f",
  numeralPRC: "#a4322a",
};

// Top-down silhouettes on a 0..40 square (drawn to evoke, not replicate).
export const FAMILY_GLYPH: Record<Family, string> = {
  fighter:
    "M20 4 L22.5 12 L23 17 L34 24 L34 27 L23 24.5 L22.5 30 L26 34 L26 36 L20 34.5 L14 36 L14 34 L17.5 30 L17 24.5 L6 27 L6 24 L17 17 L17.5 12 Z",
  bomber:
    "M20 3 L22 10 L22.5 15 L37 25 L37 28 L22.5 23 L22 31 L27 35 L27 37 L20 35 L13 37 L13 35 L18 31 L17.5 23 L3 28 L3 25 L17.5 15 L18 10 Z",
  uas:
    "M20 6 L21.5 14 L36 17 L36 19.5 L21.5 20 L21 30 L25 33 L25 35 L20 33.5 L15 35 L15 33 L19 30 L18.5 20 L4 19.5 L4 17 L18.5 14 Z",
  aew:
    "M20 5 L21.5 12 L35 20 L35 23 L21.5 20.5 L21 30 L25 34 L25 36 L20 34.5 L15 36 L15 34 L19 30 L18.5 20.5 L5 23 L5 20 L18.5 12 Z M20 12 A6.5 6.5 0 1 0 20.01 12 Z",
  ship:
    "M20 3 L25 12 L25 30 L22 37 L18 37 L15 30 L15 12 Z M17.5 15 h5 v6 h-5 Z",
  ada:
    "M11 34 h18 v3 H11 Z M13 34 L17 14 L18.5 14 L16 34 Z M19 34 L20 10 L21 34 Z M24 34 L21.5 14 L23 14 L27 34 Z",
  transport:
    "M20 4 L21.5 12 L37 20 L37 23.5 L21.5 21 L21 31 L26 35 L26 37 L20 35.5 L14 37 L14 35 L19 31 L18.5 21 L3 23.5 L3 20 L18.5 12 Z",
};
