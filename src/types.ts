// TypeScript mirror of afwip/view/models.py — the backend JSON contract.

export interface TokenView {
  uid: number;
  side: string;
  band: string;
  fogged: boolean;
  av: number;
  acquired: boolean;
  label: string;
  type: string | null;
  winchester: boolean | null;
  grounded: boolean | null;
}

export interface BandView {
  band: string;
  header: string;
  tokens: TokenView[];
}

export interface BoardView {
  bands: BandView[];
}

export interface SquadronView {
  card_id: number;
  name: string;
  token_type: string;
  status: string;
  damage: number;
  tokens_lost: number;
  at_contingency_location: boolean;
  grounded_tokens: number;
}

export interface SquadronPanelView {
  side: string;
  squadrons: SquadronView[];
  face_down: number;
  off_board_naval: number;
}

export interface HandCardView {
  card_id: number;
  name: string;
  revealed: boolean;
}

export interface HandView {
  side: string;
  cards: HandCardView[];
  hidden_count: number;
  spent_count: number;
  spent: HandCardView[]; // enablers this side has played (public, both sides)
}

export interface StatusView {
  campaign: number;
  ato_cycle: number;
  total_ato_cycles: number;
  turn_number: number;
  phase: string;
  active_side: string;
  initiative: string | null;
  vp: Record<string, number>;
  cyber: Record<string, number>;
  missions: Record<string, string | null>;
  postures: Record<string, string | null>;
  base_damage: Record<string, number>;
  intel: Record<string, string>;
}

export interface CaptureEntryView {
  kind: string; // "token" | "squadron"
  label: string;
  vp: number;
  victim_side: string;
  on_ground: boolean;
  token_type: string | null;
  uid: number | null;
  card_id: number | null;
  ato_cycle: number;
}

export interface CapturesPanelView {
  side: string;
  entries: CaptureEntryView[];
}

export interface ChoiceView {
  index: number;
  label: string;
  kind: string;
  category: string;
  card_id: number | null;
  band: string | null;
  actor_uid: number | null;
  target_uid: number | null;
  branch_idx: number | null;
  alloc_kind: string | null;
}

export interface DecisionView {
  node_type: string;
  side: string;
  prompt: string;
  choices: ChoiceView[];
}

export interface EventView {
  step: number;
  type: string;
  side: string | null;
  message: string;
  public: boolean;
  redacted: string | null;
  data: Record<string, unknown>;
}

export interface DieRoll {
  value: number;    // final result (natural + bonus)
  natural: number;  // the chosen die before any bonus
  bonus: number;    // value - natural
  mode: "NORMAL" | "ADVANTAGE" | "DISADVANTAGE";
  dice: number[];   // raw dice rolled (2 when at advantage/disadvantage)
  role?: string;    // what this roll was for: "hit" | "damage" | "acquire"
}

export interface CaptureLineView {
  label: string;
  vp: number;
  on_ground: boolean;
}

export interface AccruedLineView {
  reason: string;
  points: number;
}

export interface SideScoreView {
  side: string;
  mission: string;
  ato_total: number;
  campaign_total: number;
  token_captures: CaptureLineView[];
  squadron_captures: CaptureLineView[];
  accrued: AccruedLineView[];
}

export interface ScoreReportView {
  ato: number;
  sides: SideScoreView[];
}

export interface AgentMoveView {
  decision_no: number;
  side: string;
  label: string;
  reasoning: string;
}

export interface GameView {
  game_id: string;
  revision: number;
  node_id: string | null;
  viewer: string | null;
  mode: string;
  terminal: boolean;
  winner: string | null;
  handoff_required: boolean;
  roll_pending: boolean;
  recording: boolean;        // a trajectory is being captured (savable on demand)
  trajectory_saved: boolean; // the tape has been written to disk this session
  agent_log: AgentMoveView[]; // recent agent moves + reasoning (Watch Agents only)
  status: StatusView;
  board: BoardView;
  squadrons: SquadronPanelView[];
  hands: HandView[];
  captures: CapturesPanelView[];
  decision: DecisionView | null;
  event_log: EventView[];
  score_report: ScoreReportView | null;
  llm_export: string | null; // machine-readable state+actions for "Copy for AI"
}

export interface ConfigView {
  campaigns: number[];
  modes: string[];
  sides: string[];
  agents: string[];
}

// Highlight state driven by hovering a choice button.
export interface Highlight {
  actorUid: number | null;
  targetUid: number | null;
  band: string | null;
}
