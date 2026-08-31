// Shared API types — mirror the backend Pydantic schemas (backend/app/schemas.py)
// and the compare/economics router payloads. Keeps the UI off `any`.

export type CorpusStatus = "new" | "training" | "ready" | "failed";
export type JobStatus = "pending" | "running" | "succeeded" | "failed" | "canceled";
// The resumable onboarding wizard's 5 steps + the terminal "ready". Single source of truth
// mirrors backend schemas.ONBOARDING_STEPS.
export type OnboardingStep = "name" | "documents" | "model" | "review" | "onboarding" | "ready";
export const ONBOARDING_STEPS: OnboardingStep[] =
  ["name", "documents", "model", "review", "onboarding", "ready"];

export interface Corpus {
  id: string;
  name: string;
  source_type: string;
  status: CorpusStatus;
  n_documents: number;
  // Resumable onboarding: the step the user left off at + their chosen model tier.
  onboarding_step: OnboardingStep;
  model_tier: string | null;
  mcp_token: string | null;
  n_cartridges: number | null;
  train_seconds: number | null;
  corpus_tokens: number | null;
  created_at: string;
}

// Per-document parse lifecycle (shared contract with the backend): a freshly
// uploaded file starts "pending"/"parsing" and settles on "parsed" or "failed".
export type ParseStatus = "pending" | "parsing" | "parsed" | "failed";

export interface Document {
  id: string;
  filename: string;
  size: number;
  // Per-document onboarding progress surfaced on wizard resume.
  parse_status?: ParseStatus;
  // Human-readable reason a parse failed (only set when parse_status === "failed").
  parse_error?: string;
  onboard_status?: string;
}

// A document-source connector (GET /connectors). `filesystem` is the built-in
// upload; external ones (google_drive, sharepoint) render disabled with a
// "coming soon" hint until their OAuth apps are configured (available=false).
export interface Connector {
  id: string;
  label: string;
  available: boolean;
  description: string;
}
export interface ConnectorsResponse {
  connectors: Connector[];
}

// The onboarding "choose model" step (GET /models). Tiers with available=false render but are
// not selectable — "coming soon".
export interface ModelTier {
  id: string;
  label: string;
  description: string;
  precision: string;
  context_tokens: number;
  available: boolean;
}
export interface ModelTiers {
  default_tier: string;
  tiers: ModelTier[];
}

// The resumable-onboarding snapshot (GET /corpora/{id}/onboarding).
export interface OnboardingState {
  corpus_id: string;
  onboarding_step: OnboardingStep;
  status: CorpusStatus;
  model_tier: string | null;
  model_ref: string | null;
  n_documents: number;
  documents: Document[];
}

// The review step's pre-run sizing summary (GET /corpora/{id}/estimate).
export interface OnboardEstimate {
  n_documents: number;
  total_bytes: number;
  file_types: Record<string, number>;
  model_tier: string | null;
  est_seconds: number;
  est_cost_ondemand: number;
  est_cart_gb: number;
  gpu_hourly_ondemand: number;
  seconds_per_doc: number;
}

export interface Job {
  id: string;
  corpus_id: string;
  kind: string;
  status: JobStatus;
  detail: string;
  progress: number;
  eta_seconds: number | null;
  created_at: string;
  updated_at: string;
}

export interface User {
  id: string;
  email: string;
  tenant_id: string;
  // Tenant role: "admin" gates the Team section + all /admin endpoints. Everyone else is "member".
  role: "admin" | "member" | string;
  // True for Engram staff with cross-tenant/platform access (separate from tenant role).
  platform_admin?: boolean;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

// --- Team management (tenant-admin) -------------------------------------------------------------
// A tenant member as returned by GET /admin/members.
export type MemberRole = "admin" | "member";
export interface Member {
  id: string;
  email: string;
  name: string | null;
  role: MemberRole;
  // "active" once they've accepted; "invited" while an invite is outstanding.
  status?: "active" | "invited" | string;
  created_at?: string;
}

// An outstanding invite returned by POST /admin/invites. `invite_link` is shown to copy
// because outbound email may be gated off in some environments.
export interface Invite {
  id: string;
  email: string;
  role: MemberRole;
  invite_link: string;
  expires_at?: string;
  created_at?: string;
}

export interface SourceRef {
  id: string;
  title: string;
}

export interface ChatResponse {
  answer: string;
  used_docs: string[];
  sources?: SourceRef[];
}

export type StrategyKey = "everyday" | "rag";

export interface CompareStrategy {
  key: StrategyKey;
  label: string;
  answer: string | null;
  latency_ms: number | null;
  prompt_tokens: number | null;
  gen_tokens: number | null;
  feasible: boolean;
  measured: boolean;
  used_docs: string[] | null;
  cost_per_query: number | null;
  cost_per_month: number | null;
  note: string | null;
  // Engram Smart CAG's adaptive routing readout (null for RAG): the tier it picked + confidence.
  cart_tokens?: number | null;
  tier?: "cartridge" | "cart+docs" | null;
  confidence?: number | null;
  raw_tokens?: number | null;
}

export interface CompareSummary {
  cheaper_than_rag_x: number | null;
  faster_than_rag_x: number | null;
}

export interface CompareResult {
  strategies: CompareStrategy[];
  summary: CompareSummary;
  corpus_tokens: number;
  queries_per_month: number;
  k: number;
  // Present on the production vLLM serve path: both latency AND $/query are measured on the deployment.
  measured?: boolean;
  measured_on?: { model: string; instance: string };
}

// Measured aggregate over recent live queries (the /demo page's real numbers).
export interface MeasuredSummary {
  measured: boolean;
  n: number;
  model: string;
  instance: string;
  cart?: { latency_ms: number | null; prompt_tokens: number | null; resident_kv_tokens: number | null; cost_per_query: number | null };
  rag?: { latency_ms: number | null; prompt_tokens: number | null; cost_per_query: number | null };
  savings?: {
    faster_than_rag_x: number | null;
    fewer_prefill_tokens_x: number | null;
    cheaper_than_rag_x: number | null;
  };
}

export interface Economics {
  trained: boolean;
  n_cartridges: number;
  corpus_tokens: number;
  train_seconds: number | null;
  gpu_hourly_ondemand: number;
  gpu_hourly_spot: number;
  train_cost_ondemand: number;
  train_cost_spot: number;
  cost_per_cart_ondemand: number | null;
  queries_per_month: number;
  per_query: { everyday: number; rag: number };
  per_query_measured?: boolean; // true when per_query came from the live serving path, not the model
  breakeven_vs_rag: number | null;
}

export interface CostComparison {
  inputs: { corpus_tokens: number; queries_per_month: number };
  strategies: {
    key: string;
    label: string;
    per_query: number | null;
    per_month: number | null;
    feasible: boolean;
    quality: string;
    note: string;
  }[];
  savings: {
    vs_rag_x: number | null;
    vs_rag_pct: number | null;
  };
  measured: MeasuredSummary; // real per-query numbers from this deployment (measured.measured=false until first query)
}

// A saved fleet scale-test run. `points` is the per-level series the ramp produced
// ({ u, cart:{qps,ttft,lat,...}, rag:{...} }) — stored verbatim, so typed loosely.
export interface ScaleRun {
  id: string;
  corpus_id: string;
  max_concurrency: number;
  n_queries: number;
  points: any[];
  created_at: string;
}

export interface AuthConfig {
  google_enabled: boolean;
}
