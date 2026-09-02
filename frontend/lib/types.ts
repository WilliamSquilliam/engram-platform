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
  // One-sentence LLM description written at onboarding; shown as a secondary line when present.
  description?: string | null;
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

// An external-source connector's provider id (as used in the connector routes).
export type ConnectorProvider = "google_drive" | "sharepoint" | string;

// A live connection for this workspace (GET /connectors/connections). One row per
// account the operator has linked, newest usable for the auto-open-after-OAuth flow.
export interface ConnectorConnection {
  id: string;
  provider: ConnectorProvider;
  account_label: string;
  created_at: string;
}

// A folder listed while browsing a connection (GET .../browse). `id` is an opaque
// string — the UI never parses it, it just passes it back to drill in or import.
export interface BrowseFolder {
  id: string;
  name: string;
}
// One level of a connection's folder tree. `path_hint` is a breadcrumb the backend
// writes (e.g. "My Drive / Reports"); `supported_files` counts importable files here.
export interface BrowseResult {
  folders: BrowseFolder[];
  supported_files: number;
  path_hint: string;
}

// Background import lifecycle (GET /corpora/{id}/import-status). "limited" means the
// beta document cap was reached mid-import — everything imported so far is KEPT.
export type ImportState = "none" | "running" | "done" | "failed" | "limited";
export interface ImportStatus {
  state: ImportState;
  imported: number;
  skipped: number;
  failed: number;
  folder_name: string;
  error: string | null;
}

// The onboarding "choose model" step (GET /models). Tiers with available=false render but are
// not selectable — "coming soon".
export interface ModelTier {
  id: string;
  label: string;
  // Public model name shown to users; falls back to `label` when absent.
  display_name?: string;
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
  // False when the GPU onboard plane is down — gates the review step's Start onboarding button.
  serving_up: boolean;
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
  // Display name (set at accept-invite / from Google; older accounts may lack one).
  name?: string | null;
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
// Mirrors backend MemberResp exactly (id/email/name/role/is_active).
export interface Member {
  id: string;
  email: string;
  name?: string | null;
  role: MemberRole;
  is_active?: boolean;
}

// A pending (unaccepted) invite as listed under GET /admin/members. No token/link is
// re-exposed here — the accept link is shown once, at creation (POST /admin/invites -> a
// gated {status, invite_link} where the link is present only when EMAIL_BACKEND=none).
export interface Invite {
  id: string;
  email: string;
  role: MemberRole;
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

// --- E10 Admin Dashboard (tenant-admin) --------------------------------------------------------
// Per-corpus usage row in GET /admin/usage.
export interface UsageByCorpus {
  corpus_id: string;
  name: string;
  queries: number;
  documents: number;
  storage_gb: number;
  gpu_seconds: number;
}
// One point on the queries-over-time series (recharts x=date, y=queries).
export interface UsageSeriesPoint {
  date: string;
  queries: number;
}
// GET /admin/usage — headline usage across the tenant + a time series + per-corpus breakdown.
export interface AdminUsage {
  queries: number;
  documents: number;
  storage_gb: number;
  gpu_seconds: number;
  n_corpora: number;
  by_corpus: UsageByCorpus[];
  series: UsageSeriesPoint[];
}

// The two-meter rate card (pricing.rate_card() on the backend): memory is billed per document
// per month, inference per 1,000 queries, and onboarding is free (per_onboarded_doc_usd is 0).
// The old per_gb_month_usd key is gone — storage is priced per document now, not per GB.
export interface RateCard {
  per_1k_queries_usd: number;
  per_doc_month_usd: number;
  per_onboarded_doc_usd: number; // 0.0 — adding documents is free
  currency: string;
}

// GET /admin/billing — the current plan, its limits, usage-against-limits, and an estimated cost.
// A shell for now (billing management is coming soon), so limits/usage are open-ended maps.
export interface AdminBilling {
  plan: string;
  limits: Record<string, number>;
  // Open-ended so the extra aggregate keys the backend puts here (storage_gb, n_corpora, …) are covered.
  usage: Record<string, number>;
  // The plan's rate card, surfaced so pricing is visible in one place. Loosely typed to tolerate the
  // "currency" string alongside the numeric rates; RateCard names the keys the UI actually renders.
  rate_card: RateCard & Record<string, number | string>;
  estimated_cost_usd: number;
  currency: string;
  period: string;
}

// GET /billing/status (admin-gated) — the dark-launch flag + the live rate card. Nothing billing
// related becomes visible until `enabled` flips true; `portal_available` gates the Stripe portal.
export interface BillingStatus {
  enabled: boolean;
  rate_card: RateCard;
  portal_available: boolean;
}

// --- E11 Platform Admin (Engram staff only) ----------------------------------------------------
// A tenant row in GET /platform-admin/tenants.
export interface Tenant {
  id: string;
  name: string;
  created_at: string;
  n_users: number;
  n_corpora: number;
  plan: string;
  status: string;
}

// Cost-per-tenant row + fleet totals in GET /platform-admin/usage.
export interface PlatformTenantUsage {
  tenant_id: string;
  name: string;
  queries: number;
  storage_gb: number;
  gpu_seconds: number;
  est_cost_usd: number;
}
export interface PlatformUsage {
  tenants: PlatformTenantUsage[];
  totals: {
    queries?: number;
    storage_gb?: number;
    gpu_seconds?: number;
    est_cost_usd?: number;
    [k: string]: number | undefined;
  };
}

// --- GPU serving control (platform admin) ------------------------------------------------------
// The serving box is a single Lambda Cloud instance. There's no pause: stop terminates it (GPU
// billing -> $0/hr), start launches + auto-provisions a fresh one. Data survives across stop/start.
export type GpuState =
  | "offline"
  | "booting"
  | "provisioning"
  | "warming"
  | "serving"
  | "terminating";

export interface GpuInstance {
  id: string;
  name: string;
  type: string;
  region: string;
  ip: string | null;
  price_cents_per_hour: number | null;
}

// GET /platform-admin/gpu/status. enabled=false hides the whole panel.
export interface GpuStatus {
  enabled: boolean;
  state: GpuState;
  instance: GpuInstance | null;
  serve_reachable: boolean;
  engine_ready: boolean;
  onboard_reachable: boolean;
  dns_pointed: boolean | null;
  hourly_usd: number | null;
}

// POST /platform-admin/gpu/start (202) and /stop (202) — the state the transition moves into.
export interface GpuActionResponse {
  instance_id?: string;
  state: GpuState;
}

// A pending waitlist entry in GET /platform-admin/access-requests.
export interface AccessRequest {
  id: string;
  email: string;
  name: string;
  tenant_name: string;
  reason?: string | null;
  status: string;
  created_at: string;
}
// POST /platform-admin/access-requests/{id}/approve returns a one-time invite link to share.
export interface ApproveAccessResponse {
  status: string;
  invite_link?: string | null;
}
