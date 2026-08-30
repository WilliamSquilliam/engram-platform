// Shared API types — mirror the backend Pydantic schemas (backend/app/schemas.py)
// and the compare/economics router payloads. Keeps the UI off `any`.

export type CorpusStatus = "new" | "training" | "ready" | "failed";
export type JobStatus = "pending" | "running" | "succeeded" | "failed" | "canceled";

export interface Corpus {
  id: string;
  name: string;
  source_type: string;
  status: CorpusStatus;
  n_documents: number;
  mcp_token: string | null;
  n_cartridges: number | null;
  train_seconds: number | null;
  corpus_tokens: number | null;
  created_at: string;
}

export interface Document {
  id: string;
  filename: string;
  size: number;
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
  role: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface ChatResponse {
  answer: string;
  used_docs: string[];
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
