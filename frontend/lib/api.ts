// Tiny fetch wrapper around the control-plane API. Token in localStorage.
import type {
  AccessRequest,
  AdminBilling,
  AdminUsage,
  ApproveAccessResponse,
  AuthConfig,
  ChatResponse,
  CompareResult,
  ConnectorsResponse,
  Corpus,
  Document,
  Invite,
  Job,
  Member,
  MemberRole,
  ModelTiers,
  OnboardEstimate,
  OnboardingState,
  OnboardingStep,
  PlatformUsage,
  ScaleRun,
  Tenant,
  TokenResponse,
  User,
} from "./types";

// Where the browser sends API calls. Local dev: backend on :8000. On AWS the
// frontend and control-plane sit behind one internal ALB (path-routed), so set
// NEXT_PUBLIC_API_URL="same-origin" at build time and we target window.location
// .origin — the same host the app was loaded from (e.g. the SSM-tunnel localhost).
const RAW_API_URL = process.env.NEXT_PUBLIC_API_URL;
export const API_URL =
  RAW_API_URL === "same-origin"
    ? (typeof window !== "undefined" ? window.location.origin : "")
    : RAW_API_URL || "http://localhost:8000";

// "Remember me on this device" is real on both ends: remembered sessions go to
// localStorage (survive a browser restart) and the backend mints a 30-day JWT;
// non-remembered go to sessionStorage (gone when the browser closes) with the
// default short JWT. getToken prefers the session copy (it's the fresher login).
export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return sessionStorage.getItem("token") || localStorage.getItem("token");
}
export function setToken(t: string, remember: boolean) {
  const dst = remember ? localStorage : sessionStorage;
  const other = remember ? sessionStorage : localStorage;
  dst.setItem("token", t);
  other.removeItem("token"); // never leave a stale token in the other store
}
export function clearToken() {
  localStorage.removeItem("token");
  sessionStorage.removeItem("token");
}

// Invite / reset links carry their one-time token in the URL FRAGMENT (#token=..., what the
// backend emits) so the token never reaches server logs or Referer headers; ?token= is a
// fallback for hand-edited links. Fragment is client-only, so callers read it in an effect.
export function readUrlToken(): string | null {
  if (typeof window === "undefined") return null;
  const fromHash = new URLSearchParams(window.location.hash.slice(1)).get("token");
  return fromHash || new URLSearchParams(window.location.search).get("token");
}

// Cross-component sync: the sidebar and dashboard both render the corpus list
// with independent state. Any mutation (create/delete/train) dispatches this so
// every listener refetches — no global store needed.
export const CORPORA_CHANGED = "corpora:changed";
export function notifyCorporaChanged() {
  if (typeof window !== "undefined") window.dispatchEvent(new Event(CORPORA_CHANGED));
}

const JSON_HEADERS = { "Content-Type": "application/json" };

// A 401 anywhere in the app means the session token is gone or expired. Clear it and bounce to
// /login rather than letting a stale token spew errors on every panel. login() uses its own fetch
// (below), so a wrong-password 401 there never reaches this — it stays an inline error.
function handleUnauthorized() {
  clearToken();
  if (typeof window !== "undefined") window.location.href = "/login";
}

async function req<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = { ...(opts.headers as Record<string, string>) };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${API_URL}${path}`, { ...opts, headers });
  if (res.status === 401) {
    handleUnauthorized();
    throw new Error("Session expired");
  }
  if (!res.ok) {
    let msg: unknown = res.statusText;
    try {
      const j = await res.json();
      msg = j.detail || msg;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return (res.status === 204 ? null : await res.json()) as T;
}

export const api = {
  register: (email: string, password: string, tenant_name: string) =>
    req<TokenResponse>("/auth/register", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ email, password, tenant_name }),
    }),
  login: async (email: string, password: string, rememberMe = false): Promise<TokenResponse> => {
    const form = new URLSearchParams();
    form.set("username", email);
    form.set("password", password);
    form.set("remember_me", String(rememberMe)); // backend mints a 30-day JWT when true
    const res = await fetch(`${API_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form,
    });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      throw new Error(j.detail || "Login failed");
    }
    return res.json();
  },
  me: () => req<User>("/auth/me"),

  // --- Invite-only self-serve auth ---------------------------------------------------------------
  // Waitlist a prospective tenant. Always returns success-shaped (backend queues for manual review).
  requestAccess: (payload: { email: string; name: string; tenant_name: string; reason?: string }) =>
    req<{ status: string }>("/auth/request-access", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify(payload),
    }),
  // Redeem an invite token by setting a password; returns a session token so we land in the app.
  acceptInvite: (token: string, password: string, name?: string) =>
    req<TokenResponse>("/auth/accept-invite", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ token, password, ...(name ? { name } : {}) }),
    }),
  // Always returns a generic success message (never reveals whether the email exists).
  forgotPassword: (email: string) =>
    req<{ status: string }>("/auth/forgot-password", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ email }),
    }),
  resetPassword: (token: string, password: string) =>
    req<{ status: string }>("/auth/reset-password", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ token, password }),
    }),

  // --- Team management (tenant-admin only) -------------------------------------------------------
  // GET /admin/members returns the tenant's members AND its still-pending invites in one payload.
  listMembers: () => req<{ members: Member[]; invites: Invite[] }>("/admin/members"),
  // POST /admin/invites returns a gated link (present only when EMAIL_BACKEND=none; otherwise it's
  // emailed and omitted). The accept link is shown once here, never re-listed.
  createInvite: (email: string, role: MemberRole) =>
    req<{ status: string; invite_link?: string | null }>("/admin/invites", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ email, role }),
    }),
  revokeInvite: (id: string) => req<null>(`/admin/invites/${id}`, { method: "DELETE" }),
  updateMemberRole: (userId: string, role: MemberRole) =>
    req<Member>(`/admin/members/${userId}`, {
      method: "PATCH",
      headers: JSON_HEADERS,
      body: JSON.stringify({ role }),
    }),
  removeMember: (userId: string) => req<null>(`/admin/members/${userId}`, { method: "DELETE" }),

  // --- E10 Admin Dashboard (tenant-admin) -------------------------------------------------------
  // Tenant-wide usage: headline totals, a queries-over-time series, and a per-corpus breakdown.
  getUsage: () => req<AdminUsage>("/admin/usage"),
  // Plan, limits, usage-vs-limits, and an estimated cost (a shell — billing management coming soon).
  getBilling: () => req<AdminBilling>("/admin/billing"),

  // --- E11 Platform Admin (Engram staff only) ---------------------------------------------------
  // Every tenant on the platform (one row each).
  listTenants: () => req<Tenant[]>("/platform-admin/tenants"),
  // Cost-per-tenant rows + fleet totals.
  platformUsage: () => req<PlatformUsage>("/platform-admin/usage"),
  // Pending waitlist entries awaiting approval.
  listAccessRequests: () => req<AccessRequest[]>("/platform-admin/access-requests"),
  // Approve a waitlist entry — returns a one-time invite_link to copy and send.
  approveAccessRequest: (id: string) =>
    req<ApproveAccessResponse>(`/platform-admin/access-requests/${id}/approve`, { method: "POST" }),
  // Deny a waitlist entry (no invite issued).
  denyAccessRequest: (id: string) =>
    req<{ status: string }>(`/platform-admin/access-requests/${id}/deny`, { method: "POST" }),

  listCorpora: () => req<Corpus[]>("/corpora"),
  createCorpus: (name: string) =>
    req<Corpus>("/corpora", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ name }),
    }),
  getCorpus: (id: string) => req<Corpus>(`/corpora/${id}`),
  deleteCorpus: (id: string) => req<null>(`/corpora/${id}`, { method: "DELETE" }),
  listDocuments: (id: string) => req<Document[]>(`/corpora/${id}/documents`),
  // Document-source connectors. `filesystem` is the built-in upload; external
  // connectors report available=false until their OAuth apps are configured.
  connectors: () => req<ConnectorsResponse>("/connectors"),
  uploadDocuments: (id: string, files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => {
      // Preserve folder structure: react-dropzone sets `path`; a folder <input>
      // sets `webkitRelativePath`. The backend sanitizes + nests on this key.
      const withPath = f as File & { path?: string; webkitRelativePath?: string };
      const rel = (withPath.path || withPath.webkitRelativePath || f.name).replace(/^[./]+/, "");
      fd.append("files", f, rel);
    });
    return req<Document[]>(`/corpora/${id}/documents`, { method: "POST", body: fd });
  },
  train: (id: string) => req<Job>(`/corpora/${id}/train`, { method: "POST" }),
  cancelTraining: (id: string) => req<Job>(`/corpora/${id}/cancel`, { method: "POST" }),
  getJob: (jid: string) => req<Job>(`/jobs/${jid}`),
  listJobs: (id: string) => req<Job[]>(`/corpora/${id}/jobs`),
  chat: (id: string, question: string, k = 3, history: { role: string; content: string }[] = [],
         docIds?: string[]) =>
    req<ChatResponse>(`/corpora/${id}/chat`, {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ question, k, history, ...(docIds?.length ? { doc_ids: docIds } : {}) }),
    }),
  // Token-streaming chat (SSE over fetch — EventSource can't POST/attach the JWT). Emits parsed
  // events: {head, used_docs, sources} -> {delta} xN -> {done, metrics}; {error} in-band on GPU
  // failure. `history` rides prior turns; `docIds` pins the first turn's retrieval on follow-ups.
  // Returns an AbortController so the UI can Stop mid-stream.
  chatStream: (
    id: string, question: string,
    onEvent: (e: any) => void,
    opts: { k?: number; history?: { role: string; content: string }[]; docIds?: string[] } = {},
  ) => {
    const ctrl = new AbortController();
    const run = async () => {
      const token = getToken();
      const res = await fetch(`${API_URL}/corpora/${id}/chat/stream`, {
        method: "POST",
        headers: { ...JSON_HEADERS, ...(token ? { Authorization: `Bearer ${token}` } : {}) },
        body: JSON.stringify({
          question, k: opts.k ?? 3, history: opts.history ?? [],
          ...(opts.docIds?.length ? { doc_ids: opts.docIds } : {}),
        }),
        signal: ctrl.signal,
      });
      if (res.status === 401) { handleUnauthorized(); throw new Error("Session expired"); }
      if (!res.ok || !res.body) throw new Error((await res.text()) || `stream failed (${res.status})`);
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, i).trim();
          buf = buf.slice(i + 2);
          if (chunk.startsWith("data: ")) {
            // A whole frame was split off at "\n\n", so a parse failure here is a malformed complete
            // frame (not a partial — buffering already handled those). Warn instead of dropping silently.
            try { onEvent(JSON.parse(chunk.slice(6))); }
            catch (err) { console.warn("SSE frame parse failed", err, chunk); }
          }
        }
      }
    };
    return { controller: ctrl, done: run() };
  },
  // --- Resumable onboarding wizard --------------------------------------------------------------
  modelTiers: () => req<ModelTiers>("/models"),
  getOnboarding: (id: string) => req<OnboardingState>(`/corpora/${id}/onboarding`),
  patchOnboarding: (id: string, patch: { onboarding_step?: OnboardingStep; model_tier?: string }) =>
    req<OnboardingState>(`/corpora/${id}/onboarding`, {
      method: "PATCH",
      headers: JSON_HEADERS,
      body: JSON.stringify(patch),
    }),
  estimate: (id: string) => req<OnboardEstimate>(`/corpora/${id}/estimate`),
  // Step 5. Returns the onboarding state on dispatch, or {no_serving_engine: true} on the 409 gate
  // (no live model yet) so the UI shows "starts once a model is enabled" instead of throwing.
  onboard: async (id: string): Promise<OnboardingState | { no_serving_engine: true; tier?: string }> => {
    const token = getToken();
    const res = await fetch(`${API_URL}/corpora/${id}/onboard`, {
      method: "POST",
      headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    });
    if (res.status === 409) {
      const j = await res.json().catch(() => ({}));
      if (j.status === "no_serving_engine") return { no_serving_engine: true, tier: j.tier };
    }
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      throw new Error(j.detail || `onboard failed (${res.status})`);
    }
    return res.json();
  },
  // Side-by-side: cartridge alone / cart+RAG (floor) / adaptive / RAG (latency + modeled $/query).
  // side lets the UI run the two answers sequentially (cart renders while rag still generates).
  compare: (id: string, question: string, k = 3, queriesPerMonth = 100_000,
            side: "both" | "cart" | "rag" = "both") =>
    req<CompareResult>(`/corpora/${id}/compare?queries_per_month=${queriesPerMonth}&side=${side}`, {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ question, k }),
    }),
  // Token-streaming compare side (SSE over fetch — EventSource can't POST). Emits parsed
  // events: {head, sources, used_docs} -> {delta} xN -> {done, metrics} -> {summary, cost_per_query}.
  compareStream: async (
    id: string, question: string, k: number, side: "cart" | "rag",
    onEvent: (e: any) => void,
    docIds?: string[],  // rag side: reuse the cart side's retrieval (one retrieval/question)
  ) => {
    const token = getToken();
    const res = await fetch(`${API_URL}/corpora/${id}/compare/stream?side=${side}`, {
      method: "POST",
      headers: { ...JSON_HEADERS, ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      body: JSON.stringify({ question, k, ...(docIds?.length ? { doc_ids: docIds } : {}) }),
    });
    if (res.status === 401) { handleUnauthorized(); throw new Error("Session expired"); }
    if (!res.ok || !res.body) throw new Error((await res.text()) || `stream failed (${res.status})`);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i).trim();
        buf = buf.slice(i + 2);
        if (chunk.startsWith("data: ")) {
          try { onEvent(JSON.parse(chunk.slice(6))); } catch { /* partial frame */ }
        }
      }
    }
  },
  // Live scale test (SSE): the backend drives a real concurrency ramp against the GPU inside the
  // VPC and streams one frame per level — {start,...} then {level, cart:{qps,ttft,lat,...}, rag:{...}}
  // then {done}. onEvent fires per frame so the chart fills in live.
  scaleTestStream: async (
    id: string, queries: string[], maxConcurrency: number,
    onEvent: (e: any) => void,
  ) => {
    const token = getToken();
    const res = await fetch(`${API_URL}/corpora/${id}/scale-test/stream`, {
      method: "POST",
      headers: { ...JSON_HEADERS, ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      body: JSON.stringify({ queries, max_concurrency: maxConcurrency }),
    });
    if (res.status === 401) { handleUnauthorized(); throw new Error("Session expired"); }
    if (!res.ok || !res.body) throw new Error((await res.text()) || `scale test failed (${res.status})`);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i).trim();
        buf = buf.slice(i + 2);
        if (chunk.startsWith("data: ")) {
          try { onEvent(JSON.parse(chunk.slice(6))); } catch { /* partial frame */ }
        }
      }
    }
  },
  // Saved scale-test runs: persist a finished run + list past runs (newest first, points included).
  saveScaleRun: (id: string, maxConcurrency: number, nQueries: number, points: any[]) =>
    req<ScaleRun>(`/corpora/${id}/scale-runs`, {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ max_concurrency: maxConcurrency, n_queries: nQueries, points }),
    }),
  listScaleRuns: (id: string) => req<ScaleRun[]>(`/corpora/${id}/scale-runs`),
  authConfig: () => req<AuthConfig>("/auth/config"),
  googleLoginUrl: () => `${API_URL}/auth/google/login`,
};
