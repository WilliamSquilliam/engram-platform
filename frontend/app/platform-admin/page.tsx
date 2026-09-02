"use client";
// E11 — Platform Admin console (Engram staff only; gated on me.platform_admin). Three panels:
// Tenants (a table from GET /platform-admin/tenants), Cost per tenant (a table + fleet totals + a
// recharts bar chart of est_cost per tenant from GET /platform-admin/usage), and Waitlist approvals
// (pending access requests from GET /platform-admin/access-requests with Approve/Deny — approve
// returns a one-time invite_link to copy). Everyone without platform_admin is bounced home.
import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, getToken } from "@/lib/api";
import type {
  AccessRequest,
  GpuState,
  GpuStatus,
  PlatformUsage,
  Tenant,
  User,
} from "@/lib/types";
import { Card, CardBody, CardHeader, Button, Badge, cn } from "@/components/ui";

const num = (x: number | null | undefined) =>
  x === null || x === undefined ? "—" : x.toLocaleString();
const gb = (x: number | null | undefined) =>
  x === null || x === undefined ? "—" : `${x.toLocaleString(undefined, { maximumFractionDigits: 1 })} GB`;
const gpu = (s: number | null | undefined) => {
  if (s === null || s === undefined) return "—";
  if (s < 90) return `${Math.round(s)} s`;
  return `${(s / 3600).toLocaleString(undefined, { maximumFractionDigits: 1 })} h`;
};
const money = (x: number | null | undefined) =>
  x === null || x === undefined
    ? "—"
    : x.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 2 });
const fmtWhen = (iso: string) => {
  const d = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z");
  return isNaN(d.getTime())
    ? iso
    : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
};

// Status pill colors for tenants (active/trial/suspended and the like).
const STATUS_COLOR: Record<string, string> = {
  active: "green",
  trial: "blue",
  suspended: "red",
  paused: "amber",
};

// Copy-to-clipboard button with a brief "Copied" state (mirrors the Team page pattern).
function CopyButton({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      type="button"
      variant="outline"
      className="shrink-0"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {
          /* clipboard blocked — the link stays visible for manual copy */
        }
      }}
    >
      {copied ? "Copied" : "Copy"}
    </Button>
  );
}

export default function PlatformAdminPage() {
  const router = useRouter();
  const [tenants, setTenants] = useState<Tenant[]>([]);
  const [usage, setUsage] = useState<PlatformUsage | null>(null);
  const [requests, setRequests] = useState<AccessRequest[]>([]);
  // One-time invite links from this session's approvals. Held HERE (not in the row) because
  // approving refreshes the pending list, which unmounts the row — row-local state would
  // destroy the only copy of the link before the operator could send it.
  const [approvedLinks, setApprovedLinks] = useState<
    { id: string; email: string; tenant: string; link: string | null }[]
  >([]);
  const [gateChecked, setGateChecked] = useState(false);
  const [error, setError] = useState("");

  const onApproved = useCallback((r: AccessRequest, link: string | null) => {
    setApprovedLinks((a) => [...a, { id: r.id, email: r.email, tenant: r.tenant_name, link }]);
  }, []);

  const load = useCallback(async () => {
    try {
      const [ts, u, rs] = await Promise.all([
        api.listTenants(),
        api.platformUsage(),
        api.listAccessRequests(),
      ]);
      setTenants(ts);
      setUsage(u);
      setRequests(rs);
      setError("");
    } catch (err: any) {
      setError(err.message || "Couldn't load the platform console.");
    }
  }, []);

  // Gate on platform_admin: fetch /auth/me, bounce everyone else home.
  useEffect(() => {
    if (!getToken()) {
      router.push("/login");
      return;
    }
    (async () => {
      try {
        const m: User = await api.me();
        if (!m.platform_admin) {
          router.replace("/");
          return;
        }
        setGateChecked(true);
        await load();
      } catch {
        router.push("/login");
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!gateChecked) return <div className="p-8 text-slate-400">Loading…</div>;

  const totals = usage?.totals ?? {};
  // Bar chart of estimated cost per tenant (highest-spend first reads best).
  const costBars = (usage?.tenants ?? [])
    .map((t) => ({ name: t.name, cost: t.est_cost_usd }))
    .sort((a, b) => b.cost - a.cost);

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-8">
      <div>
        <h1 className="text-2xl font-semibold">Platform Admin</h1>
        <p className="text-sm text-slate-400">Fleet-wide tenants, cost, and waitlist approvals.</p>
      </div>

      {error && (
        <p data-testid="platform-error" className="text-sm text-red-400">
          {error}
        </p>
      )}

      {/* ---- GPU serving control (top: the fleet-wide switch that gates all chat/onboarding) --- */}
      <GpuServingCard />

      {/* ---- Waitlist approvals (top: it's the actionable panel) ------------------------------- */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <h2 className="font-medium">Waitlist approvals</h2>
            {requests.length > 0 && <Badge color="amber">{requests.length} pending</Badge>}
          </div>
          <p className="text-xs text-slate-400">
            Approve to issue an invite link, or deny. Approving returns a one-time link to send.
          </p>
        </CardHeader>
        <CardBody className="space-y-4">
          {approvedLinks.length > 0 && (
            <div className="space-y-2" data-testid="approved-links">
              {approvedLinks.map((a) => (
                <div key={a.id} className="rounded-md border border-emerald-900/60 bg-slate-950 p-3">
                  <p className="mb-2 text-xs text-slate-400">
                    <Badge color="green">Approved</Badge>{" "}
                    <span className="text-slate-200">{a.tenant}</span> · {a.email}
                    {a.link ? " — send them this one-time link:" : " — invite emailed."}
                  </p>
                  {a.link && (
                    <div className="flex items-center gap-2">
                      <code
                        data-testid="approved-invite-link"
                        className="min-w-0 flex-1 truncate rounded bg-slate-800 px-2 py-1.5 text-xs text-slate-100"
                      >
                        {a.link}
                      </code>
                      <CopyButton value={a.link} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
          {requests.length === 0 ? (
            <p className="text-sm text-slate-500">No pending requests.</p>
          ) : (
            <ul className="divide-y divide-slate-800" data-testid="access-requests">
              {requests.map((r) => (
                <AccessRequestRow key={r.id} request={r} onChanged={load} onApproved={onApproved} />
              ))}
            </ul>
          )}
        </CardBody>
      </Card>

      {/* ---- Tenants -------------------------------------------------------------------------- */}
      <Card>
        <CardHeader>
          <h2 className="font-medium">Tenants</h2>
          <p className="text-xs text-slate-400">{tenants.length} tenant{tenants.length === 1 ? "" : "s"} on the platform.</p>
        </CardHeader>
        <CardBody>
          {tenants.length === 0 ? (
            <p className="text-sm text-slate-500">No tenants yet.</p>
          ) : (
            <div className="overflow-x-auto rounded-lg border border-slate-800">
              <table className="w-full text-sm" data-testid="tenants-table">
                <thead className="bg-slate-950 text-xs uppercase tracking-wider text-slate-500">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium">Tenant</th>
                    <th className="px-3 py-2 text-left font-medium">Plan</th>
                    <th className="px-3 py-2 text-left font-medium">Status</th>
                    <th className="px-3 py-2 text-right font-medium">Users</th>
                    <th className="px-3 py-2 text-right font-medium">Document Bases</th>
                    <th className="px-3 py-2 text-right font-medium">Created</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {tenants.map((t) => (
                    <tr key={t.id} className="hover:bg-slate-800/40">
                      <td className="px-3 py-2 text-slate-100">{t.name}</td>
                      <td className="px-3 py-2 text-slate-300">{t.plan}</td>
                      <td className="px-3 py-2">
                        <Badge color={STATUS_COLOR[t.status] || "slate"}>{t.status}</Badge>
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">{num(t.n_users)}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{num(t.n_corpora)}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-slate-400">{fmtWhen(t.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardBody>
      </Card>

      {/* ---- Cost per tenant ------------------------------------------------------------------ */}
      <Card>
        <CardHeader>
          <h2 className="font-medium">Cost per tenant</h2>
          <p className="text-xs text-slate-400">Estimated GPU + storage cost by tenant, with fleet totals.</p>
        </CardHeader>
        <CardBody className="space-y-6">
          {/* Fleet totals. */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4" data-testid="fleet-totals">
            <Stat label="Total queries" value={num(totals.queries)} />
            <Stat label="Total storage" value={gb(totals.storage_gb)} />
            <Stat label="Total GPU time" value={gpu(totals.gpu_seconds)} />
            <Stat label="Est. cost" value={money(totals.est_cost_usd)} />
          </div>

          {/* Bar chart of est cost per tenant. */}
          {costBars.length > 0 && (
            <div className="h-64 w-full" data-testid="cost-chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={costBars} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
                  <CartesianGrid stroke="#1e293b" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "#64748b", fontSize: 11 }}
                    stroke="#334155"
                    interval={0}
                    angle={costBars.length > 6 ? -20 : 0}
                    textAnchor={costBars.length > 6 ? "end" : "middle"}
                    height={costBars.length > 6 ? 48 : 24}
                  />
                  <YAxis
                    tick={{ fill: "#64748b", fontSize: 11 }}
                    stroke="#334155"
                    width={56}
                    tickFormatter={(v: any) => money(Number(v))}
                  />
                  <Tooltip
                    cursor={{ fill: "#1e293b55" }}
                    contentStyle={{
                      background: "#0f172a",
                      border: "1px solid #1e293b",
                      borderRadius: 8,
                      color: "#e2e8f0",
                      fontSize: 12,
                    }}
                    formatter={(v: any) => [money(Number(v)), "Est. cost"]}
                  />
                  <Bar dataKey="cost" radius={[4, 4, 0, 0]}>
                    {costBars.map((_, i) => (
                      <Cell key={i} fill="#34d399" />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Per-tenant table. */}
          {usage && usage.tenants.length > 0 ? (
            <div className="overflow-x-auto rounded-lg border border-slate-800">
              <table className="w-full text-sm" data-testid="cost-table">
                <thead className="bg-slate-950 text-xs uppercase tracking-wider text-slate-500">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium">Tenant</th>
                    <th className="px-3 py-2 text-right font-medium">Queries</th>
                    <th className="px-3 py-2 text-right font-medium">Storage</th>
                    <th className="px-3 py-2 text-right font-medium">GPU time</th>
                    <th className="px-3 py-2 text-right font-medium">Est. cost</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {usage.tenants.map((t) => (
                    <tr key={t.tenant_id} className="hover:bg-slate-800/40">
                      <td className="px-3 py-2 text-slate-100">{t.name}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{num(t.queries)}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{gb(t.storage_gb)}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{gpu(t.gpu_seconds)}</td>
                      <td className="px-3 py-2 text-right font-medium tabular-nums text-emerald-300">
                        {money(t.est_cost_usd)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-sm text-slate-500">No usage yet.</p>
          )}
        </CardBody>
      </Card>
    </div>
  );
}

// ---- GPU serving control ------------------------------------------------------------------------
// The serving box is one Lambda Cloud instance. There is no pause: Stop terminates it (GPU billing
// -> $0/hr) and Start launches a fresh one that auto-provisions (~10-20 min to serving). Data is
// durable across stop/start (document bases, cartridges, model weights), so the confirm copy leads
// with that. Self-contained: owns its own status fetch + adaptive polling, independent of the
// page's tenant/usage/waitlist loads.

// Human labels + pill color per state (never show the raw enum). Pulse only on the initial launch.
const GPU_PILL: Record<GpuState, { label: string; color: string; pulse?: boolean }> = {
  serving: { label: "Serving", color: "green" },
  warming: { label: "Warming up", color: "amber" },
  provisioning: { label: "Provisioning", color: "amber" },
  booting: { label: "Starting", color: "amber", pulse: true },
  terminating: { label: "Stopping", color: "red" },
  offline: { label: "Offline", color: "slate" },
};

// One-sentence explanation of what the box is doing right now.
const GPU_STATE_LINE: Record<GpuState, string> = {
  serving: "Model loaded and answering queries.",
  warming: "Loading model weights into GPU memory.",
  provisioning: "Instance is up, installing the serving stack.",
  booting: "Launching a fresh cloud instance.",
  terminating: "Shutting the instance down. GPU billing stops when it is gone.",
  offline: "GPU is stopped. Chat and onboarding are paused until you start it.",
};

// States that are actively transitioning — poll fast (10s) so the pill tracks progress.
const GPU_TRANSITIONAL: GpuState[] = ["booting", "provisioning", "warming", "terminating"];

function GpuServingCard() {
  const [status, setStatus] = useState<GpuStatus | null>(null);
  const [loaded, setLoaded] = useState(false);
  // Inline result of a start/stop action (409 already-running / 503 no-capacity detail messages).
  const [actionError, setActionError] = useState("");
  const [pending, setPending] = useState<"start" | "stop" | null>(null);
  // Which confirm dialog is open (daisyUI modal), if any.
  const [confirming, setConfirming] = useState<"start" | "stop" | null>(null);
  // A just-fired request forces fast polling until the next status settles, even before `state`
  // has flipped to a transitional value on the server.
  const justActed = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const s = await api.gpuStatus();
      setStatus(s);
    } catch {
      // A transient status blip shouldn't blow away the last known good state; keep showing it.
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Adaptive polling: 10s while transitioning or a request just fired, 60s when settled. Re-arm the
  // timer whenever the state changes so the interval switches without a leaked timer.
  const state = status?.state;
  useEffect(() => {
    const fast =
      justActed.current || (!!state && GPU_TRANSITIONAL.includes(state));
    const id = setInterval(refresh, fast ? 10_000 : 60_000);
    return () => clearInterval(id);
  }, [state, refresh, pending]);

  async function doStart() {
    setConfirming(null);
    setPending("start");
    setActionError("");
    justActed.current = true;
    try {
      const res = await api.gpuStart();
      // Optimistically reflect the transition the server reported so the pill flips immediately.
      setStatus((s) => (s ? { ...s, state: res.state } : s));
      await refresh();
    } catch (e: any) {
      setActionError(e.message || "Couldn't start the GPU.");
    } finally {
      setPending(null);
    }
  }

  async function doStop() {
    setConfirming(null);
    setPending("stop");
    setActionError("");
    justActed.current = true;
    try {
      const res = await api.gpuStop();
      setStatus((s) => (s ? { ...s, state: res.state } : s));
      await refresh();
    } catch (e: any) {
      setActionError(e.message || "Couldn't stop the GPU.");
    } finally {
      setPending(null);
    }
  }

  // Clear the just-acted flag once the box reaches a settled state (so polling relaxes to 60s).
  useEffect(() => {
    if (state === "serving" || state === "offline") justActed.current = false;
  }, [state]);

  if (!loaded) return null; // Nothing to show until the first status lands.
  if (!status || !status.enabled) return null; // enabled=false -> render nothing at all.

  const pill = GPU_PILL[status.state] ?? GPU_PILL.offline;
  const inst = status.instance;
  const isOffline = status.state === "offline";
  // Stop is offered while the box is up or coming up; Start only when fully offline.
  const canStop = ["serving", "warming", "provisioning", "booting"].includes(status.state);
  const costLabel = isOffline
    ? "$0.00/hr while stopped"
    : status.hourly_usd != null
      ? `$${status.hourly_usd.toFixed(2)}/hr`
      : "—";

  return (
    <Card data-testid="gpu-card">
      <CardHeader>
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <h2 className="font-medium">GPU Serving</h2>
            <span
              data-testid="gpu-pill"
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium",
                pill.color === "green" && "bg-emerald-500/15 text-emerald-300",
                pill.color === "amber" && "bg-amber-500/15 text-amber-300",
                pill.color === "red" && "bg-red-500/15 text-red-300",
                pill.color === "slate" && "bg-slate-800 text-slate-300"
              )}
            >
              {pill.pulse && (
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
              )}
              {pill.label}
            </span>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {isOffline && (
              <Button
                type="button"
                data-testid="gpu-start"
                disabled={pending !== null}
                onClick={() => setConfirming("start")}
              >
                {pending === "start" ? (
                  <span className="flex items-center gap-2">
                    <Spinner /> Starting…
                  </span>
                ) : (
                  "Start GPU"
                )}
              </Button>
            )}
            {canStop && (
              <Button
                type="button"
                variant="danger"
                data-testid="gpu-stop"
                disabled={pending !== null}
                onClick={() => setConfirming("stop")}
              >
                {pending === "stop" ? (
                  <span className="flex items-center gap-2">
                    <Spinner /> Stopping…
                  </span>
                ) : (
                  "Stop GPU"
                )}
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardBody className="space-y-3">
        {/* Facts row — only meaningful when an instance exists. */}
        {inst && (
          <div
            data-testid="gpu-facts"
            className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-400"
          >
            <span>
              Type <span className="text-slate-200">{inst.type}</span>
            </span>
            <span>
              Region <span className="text-slate-200">{inst.region}</span>
            </span>
            <span>
              IP <span className="text-slate-200">{inst.ip ?? "—"}</span>
            </span>
            <span>
              Cost <span className="text-slate-200">{costLabel}</span>
            </span>
          </div>
        )}
        {/* When there's no instance, cost still reads $0.00/hr while stopped. */}
        {!inst && (
          <div className="text-xs text-slate-400">
            Cost <span className="text-slate-200">{costLabel}</span>
          </div>
        )}

        {/* One-sentence state explanation. */}
        <p data-testid="gpu-state-line" className="text-sm text-slate-300">
          {GPU_STATE_LINE[status.state]}
        </p>

        {/* Inline start/stop errors (409 already-running / 503 no-capacity), server detail verbatim. */}
        {actionError && (
          <div
            data-testid="gpu-action-error"
            role="alert"
            className="rounded-md border border-red-900/60 bg-red-950/40 px-3 py-2 text-sm text-red-300"
          >
            {actionError}
          </div>
        )}
      </CardBody>

      {/* Start confirm. */}
      <GpuConfirmDialog
        open={confirming === "start"}
        title="Start the GPU?"
        confirmLabel="Start GPU"
        onConfirm={doStart}
        onCancel={() => setConfirming(null)}
      >
        <p>
          Starting launches a fresh cloud instance that provisions itself. Chat and onboarding come
          back in roughly 10 to 20 minutes once the model is loaded.
        </p>
        <p>GPU billing starts the moment the instance launches.</p>
      </GpuConfirmDialog>

      {/* Stop confirm — destructive, so the confirm button is btn-error. */}
      <GpuConfirmDialog
        open={confirming === "stop"}
        title="Stop the GPU?"
        confirmLabel="Stop GPU"
        destructive
        onConfirm={doStop}
        onCancel={() => setConfirming(null)}
      >
        <p>Stopping ends GPU billing immediately and drops the hourly cost to $0.00.</p>
        <p>
          All data stays safe. Document bases, cartridges, and model weights live on durable
          storage and survive the stop.
        </p>
        <p>
          Chat and onboarding go offline until you start the GPU again, and anything running right
          now will fail.
        </p>
      </GpuConfirmDialog>
    </Card>
  );
}

// Small inline spinner for buttons with a request in flight.
function Spinner() {
  return (
    <span
      aria-hidden
      className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent"
    />
  );
}

// Confirm dialog on daisyUI's `modal` component (library CSS, not bespoke) — same data-theme="engram"
// approach the Stepper uses so it picks up the brand palette. Driven by a native <dialog> so Escape
// and the backdrop close it. The destructive action gets btn-error to stand out.
function GpuConfirmDialog({
  open,
  title,
  confirmLabel,
  destructive,
  onConfirm,
  onCancel,
  children,
}: {
  open: boolean;
  title: string;
  confirmLabel: string;
  destructive?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);

  // Keep the native dialog's open state in sync with `open` so backdrop/Escape and React agree.
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      data-theme="engram"
      className="modal"
      data-testid="gpu-confirm"
      onClose={onCancel}
    >
      <div className="modal-box border border-slate-800 bg-slate-900 text-slate-100">
        <h3 className="text-lg font-semibold">{title}</h3>
        <div className="mt-3 space-y-2 text-sm text-slate-300">{children}</div>
        <div className="modal-action">
          <Button type="button" variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button
            type="button"
            variant={destructive ? "danger" : "default"}
            data-testid="gpu-confirm-action"
            onClick={onConfirm}
          >
            {confirmLabel}
          </Button>
        </div>
      </div>
      {/* Clicking the backdrop cancels. */}
      <form method="dialog" className="modal-backdrop">
        <button aria-label="Close" onClick={onCancel}>
          close
        </button>
      </form>
    </dialog>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
      <div className="text-xs text-slate-400">{label}</div>
      <div className="text-2xl font-semibold tabular-nums text-slate-100">{value}</div>
    </div>
  );
}

// One pending waitlist entry with Approve / Deny. The one-time invite link is reported UP via
// onApproved — the refresh after approval unmounts this row, so the parent must own the link.
function AccessRequestRow({
  request,
  onChanged,
  onApproved,
}: {
  request: AccessRequest;
  onChanged: () => Promise<void>;
  onApproved: (r: AccessRequest, link: string | null) => void;
}) {
  const [busy, setBusy] = useState<"approve" | "deny" | null>(null);
  const [link, setLink] = useState<string | null>(null);
  const [done, setDone] = useState<"approved" | "denied" | null>(null);
  const [err, setErr] = useState("");

  async function approve() {
    setBusy("approve");
    setErr("");
    try {
      const res = await api.approveAccessRequest(request.id);
      setDone("approved");
      setLink(res.invite_link ?? null);
      onApproved(request, res.invite_link ?? null);
      onChanged().catch(() => {});
    } catch (e: any) {
      setErr(e.message || "Approve failed.");
    } finally {
      setBusy(null);
    }
  }

  async function deny() {
    setBusy("deny");
    setErr("");
    try {
      await api.denyAccessRequest(request.id);
      setDone("denied");
      onChanged().catch(() => {});
    } catch (e: any) {
      setErr(e.message || "Deny failed.");
    } finally {
      setBusy(null);
    }
  }

  return (
    <li data-testid="access-request" className="py-3">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-medium text-slate-100">{request.tenant_name}</span>
            {done === "approved" && <Badge color="green">Approved</Badge>}
            {done === "denied" && <Badge color="red">Denied</Badge>}
          </div>
          <div className="truncate text-xs text-slate-400">
            {request.name} · {request.email} · {fmtWhen(request.created_at)}
          </div>
          {request.reason && (
            <div className="mt-1 max-w-xl text-xs text-slate-500">“{request.reason}”</div>
          )}
        </div>

        {!done && (
          <div className="flex shrink-0 items-center gap-2">
            <Button
              type="button"
              data-testid="approve-request"
              disabled={busy !== null}
              onClick={approve}
            >
              {busy === "approve" ? "…" : "Approve"}
            </Button>
            <Button
              type="button"
              variant="danger"
              data-testid="deny-request"
              disabled={busy !== null}
              onClick={deny}
            >
              {busy === "deny" ? "…" : "Deny"}
            </Button>
          </div>
        )}
      </div>

      {err && <p className="mt-2 text-xs text-red-400">{err}</p>}

      {/* The one-time invite link from an approval — copy and send it. */}
      {done === "approved" &&
        (link ? (
          <div
            data-testid="approve-link-panel"
            className="mt-2 rounded-md border border-slate-800 bg-slate-950 p-3"
          >
            <p className="mb-2 text-xs text-slate-400">
              Invite link for <span className="text-slate-200">{request.email}</span> — share it:
            </p>
            <div className="flex items-center gap-2">
              <code
                data-testid="approve-invite-link"
                className={cn(
                  "min-w-0 flex-1 truncate rounded bg-slate-800 px-2 py-1.5 text-xs text-slate-100"
                )}
              >
                {link}
              </code>
              <CopyButton value={link} />
            </div>
          </div>
        ) : (
          <p className="mt-2 text-xs text-slate-400">Approved — an invite has been emailed.</p>
        ))}
    </li>
  );
}
