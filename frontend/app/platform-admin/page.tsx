"use client";
// E11 — Platform Admin console (Engram staff only; gated on me.platform_admin). Three panels:
// Tenants (a table from GET /platform-admin/tenants), Cost per tenant (a table + fleet totals + a
// recharts bar chart of est_cost per tenant from GET /platform-admin/usage), and Waitlist approvals
// (pending access requests from GET /platform-admin/access-requests with Approve/Deny — approve
// returns a one-time invite_link to copy). Everyone without platform_admin is bounced home.
import { useCallback, useEffect, useState } from "react";
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
import type { AccessRequest, PlatformUsage, Tenant, User } from "@/lib/types";
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
