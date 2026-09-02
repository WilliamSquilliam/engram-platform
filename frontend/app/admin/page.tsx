"use client";
// E10 — Admin Dashboard (tenant-admin only). Two panels: Usage (headline stat cards, a queries-over-
// time area chart, and a per-corpus table from GET /admin/usage) and Costs/billing (plan + estimated
// cost + usage-vs-limits from GET /admin/billing — a shell, billing management coming soon). Links out
// to Team and the per-corpus data controls. Non-admins who land here are bounced home, like /team.
import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, getToken } from "@/lib/api";
import type { AdminBilling, AdminUsage, BillingStatus, User } from "@/lib/types";
import { Card, CardBody, CardHeader, Button } from "@/components/ui";

// --- formatters (single place so the same number always renders the same way) -------------------
const num = (x: number | null | undefined) =>
  x === null || x === undefined ? "—" : x.toLocaleString();
const gb = (x: number | null | undefined) =>
  x === null || x === undefined ? "—" : `${x.toLocaleString(undefined, { maximumFractionDigits: 1 })} GB`;
// GPU seconds read cleanest as hours once you're past a few minutes.
const gpu = (s: number | null | undefined) => {
  if (s === null || s === undefined) return "—";
  if (s < 90) return `${Math.round(s)} s`;
  return `${(s / 3600).toLocaleString(undefined, { maximumFractionDigits: 1 })} h`;
};
const money = (x: number | null | undefined, ccy = "USD") =>
  x === null || x === undefined
    ? "—"
    : x.toLocaleString(undefined, { style: "currency", currency: ccy, maximumFractionDigits: 2 });
// Series dates arrive as ISO/date strings; show a short month-day tick.
const shortDate = (iso: string) => {
  const d = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "T00:00:00Z");
  return isNaN(d.getTime()) ? iso : d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
};

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
      <div className="text-xs text-slate-400">{label}</div>
      <div className="text-2xl font-semibold tabular-nums text-slate-100">{value}</div>
      {sub && <div className="text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

export default function AdminDashboardPage() {
  const router = useRouter();
  const [me, setMe] = useState<User | null>(null);
  const [usage, setUsage] = useState<AdminUsage | null>(null);
  const [billing, setBilling] = useState<AdminBilling | null>(null);
  // Dark-launch billing flag. Kept separate from the dashboard load: if /billing/status is missing
  // or errors, the page still renders — we just stay in the "coming soon" posture (billingStatus null).
  const [billingStatus, setBillingStatus] = useState<BillingStatus | null>(null);
  const [portalBusy, setPortalBusy] = useState(false);
  const [portalError, setPortalError] = useState("");
  const [gateChecked, setGateChecked] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      // Both panels load together; either failing surfaces one error banner.
      const [u, b] = await Promise.all([api.getUsage(), api.getBilling()]);
      setUsage(u);
      setBilling(b);
      setError("");
    } catch (err: any) {
      setError(err.message || "Couldn't load your dashboard.");
    }
    // Billing status is best-effort and non-blocking: a failure just leaves the "coming soon" state,
    // which is exactly the dark-launch default. Never let it break the dashboard.
    try {
      setBillingStatus(await api.billingStatus());
    } catch {
      setBillingStatus(null);
    }
  }, []);

  // Open the Stripe customer portal. Only reachable when billing is enabled (button is gated).
  const openPortal = useCallback(async () => {
    setPortalBusy(true);
    setPortalError("");
    try {
      const { url } = await api.billingPortal();
      window.location.assign(url);
    } catch (err: any) {
      setPortalError(err.message || "Couldn't open billing right now. Please try again.");
      setPortalBusy(false);
    }
  }, []);

  // Gate on admin role: fetch /auth/me, bounce non-admins home, otherwise load.
  useEffect(() => {
    if (!getToken()) {
      router.push("/login");
      return;
    }
    (async () => {
      try {
        const m = await api.me();
        setMe(m);
        if (m.role !== "admin") {
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

  // Hold the page until the admin gate resolves so we never flash it to a non-admin.
  if (!gateChecked) return <div className="p-8 text-slate-400">Loading…</div>;

  const ccy = billing?.currency || "USD";
  // The live two-meter rate card (memory per doc/month + inference per 1k queries; onboarding free).
  // Comes from /admin/billing's rate_card — the old per_gb_month storage rate is gone.
  const rc = billing?.rate_card;
  // Beta caps stay invisible until hit, so we deliberately do NOT render a usage-vs-limits section
  // here (no bars, no limit numbers). The estimated cost below is the single billing figure we show.

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-8">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Admin Dashboard</h1>
          <p className="text-sm text-slate-400">Usage and costs across your workspace.</p>
        </div>
        <div className="flex items-center gap-2">
          <Link href="/team">
            <Button variant="outline" data-testid="dashboard-team-link">Manage team</Button>
          </Link>
        </div>
      </div>

      {error && (
        <p data-testid="dashboard-error" className="text-sm text-red-400">
          {error}
        </p>
      )}

      {/* ---- Usage ---------------------------------------------------------------------------- */}
      <Card>
        <CardHeader>
          <h2 className="font-medium">Usage</h2>
          <p className="text-xs text-slate-400">Queries, documents, storage, and GPU time this period.</p>
        </CardHeader>
        <CardBody className="space-y-6">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5" data-testid="usage-stats">
            <Stat label="Queries" value={num(usage?.queries)} />
            <Stat label="Documents" value={num(usage?.documents)} />
            <Stat label="Storage" value={gb(usage?.storage_gb)} />
            <Stat label="GPU time" value={gpu(usage?.gpu_seconds)} />
            <Stat label="Document Bases" value={num(usage?.n_corpora)} />
          </div>

          {/* Queries over time — recharts area chart. */}
          <div>
            <div className="mb-2 text-xs uppercase tracking-wider text-slate-500">Queries over time</div>
            {usage && usage.series.length > 0 ? (
              <div className="h-64 w-full" data-testid="usage-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={usage.series} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
                    <defs>
                      <linearGradient id="queriesFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#34d399" stopOpacity={0.35} />
                        <stop offset="100%" stopColor="#34d399" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#1e293b" vertical={false} />
                    <XAxis
                      dataKey="date"
                      tickFormatter={shortDate}
                      tick={{ fill: "#64748b", fontSize: 11 }}
                      stroke="#334155"
                    />
                    <YAxis
                      tick={{ fill: "#64748b", fontSize: 11 }}
                      stroke="#334155"
                      allowDecimals={false}
                      width={48}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#0f172a",
                        border: "1px solid #1e293b",
                        borderRadius: 8,
                        color: "#e2e8f0",
                        fontSize: 12,
                      }}
                      labelFormatter={(l) => shortDate(String(l))}
                      formatter={(v: any) => [num(Number(v)), "Queries"]}
                    />
                    <Area
                      type="monotone"
                      dataKey="queries"
                      stroke="#34d399"
                      strokeWidth={2}
                      fill="url(#queriesFill)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="text-sm text-slate-500">No query activity yet.</p>
            )}
          </div>

          {/* Per-corpus breakdown table. */}
          <div>
            <div className="mb-2 text-xs uppercase tracking-wider text-slate-500">By document base</div>
            {usage && usage.by_corpus.length > 0 ? (
              <div className="overflow-x-auto rounded-lg border border-slate-800">
                <table className="w-full text-sm" data-testid="usage-by-corpus">
                  <thead className="bg-slate-950 text-xs uppercase tracking-wider text-slate-500">
                    <tr>
                      <th className="px-3 py-2 text-left font-medium">Document Base</th>
                      <th className="px-3 py-2 text-right font-medium">Queries</th>
                      <th className="px-3 py-2 text-right font-medium">Documents</th>
                      <th className="px-3 py-2 text-right font-medium">Storage</th>
                      <th className="px-3 py-2 text-right font-medium">GPU time</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {usage.by_corpus.map((c) => (
                      <tr key={c.corpus_id} className="hover:bg-slate-800/40">
                        <td className="px-3 py-2">
                          <Link
                            href={`/document-base/${c.corpus_id}`}
                            className="text-slate-100 hover:text-emerald-300 hover:underline"
                          >
                            {c.name}
                          </Link>
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">{num(c.queries)}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{num(c.documents)}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{gb(c.storage_gb)}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{gpu(c.gpu_seconds)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-slate-500">No document bases yet.</p>
            )}
          </div>
        </CardBody>
      </Card>

      {/* ---- Costs / billing ------------------------------------------------------------------ */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <h2 className="font-medium">Costs &amp; billing</h2>
            <span className="rounded-full bg-slate-800 px-2 py-0.5 text-xs text-slate-400">
              Billing management coming soon
            </span>
          </div>
          <p className="text-xs text-slate-400">Your plan, estimated cost this period, and usage against your limits.</p>
        </CardHeader>
        <CardBody className="space-y-6">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <Stat label="Plan" value={billing?.plan ?? "—"} />
            <Stat
              label="Estimated cost"
              value={money(billing?.estimated_cost_usd, ccy)}
              sub={billing?.period ? `Period: ${billing.period}` : undefined}
            />
            <Stat label="Currency" value={ccy} />
          </div>

          {/* Two-meter rate card: memory (per document per month) + inference (per 1,000 questions),
              with onboarding called out as free. Single source of truth for what things cost. */}
          {rc && (
            <div className="space-y-3" data-testid="rate-card">
              <div className="text-xs uppercase tracking-wider text-slate-500">What you pay for</div>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
                  <div className="text-sm font-medium text-slate-200">Memory</div>
                  <div className="mt-1 text-lg font-semibold tabular-nums text-slate-100">
                    {money(rc.per_doc_month_usd, ccy)} per document per month
                  </div>
                  <div className="mt-1 text-xs text-slate-500">
                    What it costs to keep your documents ready to answer questions.
                  </div>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
                  <div className="text-sm font-medium text-slate-200">Questions answered</div>
                  <div className="mt-1 text-lg font-semibold tabular-nums text-slate-100">
                    {money(rc.per_1k_queries_usd, ccy)} per 1,000 questions answered
                  </div>
                  <div className="mt-1 text-xs text-slate-500">
                    You only pay for the questions your team actually asks.
                  </div>
                </div>
              </div>
              <p className="text-sm text-emerald-300" data-testid="onboarding-free">
                Adding documents is free.
              </p>
            </div>
          )}

          {/* Billing management. Dark launch: until the backend flips `enabled` true, this looks
              exactly as it did before — a "coming soon" posture with no button and no Stripe mention.
              When enabled, a Manage billing button opens the Stripe customer portal. */}
          {billingStatus?.enabled ? (
            <div className="space-y-2" data-testid="billing-management">
              <Button onClick={openPortal} disabled={portalBusy} data-testid="manage-billing">
                {portalBusy ? "Opening…" : "Manage billing"}
              </Button>
              {portalError && <p className="text-sm text-red-400">{portalError}</p>}
              <p className="text-xs text-slate-500">
                Update your payment method, download invoices, and manage your plan.
              </p>
            </div>
          ) : (
            <p className="text-xs text-slate-500">
              Self-serve billing is on the way. For plan changes or invoices, reach out to your Engram contact.
            </p>
          )}
        </CardBody>
      </Card>

      {/* Data controls: per-corpus training cost & break-even live under each corpus. */}
      <p className="text-xs text-slate-500">
        Looking for per-document-base cost and break-even? Open a document base and visit its{" "}
        <span className="text-slate-300">Costs</span> tab. Manage teammates on the{" "}
        <Link href="/team" className="text-emerald-300 hover:underline">Team</Link> page.
      </p>
    </div>
  );
}
