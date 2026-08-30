"use client";
// Costs section: what the last training run cost (measured GPU wall-clock x GPU
// $/hr) and the break-even — how many queries until that one-time cost is repaid
// by the per-query saving vs RAG (the only realistic baseline).
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { api } from "@/lib/api";
import { atScale, costPerQuery, SCALE_MAX } from "@/lib/scale";
import type { ScaleRun } from "@/lib/types";
import { Card, CardBody, CardHeader } from "@/components/ui";

function money(x: number | null | undefined): string {
  if (x === null || x === undefined) return "—";
  if (x >= 1) return `$${x.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  if (x >= 0.001) return `$${x.toFixed(4)}`;
  return `$${x.toExponential(1)}`;
}

function qty(x: number | null | undefined): string {
  return x === null || x === undefined ? "—" : x.toLocaleString();
}

// Cost per 1,000 queries — more readable than per-query micro-cents.
function per1k(perQuery: number | null | undefined): string {
  if (perQuery === null || perQuery === undefined) return "—";
  const x = perQuery * 1000;
  if (x >= 1) return `$${x.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  if (x >= 0.001) return `$${x.toLocaleString(undefined, { maximumFractionDigits: 3 })}`;
  return `$${x.toExponential(1)}`;
}

// Snap a slider value to 2 significant figures so the label reads cleanly
// (123,456 -> 120,000) and the chosen volume is a round, shareable number.
function snapQpm(x: number): number {
  const lo = 1_000, hi = 10_000_000;
  const clamped = Math.min(hi, Math.max(lo, x));
  const mag = Math.pow(10, Math.floor(Math.log10(clamped)) - 1);
  return Math.round(clamped / mag) * mag;
}

function Stat({
  label,
  value,
  sub,
  highlight,
}: {
  label: string;
  value: string;
  sub?: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={
        highlight
          ? "rounded-lg border border-emerald-500/40 bg-emerald-500/10 p-3"
          : "rounded-lg border border-slate-800 p-3"
      }
    >
      <div className={highlight ? "text-xs text-emerald-300" : "text-xs text-slate-400"}>{label}</div>
      <div className={`text-lg font-semibold tabular-nums ${highlight ? "text-emerald-200" : ""}`}>
        {value}
      </div>
      {sub && <div className={highlight ? "text-xs text-emerald-400/70" : "text-xs text-slate-500"}>{sub}</div>}
    </div>
  );
}

// Human-readable time-to-payback: sub-day, days, or months, whichever reads cleanest.
function payback(days: number | null): string {
  if (days === null || !isFinite(days) || days <= 0) return "—";
  if (days < 1) return "< 1 day";
  if (days < 90) return `~${Math.round(days)} days`;
  return `~${Math.round(days / 30)} months`;
}

export default function CostsPage() {
  const { id } = useParams() as { id: string };
  const [e, setE] = useState<any>(null);
  const [err, setErr] = useState("");
  const [runs, setRuns] = useState<ScaleRun[]>([]);
  // Queries/month drives the break-even: RAG's per-query cost and how fast the
  // one-time training cost amortizes both move with volume. Slider is log-scale 1k→10M.
  const [qpm, setQpm] = useState(100_000);

  useEffect(() => {
    // Debounced so dragging the slider doesn't flood the API (the endpoint just
    // recomputes from stored corpus fields — cheap, no GPU).
    const t = setTimeout(() => {
      api.economics(id, qpm).then(setE).catch((x) => setErr(x.message || "failed"));
    }, 200);
    return () => clearTimeout(t);
  }, [id, qpm]);

  // This corpus's saved scale runs: price from the MOST RECENT run's numbers (below).
  useEffect(() => {
    api.listScaleRuns(id).then(setRuns).catch(() => { /* fall back to the probe constants */ });
  }, [id]);

  if (err) return <p className="text-sm text-red-400">{err}</p>;
  if (!e) return <p className="text-sm text-slate-400">Loading…</p>;
  if (!e.trained)
    return <p className="text-sm text-slate-400">Train the corpus to see cost and break-even.</p>;

  // Per-query cost AT FLEET SCALE — NOT a single idle query. With many concurrent queries,
  // continuous batching amortizes the GPU, so $/query drops far below the single-query figure.
  // RAG is the churned-cache case (other tenants evict its prefix cache), where cartridges win.
  // Source of truth: the MOST RECENT saved scale run's highest-concurrency point (this corpus, this
  // GPU, measured); the decisive-probe constants (lib/scale.ts) are only the no-runs-yet fallback.
  const latestPts = runs[0]?.points ?? [];
  const top = latestPts
    .filter((p: any) => p?.cart?.qps > 0 && p?.rag?.qps > 0)
    .reduce((a: any, b: any) => (a && a.u >= b.u ? a : b), null);
  const fleetU = top?.u ?? SCALE_MAX;
  const measuredLive = top != null; // true when priced from this corpus's own latest run
  const engPerQuery = costPerQuery(top?.cart.qps ?? atScale.cart.qps);
  const ragPerQuery = costPerQuery(top?.rag.qps ?? atScale.rag.qps);
  // Volume-dependent figures — these move with the slider (the per-query numbers don't).
  const monthlyEveryday = engPerQuery * qpm;
  const monthlyRag = ragPerQuery * qpm;
  const monthlySavings = monthlyRag - monthlyEveryday;
  // Time-to-payback: one-time training cost ÷ the daily saving at this volume.
  const paybackDays =
    monthlySavings > 0 && e.train_cost_ondemand != null
      ? e.train_cost_ondemand / (monthlySavings / 30)
      : null;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <h2 className="font-medium">Training cost (one-time)</h2>
          <p className="text-xs text-slate-400">
            Measured GPU wall-clock × GPU $/hr. Read-once: you pay this once, then every query
            rides the cheap multiplexed path.
          </p>
        </CardHeader>
        <CardBody className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          <Stat label="Cartridges trained" value={qty(e.n_cartridges)} />
          <Stat label="Corpus size" value={`${qty(e.corpus_tokens)} tok`} />
          <Stat label="GPU time" value={`${e.train_seconds?.toFixed(1) ?? "—"} s`} />
          <Stat
            label="Training cost (on-demand)"
            value={money(e.train_cost_ondemand)}
            sub={`@ $${e.gpu_hourly_ondemand}/hr · ${money(e.cost_per_cart_ondemand)}/cart`}
          />
          <Stat
            label="Training cost (spot)"
            value={money(e.train_cost_spot)}
            sub={`@ $${e.gpu_hourly_spot}/hr`}
          />
        </CardBody>
      </Card>

      <Card>
        <CardHeader>
          <h2 className="font-medium">Cost per 1,000 queries & break-even</h2>
          <p className="text-xs text-slate-400">
            Priced at fleet scale ({fleetU} concurrent queries, churned cache
            {measuredLive ? ", from this corpus's latest scale-test run" : ""}) — the realistic
            multi-tenant cost, not a single idle query. Break-even = training cost ÷ per-query saving.
          </p>
        </CardHeader>
        <CardBody className="space-y-4">
          <div>
            <div className="flex items-center justify-between text-xs text-slate-400">
              <label htmlFor="qpm">Queries per month</label>
              <span className="tabular-nums text-slate-200">{qty(qpm)}</span>
            </div>
            <input
              id="qpm"
              type="range"
              min={3}
              max={7}
              step={0.02}
              value={Math.log10(qpm)}
              onChange={(ev) => setQpm(snapQpm(Math.pow(10, parseFloat(ev.target.value))))}
              className="mt-2 w-full accent-emerald-500"
              data-testid="qpm-slider"
            />
            <div className="mt-2 flex gap-2">
              {[10_000, 100_000, 1_000_000].map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => setQpm(p)}
                  className={`rounded border px-2 py-0.5 text-xs ${
                    qpm === p
                      ? "border-emerald-500 text-emerald-300"
                      : "border-slate-700 text-slate-400 hover:border-slate-500"
                  }`}
                >
                  {p.toLocaleString()}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <Stat
              label="Engram / 1k queries"
              value={per1k(engPerQuery)}
              sub={`at ${fleetU} concurrent · ${measuredLive ? "latest run" : "measured"}`}
            />
            <Stat
              label="RAG / 1k queries"
              value={per1k(ragPerQuery)}
              sub={`churned cache · ${measuredLive ? "latest run" : "measured"}`}
            />
          </div>

          {/* Volume totals — these scale with the slider, so moving it is meaningful. */}
          <div className="grid grid-cols-3 gap-3" data-testid="monthly">
            <Stat label="Engram / month" value={money(monthlyEveryday)} sub={`${qty(qpm)} queries`} />
            <Stat label="RAG / month" value={money(monthlyRag)} sub={`${qty(qpm)} queries`} />
            <Stat
              label="You save / month"
              value={money(monthlySavings)}
              sub="vs RAG, same model"
              highlight
            />
          </div>

          <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-3 text-sm" data-testid="breakeven">
            The one-time training cost ({money(e.train_cost_ondemand)}) pays for itself in{" "}
            <b>{payback(paybackDays)}</b> at {qty(qpm)} queries/month vs RAG (the realistic same-model
            baseline). Raise the volume and the payback shortens.
          </div>
        </CardBody>
      </Card>

      <p className="text-xs text-slate-400">
        Per-query costs come from {measuredLive
          ? `this corpus's most recent scale-test run at ${fleetU} concurrent queries`
          : `the measured scale test at ${fleetU} concurrent queries`} (GPU $/hr ÷
        sustained throughput) — the fleet operating point, where continuous batching amortizes the GPU,
        not a single idle query. Onboarding GPU time is the real measured wall-clock.
      </p>
    </div>
  );
}
