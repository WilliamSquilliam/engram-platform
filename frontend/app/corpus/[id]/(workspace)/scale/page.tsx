"use client";
// Scale Test: fire a REAL concurrency ramp at the GPU and fill the chart in live. The load is driven
// server-side (the backend hits the Inference Service inside the VPC, since a browser can't generate
// true concurrency), and each level's measured result streams back here as it completes. Cartridges
// stay resident; RAG re-reads its documents on every request (multi-tenant churned cache).
import { useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import { api } from "@/lib/api";
import { SCALE_MAX, GPU_HOURLY } from "@/lib/scale";
import type { ScaleRun } from "@/lib/types";

type Arm = { qps: number; ttft: number | null; lat: number | null; ttft_p95?: number | null };
type Point = { u: number; cart: Arm; rag: Arm };

const DEFAULT_QUERIES = [
  "What features make an urban legend go viral?",
  "What criteria make a transaction successful in evaluating a speech recognition system?",
  "How does the compositional neural network approach to question answering work?",
  "How are events detected and extracted from text documents?",
  "How are paraphrases generated from latent-variable PCFGs for semantic parsing?",
  "How does joint learning of an ontology and semantic parser from text work?",
  "What are empirical Gaussian priors used for in cross-lingual transfer learning?",
];

const W = 940, H = 300, PADL = 48, PADR = 16, PADT = 14, PADB = 28;
const clampConc = (v: number) => Math.max(1, Math.min(SCALE_MAX, Number.isFinite(v) ? v : SCALE_MAX));
const costPer1k = (qps: number) => (qps > 0 ? "$" + (GPU_HOURLY / (qps * 3600) * 1000).toFixed(2) : "—");
const fmtQ = (q: number) => q.toFixed(1);
const fmtT = (t: number | null) => (t == null ? "—" : Math.round(t) + " ms");
const fmtL = (l: number | null) => (l == null ? "—" : (l / 1000).toFixed(2) + " s");
// Saved-run timestamps arrive as naive-UTC ISO; treat as UTC then render in the viewer's locale.
const fmtWhen = (iso: string) =>
  new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z")
    .toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

export default function ScalePage() {
  const { id } = useParams() as { id: string };
  const [maxConc, setMaxConc] = useState(SCALE_MAX);
  const [queries, setQueries] = useState<string[]>(DEFAULT_QUERIES);
  const [points, setPoints] = useState<Point[]>([]);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [current, setCurrent] = useState(0);
  const [err, setErr] = useState("");
  const [runs, setRuns] = useState<ScaleRun[]>([]);                     // saved past runs, newest first
  const [selectedId, setSelectedId] = useState<string | null>(null);   // null = live / new run
  const [wlOpen, setWlOpen] = useState(true);                          // workload pane starts expanded
  const abort = useRef(false);

  useEffect(() => () => { abort.current = true; }, []);

  // Load saved runs on mount; if any exist, open on the most recent so the page shows finished numbers.
  useEffect(() => {
    api.listScaleRuns(id).then((rs) => {
      setRuns(rs);
      if (rs.length) showRun(rs[0]);
    }).catch(() => { /* no saved history yet */ });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  // Re-display a saved run's finished numbers (no GPU work — just load its stored per-level points).
  function showRun(r: ScaleRun) {
    abort.current = true;
    setRunning(false); setErr(""); setCurrent(0);
    setSelectedId(r.id);
    setPoints((r.points || []) as Point[]);
    setDone(true);
  }

  function run() {
    abort.current = false;
    setSelectedId(null);
    setPoints([]); setDone(false); setErr(""); setRunning(true); setCurrent(0);
    const qs = queries.map((q) => q.trim()).filter(Boolean);
    const conc = clampConc(maxConc);
    const acc: Point[] = [];   // accumulate points here (the onEvent closure can't read live state)
    api.scaleTestStream(id, qs, conc, (e) => {
      if (abort.current) return;
      if (e.error) { setErr(e.error); setRunning(false); return; }
      if (e.start) { setCurrent(e.levels?.[0] ?? 1); return; }
      if (e.level) {
        const pt = { u: e.level, cart: e.cart, rag: e.rag };
        acc.push(pt);
        setPoints((p) => [...p, pt]);
        setCurrent(e.level);
      }
      if (e.done) {
        setDone(true); setRunning(false);
        if (acc.length)   // persist the finished run, then select it so it appears in the dropdown
          api.saveScaleRun(id, conc, qs.length, acc)
            .then((saved) => api.listScaleRuns(id).then((rs) => { setRuns(rs); setSelectedId(saved.id); }))
            .catch(() => { /* saving is best-effort; the run still displays */ });
      }
    }).catch((x) => { setErr(String(x?.message || x)); setRunning(false); });
  }
  function reset() {
    abort.current = true;
    setRunning(false); setDone(false); setPoints([]); setCurrent(0); setErr(""); setSelectedId(null);
  }

  // x-axis / dial scale to the displayed run's ceiling (a saved run keeps its own max concurrency).
  const selectedRun = selectedId ? runs.find((r) => r.id === selectedId) : null;
  const mc = clampConc(selectedRun ? selectedRun.max_concurrency : maxConc);
  const latest = points.length ? points[points.length - 1] : null;
  const qm = Math.max(2, Math.ceil((points.reduce((m, p) => Math.max(m, p.cart.qps, p.rag.qps), 0) || 4) / 2) * 2);
  const xOf = (u: number) => PADL + ((u - 1) / Math.max(mc - 1, 1)) * (W - PADL - PADR);
  const yOf = (q: number) => H - PADB - (q / qm) * (H - PADT - PADB);
  const linePts = (arm: "cart" | "rag") => points.map((p) => [xOf(p.u), yOf(p[arm].qps)] as const);
  const toPath = (pts: readonly (readonly [number, number])[]) =>
    pts.map((p, i) => (i ? "L " : "M ") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
  const cartPts = linePts("cart"), ragPts = linePts("rag");
  const area = points.length > 1
    ? toPath(cartPts) + " " + [...ragPts].reverse().map((p) => "L " + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ") + " Z"
    : "";

  const gridQ: number[] = [];
  const gstep = qm <= 12 ? 2 : qm <= 24 ? 4 : 8;
  for (let q = 0; q <= qm; q += gstep) gridQ.push(q);
  const ticks = Array.from(new Set([1, Math.round(mc * 0.25), Math.round(mc * 0.5), Math.round(mc * 0.75), mc]
    .map((v) => Math.max(1, Math.min(mc, v)))));

  const fin = done && latest ? latest : null;
  const tp = fin && fin.rag.qps > 0 ? Math.round((fin.cart.qps - fin.rag.qps) / fin.rag.qps * 100) : null;
  const ttftPct = fin && fin.rag.ttft && fin.cart.ttft ? Math.round((fin.rag.ttft - fin.cart.ttft) / fin.rag.ttft * 100) : null;
  const costPct = fin && fin.cart.qps > 0 ? Math.round((1 - fin.rag.qps / fin.cart.qps) * 100) : null;

  return (
    <div className="max-w-5xl space-y-4">
      <div className="flex items-start justify-between gap-4">
        <p className="max-w-2xl text-sm leading-relaxed text-slate-400">
          Same model, same GPU, same retriever on both sides. This fires a <b className="text-slate-200">real
          concurrency ramp</b> at the serving GPU — from 1 up to your max in-flight queries — and plots each
          level as it completes. Cartridges stay resident; RAG re-reads its documents on every request
          (multi-tenant churned cache). Watch the capacity gap open as load climbs.
        </p>
        <div className="flex shrink-0 items-center gap-2">
          <button onClick={reset} disabled={running}
            className="rounded-md border border-slate-700 px-3 py-2 text-sm text-slate-300 hover:bg-slate-800 disabled:opacity-50">Reset</button>
          <button onClick={run} disabled={running}
            className="rounded-md bg-emerald-400 px-4 py-2 text-sm font-semibold text-emerald-950 shadow hover:brightness-105 disabled:opacity-50">
            {running ? "Running…" : "▶ Run scale test"}
          </button>
        </div>
      </div>

      {err && <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300">{err}</div>}

      <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
        <details open={wlOpen} onToggle={(e) => setWlOpen(e.currentTarget.open)}
          className="flex-1 rounded-xl border border-slate-800 bg-slate-900">
        <summary className="flex cursor-pointer select-none items-center gap-2 px-4 py-3 text-sm">
          <span className="text-emerald-400">▸</span> Workload
          <b className="font-semibold text-slate-400">· {queries.length} quer{queries.length === 1 ? "y" : "ies"}</b>
          <span className="ml-auto text-xs text-slate-500">add or edit the queries this test runs</span>
        </summary>
        <div className="max-h-72 overflow-y-auto border-t border-slate-800 px-3 py-2">
          {queries.map((q, i) => (
            <div key={i} className="flex items-center gap-2 py-0.5">
              <span className="w-5 shrink-0 text-right text-xs tabular-nums text-slate-500">{i + 1}</span>
              <input value={q} placeholder="Type a query…" disabled={running}
                onChange={(ev) => setQueries((qs) => qs.map((x, j) => (j === i ? ev.target.value : x)))}
                className="flex-1 rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-300 outline-none focus:border-emerald-800 disabled:opacity-60" />
              <button title="Remove" disabled={running} onClick={() => setQueries((qs) => qs.filter((_, j) => j !== i))}
                className="px-1.5 text-slate-500 hover:text-red-400 disabled:opacity-40">×</button>
            </div>
          ))}
        </div>
        <div className="flex items-center gap-3 px-3.5 pb-3 pt-2">
          <button onClick={() => setQueries((qs) => [...qs, ""])} disabled={running}
            className="rounded-md border border-dashed border-slate-700 px-3 py-1.5 text-xs text-slate-400 hover:border-emerald-800 hover:text-emerald-400 disabled:opacity-50">
            + Add query
          </button>
          <span className="text-xs text-slate-500">Each concurrent slot pulls the next query from this list.</span>
        </div>
        </details>

        <div className="shrink-0 space-y-4 lg:w-64">
          {runs.length > 0 && (
            <div className="rounded-xl border border-slate-800 bg-slate-900 px-4 py-3">
              <label htmlFor="pastrun" className="block text-sm text-slate-400">Saved runs</label>
              <select id="pastrun" value={selectedId ?? "live"} disabled={running}
                onChange={(ev) => {
                  if (ev.target.value === "live") { reset(); return; }
                  const r = runs.find((x) => x.id === ev.target.value);
                  if (r) showRun(r);
                }}
                className="mt-2 w-full rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-200 outline-none focus:border-emerald-800 disabled:opacity-60">
                <option value="live">Live / new run</option>
                {runs.map((r, i) => (
                  <option key={r.id} value={r.id}>
                    {fmtWhen(r.created_at)} · {r.max_concurrency} concurrent{i === 0 ? " · latest" : ""}
                  </option>
                ))}
              </select>
              <p className="mt-2 text-xs text-slate-500">Pick a past run to load its finished numbers.</p>
            </div>
          )}
          <div className="rounded-xl border border-slate-800 bg-slate-900 px-4 py-3">
            <label htmlFor="maxconc" className="block text-sm text-slate-400">Max concurrent queries</label>
            <input id="maxconc" type="number" min={1} max={SCALE_MAX} step={1} value={maxConc} disabled={running}
              onChange={(ev) => setMaxConc(clampConc(parseInt(ev.target.value, 10)))}
              className="mt-2 w-full rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-center text-[15px] font-semibold text-slate-100 outline-none focus:border-emerald-800 disabled:opacity-60" />
            <p className="mt-2 text-xs text-slate-500">the ramp climbs from 1 to this many in-flight queries</p>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4 rounded-xl border border-slate-800 bg-slate-900 px-5 py-3.5">
        <span className="text-xs uppercase tracking-widest text-slate-500">{running ? "Testing at" : "Concurrent"}&nbsp;queries</span>
        <span className="min-w-[70px] text-3xl font-bold tabular-nums">{current || (latest?.u ?? 0)}</span>
        <span className="h-2 flex-1 overflow-hidden rounded-md border border-slate-800 bg-slate-950">
          <i className="block h-full rounded-md bg-gradient-to-r from-cyan-400 to-emerald-400 transition-[width] duration-300"
            style={{ width: (mc <= 1 ? 100 : (((current || latest?.u || 0) - 1) / (mc - 1)) * 100) + "%" }} />
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Panel accent name="Engram Smart CAG" sub="read-once · cartridge KV stays on the GPU" arm={latest?.cart} />
        <Panel name="Standard RAG" sub="re-reads retrieved text on every request" arm={latest?.rag} />
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
        <div className="mb-1.5 text-xs uppercase tracking-wider text-slate-500">Throughput per GPU vs concurrent queries</div>
        <div className="mb-2 flex flex-wrap gap-4 text-xs text-slate-400">
          <span className="inline-flex items-center gap-1.5"><i className="inline-block h-[3px] w-3.5 rounded bg-emerald-400" /> Smart CAG</span>
          <span className="inline-flex items-center gap-1.5"><i className="inline-block h-[3px] w-3.5 rounded bg-amber-500" /> Standard RAG (churned cache)</span>
          <span className="inline-flex items-center gap-1.5 text-slate-500"><i className="inline-block h-[3px] w-3.5 rounded bg-emerald-400/25" /> freed capacity</span>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={300} preserveAspectRatio="none">
          {gridQ.map((q) => (
            <g key={q}>
              <line x1={PADL} y1={yOf(q)} x2={W - PADR} y2={yOf(q)} stroke="#132033" strokeWidth={1} />
              <text x={PADL - 8} y={yOf(q) + 4} textAnchor="end" fill="#475569" fontSize={11}>{q}</text>
            </g>
          ))}
          {ticks.map((tk) => (
            <text key={tk} x={xOf(tk)} y={H - 8} textAnchor="middle" fill="#475569" fontSize={11}>{tk}</text>
          ))}
          {area && <path d={area} fill="rgba(52,211,153,.12)" />}
          {ragPts.length > 0 && <path d={toPath(ragPts)} fill="none" stroke="#f59e0b" strokeWidth={2.5} strokeLinejoin="round" />}
          {cartPts.length > 0 && <path d={toPath(cartPts)} fill="none" stroke="#34d399" strokeWidth={2.5} strokeLinejoin="round" />}
          {points.map((p) => (
            <g key={p.u}>
              <circle cx={xOf(p.u)} cy={yOf(p.rag.qps)} r={3.5} fill="#f59e0b" />
              <circle cx={xOf(p.u)} cy={yOf(p.cart.qps)} r={3.5} fill="#34d399" />
            </g>
          ))}
          {!points.length && <text x={W / 2} y={H / 2} textAnchor="middle" fill="#334155" fontSize={14}>Run the test to plot live throughput</text>}
        </svg>
      </div>

      {fin && (
        <div className="rounded-xl border border-emerald-900/60 bg-gradient-to-b from-emerald-400/[.08] to-transparent p-5 text-[15px] leading-relaxed">
          At <b className="text-emerald-400">{fin.u} concurrent quer{fin.u === 1 ? "y" : "ies"}</b>, Smart CAG served{" "}
          {tp != null && <><b className="text-emerald-400">{tp}% more queries per GPU</b>{" "}</>}
          {ttftPct != null && <>at <b className="text-emerald-400">{ttftPct}% lower time-to-first-token</b>{" "}</>}
          {costPct != null && <>and <b className="text-emerald-400">{costPct}% lower GPU cost per query</b></>}.
          <div className="mt-2 text-[13.5px] text-slate-400">
            RAG&apos;s throughput flatlines because it spends the GPU re-reading the same documents on every request;
            cartridges skip that work — the freed capacity is the shaded gap above.
          </div>
        </div>
      )}

      <p className="text-[11.5px] leading-relaxed text-slate-500">
        Every point is measured live on the serving GPU: the backend drives the concurrency ramp inside the VPC
        against the Inference Service and reports server-measured time-to-first-token and latency; throughput is
        completed queries per wall-second, GPU cost is the box&apos;s $/hr ÷ sustained throughput.
      </p>
    </div>
  );
}

function Panel({ name, sub, arm, accent }: { name: string; sub: string; arm?: Arm; accent?: boolean }) {
  return (
    <div className={"rounded-2xl border border-slate-800 bg-slate-900 p-5" + (accent ? " ring-1 ring-inset ring-emerald-400/15" : "")}>
      <h3 className="flex items-center gap-2 text-[15px] font-medium">
        <span className={"h-2.5 w-2.5 rounded-full " + (accent ? "bg-emerald-400 shadow-[0_0_10px] shadow-emerald-400" : "bg-amber-500")} />
        {name}
      </h3>
      <div className="mb-3.5 text-xs text-slate-500">{sub}</div>
      <div className="text-4xl font-bold tabular-nums leading-none">{arm ? fmtQ(arm.qps) : "—"}<span className="ml-2 text-base font-semibold text-slate-400">queries / sec</span></div>
      <Row k="Time to first token" v={fmtT(arm?.ttft ?? null)} />
      <Row k="Answer latency" v={fmtL(arm?.lat ?? null)} />
      <Row k="GPU cost / 1k queries" v={arm ? costPer1k(arm.qps) : "—"} />
    </div>
  );
}
function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="mt-0 flex justify-between border-t border-slate-800 py-2.5 tabular-nums first:mt-3.5">
      <span className="text-[13px] text-slate-400">{k}</span>
      <span className="text-sm font-semibold">{v}</span>
    </div>
  );
}
