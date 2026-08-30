"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { Button, Card, CardBody, CardHeader, Badge } from "@/components/ui";

const CORPUS_PRESETS = [
  { label: "100k (small KB)", v: 100_000 },
  { label: "1M (large KB)", v: 1_000_000 },
  { label: "10M (huge KB)", v: 10_000_000 },
];
const VOLUME_PRESETS = [
  { label: "10k / mo", v: 10_000 },
  { label: "100k / mo", v: 100_000 },
  { label: "1M / mo", v: 1_000_000 },
];

const COLOR: Record<string, string> = {
  cartridge: "bg-emerald-500",
  rag: "bg-amber-500",
};

function money(x: number | null): string {
  if (x === null || x === undefined) return "—";
  if (x >= 1) return `$${x.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
  if (x >= 0.001) return `$${x.toFixed(4)}`;
  return `$${x.toExponential(1)}`;
}

// Cost per 1,000 queries — more readable than per-query micro-cents.
function per1k(x: number | null | undefined): string {
  if (x === null || x === undefined) return "—";
  const v = x * 1000;
  if (v >= 1) return `$${v.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  if (v >= 0.001) return `$${v.toLocaleString(undefined, { maximumFractionDigits: 3 })}`;
  return `$${v.toExponential(1)}`;
}

function ms(x: number | null | undefined): string {
  if (x === null || x === undefined) return "—";
  return x >= 1000 ? `${(x / 1000).toFixed(2)} s` : `${Math.round(x)} ms`;
}
function tok(x: number | null | undefined): string {
  return x === null || x === undefined ? "—" : `${x.toLocaleString()} tok`;
}
function xstr(x: number | null | undefined): string {
  return x && x > 1 ? `${x}×` : "—";
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-slate-800 p-3">
      <div className="text-xs text-slate-400">{label}</div>
      <div className="text-lg font-semibold tabular-nums">{value}</div>
      {sub && <div className="text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

export default function DemoPage() {
  const [corpus, setCorpus] = useState(1_000_000);
  const [volume, setVolume] = useState(100_000);
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState("");

  async function load() {
    setErr("");
    try {
      setData(await api.costComparison(corpus, volume));
    } catch (e: any) {
      setErr(e.message || "failed");
    }
  }
  useEffect(() => {
    load();
  }, [corpus, volume]);

  const maxPerQuery = data
    ? Math.max(...data.strategies.filter((s: any) => s.per_query != null).map((s: any) => s.per_query))
    : 1;

  return (
    <main className="max-w-4xl mx-auto p-6">
      <header className="flex items-center justify-between mb-2">
        <h1 className="text-2xl font-semibold">Cost Comparison</h1>
        <Link href="/" className="text-sm text-slate-400 hover:text-slate-200">
          ← Back to App
        </Link>
      </header>
      <p className="text-sm text-slate-400 mb-6">
        Serving Q&A over a fixed corpus: <b>this platform (cartridges)</b> vs <b>RAG</b> on the same
        open model — the realistic baseline. Pick a corpus size and monthly query volume.
      </p>

      <Card className="mb-6">
        <CardBody className="space-y-4">
          <div>
            <div className="text-sm font-medium mb-1">Corpus size</div>
            <div className="flex flex-wrap gap-2">
              {CORPUS_PRESETS.map((p) => (
                <Button
                  key={p.v}
                  data-testid={`corpus-${p.v}`}
                  variant={corpus === p.v ? "default" : "outline"}
                  onClick={() => setCorpus(p.v)}
                >
                  {p.label}
                </Button>
              ))}
            </div>
          </div>
          <div>
            <div className="text-sm font-medium mb-1">Query volume</div>
            <div className="flex flex-wrap gap-2">
              {VOLUME_PRESETS.map((p) => (
                <Button
                  key={p.v}
                  data-testid={`volume-${p.v}`}
                  variant={volume === p.v ? "default" : "outline"}
                  onClick={() => setVolume(p.v)}
                >
                  {p.label}
                </Button>
              ))}
            </div>
          </div>
        </CardBody>
      </Card>

      {err && <p className="text-red-400 text-sm">{err}</p>}

      {data && (
        <>
          {data.measured?.measured && (
            <Card className="mb-6 border-emerald-500/40 bg-emerald-500/10" data-testid="measured">
              <CardHeader>
                <h2 className="font-medium">Measured live on this deployment</h2>
                <p className="text-xs text-slate-400">
                  Real per-query numbers from {data.measured.n}{" "}
                  {data.measured.n === 1 ? "query" : "queries"} on <b>{data.measured.model}</b>
                  {data.measured.instance ? ` (${data.measured.instance})` : ""} — latency and tokens
                  collected at run time. $ = measured TTFT + a standard 150-token answer at the measured decode rate, × this
                  instance&apos;s real on-demand $/hr ÷ an assumed 50% multiplexed utilisation.
                </p>
              </CardHeader>
              <CardBody className="space-y-3">
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  <Stat label="Cartridge latency" value={ms(data.measured.cart?.latency_ms)} />
                  <Stat label="RAG latency" value={ms(data.measured.rag?.latency_ms)} />
                  <Stat label="Faster than RAG" value={xstr(data.measured.savings?.faster_than_rag_x)} />
                  <Stat
                    label="Cartridge prefill / query"
                    value={tok(data.measured.cart?.prompt_tokens)}
                    sub={`${(data.measured.cart?.resident_kv_tokens ?? 0).toLocaleString()} resident KV (read once)`}
                  />
                  <Stat
                    label="RAG prefill / query"
                    value={tok(data.measured.rag?.prompt_tokens)}
                    sub="re-prefilled every query"
                  />
                  <Stat label="Fewer prefill tok" value={xstr(data.measured.savings?.fewer_prefill_tokens_x)} />
                  <Stat label="Cartridge / 1k queries" value={per1k(data.measured.cart?.cost_per_query)} />
                  <Stat label="RAG / 1k queries" value={per1k(data.measured.rag?.cost_per_query)} />
                  <Stat label="Cheaper than RAG" value={xstr(data.measured.savings?.cheaper_than_rag_x)} />
                </div>
              </CardBody>
            </Card>
          )}

          <Card className="mb-6 border-emerald-500/30 bg-emerald-500/10">
            <CardBody>
              <div className="text-sm text-slate-300">
                Modeled projection — at this corpus + volume, cartridges are
              </div>
              <div className="text-2xl font-semibold mt-1" data-testid="headline">
                {data.savings.vs_rag_x && data.savings.vs_rag_x > 1
                  ? `${data.savings.vs_rag_x}× cheaper than RAG`
                  : "comparable to RAG — the edge grows with corpus size & reuse"}
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHeader>
              <h2 className="font-medium">Cost per 1,000 queries</h2>
            </CardHeader>
            <CardBody className="space-y-4">
              {data.strategies.map((s: any) => (
                <div key={s.key} data-testid={`row-${s.key}`}>
                  <div className="flex items-center justify-between text-sm mb-1">
                    <span className="font-medium">
                      {s.label}{" "}
                      {s.key === "cartridge" && <Badge color="green">This platform</Badge>}
                      {!s.feasible && <Badge color="red">Not possible</Badge>}
                    </span>
                    <span className="tabular-nums">
                      <b data-testid={`perq-${s.key}`}>{per1k(s.per_query)}</b>
                      <span className="text-slate-400"> / 1k q</span>
                      <span className="text-slate-400"> · {money(s.per_month)}/mo</span>
                    </span>
                  </div>
                  <div className="h-3 rounded bg-slate-800">
                    <div
                      className={`h-3 rounded ${COLOR[s.key]}`}
                      style={{ width: s.per_query != null ? `${Math.max(1, (s.per_query / maxPerQuery) * 100)}%` : "0%" }}
                    />
                  </div>
                  <div className="text-xs text-slate-400 mt-1">{s.quality} — {s.note}</div>
                </div>
              ))}
            </CardBody>
          </Card>

          <p className="text-xs text-slate-400 mt-4">
            {data.measured?.measured
              ? "The card above is measured live on this deployment (latency + prefill + $/query from real queries). "
              : "Run a query in a corpus to populate the measured card above. "}
            The projection here scales those economics to the chosen corpus size &amp; volume at
            representative mid-2026 prices (model in <code>backend/app/metrics.py</code>). Cartridge cost
            is the multiplexed marginal — the moat is utilisation, not the token spread.
          </p>
        </>
      )}
    </main>
  );
}
