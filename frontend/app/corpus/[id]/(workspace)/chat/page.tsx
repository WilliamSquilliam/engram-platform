"use client";
// Compare section: ask one question, answer it two ways side by side — Engram Smart CAG (the adaptive
// cartridge router) vs RAG (the baseline) — showing the answers and the sources each used. Metrics and
// cost live on the Scale Test tab; this tab is purely for reading and comparing the answers.
import { useState } from "react";
import { useParams } from "next/navigation";
import { api } from "@/lib/api";
import { Button, Card, CardBody, Input } from "@/components/ui";

const ACCENT: Record<string, string> = {
  everyday: "border-blue-500/40 bg-blue-500/5",
  rag: "border-amber-500/30",
};

export default function ComparePage() {
  const { id } = useParams() as { id: string };
  const [question, setQuestion] = useState("");
  const [data, setData] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  // One streamed side: appends {delta} tokens into the strategy card as the model writes, and fills
  // the Sources list on {head}. Returns the head event's used_docs so the RAG side can reuse the cart
  // side's retrieval — one retrieval round-trip per question, identical evidence on both sides.
  async function streamSide(q: string, key: "everyday" | "rag", side: "cart" | "rag",
                            docIds?: string[]): Promise<string[]> {
    const label = key === "everyday" ? "Engram Smart CAG" : "RAG";
    let usedDocs: string[] = [];
    const upsert = (patch: any) =>
      setData((prev: any) => {
        const strategies = [...(prev?.strategies || [])];
        const i = strategies.findIndex((x: any) => x.key === key);
        const cur = i >= 0 ? strategies[i] : { key, label, answer: "" };
        const next = { ...cur, ...(typeof patch === "function" ? patch(cur) : patch) };
        if (i >= 0) strategies[i] = next; else strategies.push(next);
        return { ...(prev || {}), strategies };
      });
    upsert({ answer: "" });
    await api.compareStream(id, q, 3, side, (e: any) => {
      if (e.head) { usedDocs = e.used_docs || []; upsert({ sources: e.sources }); }
      else if (e.delta) upsert((cur: any) => ({ answer: (cur.answer || "") + e.delta }));
      else if (e.error) upsert((cur: any) => ({ answer: (cur.answer || "") + `\n(${e.error})` }));
    }, docIds);
    return usedDocs;
  }

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim() || busy) return;
    setBusy(true);
    setErr("");
    const q = question.trim();
    setData({ strategies: [] });
    try {
      // Smart CAG streams in first; RAG streams after, reusing the cart side's retrieved doc_ids.
      const docs = await streamSide(q, "everyday", "cart");
      await streamSide(q, "rag", "rag", docs);
    } catch (e: any) {
      setErr(e.message || "compare failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardBody>
          <form onSubmit={ask} className="flex gap-2">
            <Input
              data-testid="compare-input"
              value={question}
              onChange={(e: any) => setQuestion(e.target.value)}
              placeholder="Ask a question — answered two ways…"
              disabled={busy}
            />
            <Button data-testid="compare-send" type="submit" disabled={busy || !question.trim()}>
              {busy ? "Running…" : "Compare"}
            </Button>
          </form>
          {err && <p className="mt-2 text-sm text-red-400">{err}</p>}
        </CardBody>
      </Card>

      {busy && (
        <p className="text-sm text-slate-400">
          Answering with Engram Smart CAG and RAG on the same model, over the same retrieved documents…
        </p>
      )}

      {data && (
        <div className="grid md:grid-cols-2 gap-4" data-testid="compare-grid">
          {busy && !data.strategies.some((x: any) => x.key === "rag") && (
            <Card className="order-2 border-amber-500/30">
              <CardBody className="space-y-3">
                <span className="font-medium text-sm">RAG</span>
                <div className="min-h-[5rem] rounded-md bg-slate-800/50 p-2 text-sm text-slate-400">
                  Generating the RAG baseline…
                </div>
              </CardBody>
            </Card>
          )}
          {data.strategies.map((st: any) => (
            <Card key={st.key} className={ACCENT[st.key] || ""}>
              <CardBody className="space-y-3">
                <span className="font-medium text-sm">{st.label}</span>

                <div
                  data-testid={`answer-${st.key}`}
                  className="min-h-[5rem] whitespace-pre-wrap rounded-md bg-slate-800/50 p-2 text-sm text-slate-200"
                >
                  {st.answer || "—"}
                </div>

                {st.sources?.length > 0 && (
                  <div className="text-xs text-slate-300" data-testid={`sources-${st.key}`}>
                    <span className="text-slate-400">Sources</span>
                    <ul className="mt-1 space-y-0.5">
                      {st.sources.map((src: any) => (
                        <li key={src.id} className="truncate">
                          <span className="text-slate-200">{src.title}</span>{" "}
                          <span className="text-slate-500">({src.id})</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </CardBody>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
