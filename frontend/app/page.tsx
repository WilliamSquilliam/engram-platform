"use client";
import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { api, getToken, clearToken, notifyCorporaChanged, CORPORA_CHANGED } from "@/lib/api";
import { Button, Card, CardBody, Badge, ProgressBar } from "@/components/ui";

const STATUS_COLOR: Record<string, string> = {
  new: "slate",
  training: "amber",
  ready: "green",
  failed: "red",
};

function hrefFor(c: any): string {
  return c.status === "ready" ? `/document-base/${c.id}/chat` : `/document-base/${c.id}/setup`;
}

export default function Dashboard() {
  const router = useRouter();
  const [corpora, setCorpora] = useState<any[]>([]);
  const pollRef = useRef<any>(null);

  async function load() {
    try {
      setCorpora(await api.listCorpora());
    } catch {
      clearToken();
      router.push("/login");
    }
  }

  useEffect(() => {
    if (!getToken()) {
      router.push("/login");
      return;
    }
    load();
    window.addEventListener(CORPORA_CHANGED, load);
    return () => {
      clearInterval(pollRef.current);
      window.removeEventListener(CORPORA_CHANGED, load);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    clearInterval(pollRef.current);
    if (corpora.some((c) => c.status === "training")) {
      pollRef.current = setInterval(async () => {
        try {
          setCorpora(await api.listCorpora());
        } catch {
          /* transient */
        }
      }, 4000);
    }
    return () => clearInterval(pollRef.current);
  }, [corpora]);

  return (
    <div className="mx-auto max-w-6xl p-8">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold">Your Document Bases</h1>
        <p className="text-sm text-slate-400">
          Each document base is a knowledge base compiled into composable KV cartridges.
        </p>
      </div>

      {corpora.length === 0 ? (
        <Card>
          <CardBody className="flex flex-col items-center gap-3 py-12 text-center">
            <p className="text-sm text-slate-400">No document bases yet.</p>
            <Link href="/document-base/new">
              <Button data-testid="empty-new-corpus">+ New Document Base</Button>
            </Link>
          </CardBody>
        </Card>
      ) : (
        <div data-testid="corpus-list" className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {corpora.map((c) => (
            <CorpusCard key={c.id} c={c} onDeleted={load} />
          ))}
        </div>
      )}
    </div>
  );
}

function CorpusCard({ c, onDeleted }: { c: any; onDeleted: () => void }) {
  const router = useRouter();
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);

  async function del() {
    setBusy(true);
    try {
      await api.deleteCorpus(c.id);
      notifyCorporaChanged();
      await onDeleted();
    } finally {
      setBusy(false);
      setConfirm(false);
    }
  }

  return (
    <Card className="flex flex-col transition hover:border-slate-600">
      <CardBody className="flex flex-1 flex-col gap-3">
        <div className="flex items-start justify-between gap-2">
          <button
            onClick={() => router.push(hrefFor(c))}
            data-testid="corpus-item-name"
            className="min-w-0 flex-1 text-left font-medium hover:underline"
          >
            <span className="truncate">{c.name}</span>
          </button>
          <Badge color={STATUS_COLOR[c.status] || "slate"}>
            <span className="capitalize">{c.status}</span>
          </Badge>
        </div>

        <div className="text-xs text-slate-400">
          {c.n_documents} document{c.n_documents === 1 ? "" : "s"} · {c.source_type}
          {c.n_cartridges ? ` · ${c.n_cartridges} cartridges` : ""}
        </div>

        {c.status === "training" && <ProgressBar indeterminate />}

        <div className="mt-auto flex items-center justify-between pt-2">
          <Button variant="outline" onClick={() => router.push(hrefFor(c))}>
            {c.status === "ready" ? "Open" : "Resume setup"}
          </Button>
          {confirm ? (
            <span className="flex items-center gap-2 text-xs">
              <button
                onClick={del}
                disabled={busy}
                data-testid="confirm-delete"
                className="font-medium text-red-600 hover:underline"
              >
                {busy ? "Deleting…" : "Confirm delete"}
              </button>
              <button onClick={() => setConfirm(false)} className="text-slate-400 hover:underline">
                Cancel
              </button>
            </span>
          ) : (
            <button
              onClick={() => setConfirm(true)}
              data-testid="delete-corpus"
              className="text-xs text-slate-400 hover:text-red-600"
            >
              Delete
            </button>
          )}
        </div>
      </CardBody>
    </Card>
  );
}
