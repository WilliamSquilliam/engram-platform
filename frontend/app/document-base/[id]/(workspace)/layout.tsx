"use client";
// Workspace shell for a READY corpus: a header (name · status · delete) with
// horizontal section tabs underneath. The global left sidebar (AppShell) handles
// corpus switching, so this is the per-corpus chrome only. Bounces to setup if the
// corpus isn't ready yet.
import { useEffect, useState } from "react";
import { useParams, usePathname, useRouter } from "next/navigation";
import Link from "next/link";
import { api, getToken, notifyCorporaChanged } from "@/lib/api";
import { Badge, cn } from "@/components/ui";

// Chat-only MVP: chat IS the product surface, so the workspace nav is just Chat + Documents.
// Compare / Scale Test / Costs are retired from the chrome (their pages stay routable for internal
// use, but nothing links to them). MCP Server likewise stays routable but out of the nav.
const NAV = [
  { seg: "chat", label: "Chat" },
  { seg: "documents", label: "Documents" },
];

export default function WorkspaceLayout({ children }: { children: React.ReactNode }) {
  const { id } = useParams() as { id: string };
  const router = useRouter();
  const pathname = usePathname() || "";
  const [corpus, setCorpus] = useState<any>(null);
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!getToken()) {
      router.push("/login");
      return;
    }
    (async () => {
      try {
        const c = await api.getCorpus(id);
        if (c.status !== "ready") {
          router.replace(`/corpus/${id}/setup`);
          return;
        }
        setCorpus(c);
      } catch {
        router.push("/login");
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  async function del() {
    setBusy(true);
    try {
      await api.deleteCorpus(id);
      notifyCorporaChanged();
      router.push("/");
    } catch {
      setBusy(false);
    }
  }

  if (!corpus) return <div className="p-8 text-slate-400">Loading…</div>;

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-slate-800 bg-slate-900 px-8 pt-5">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold">{corpus.name}</h1>
          <div className="flex items-center gap-3">
            <Badge color="green">
              <span data-testid="corpus-status" className="capitalize">{corpus.status}</span>
            </Badge>
            {confirm ? (
              <span className="flex items-center gap-2 text-xs">
                <button
                  onClick={del}
                  disabled={busy}
                  data-testid="confirm-delete"
                  className="font-medium text-red-400 hover:underline"
                >
                  {busy ? "Deleting…" : "Confirm Delete"}
                </button>
                <button onClick={() => setConfirm(false)} className="text-slate-400 hover:underline">
                  Cancel
                </button>
              </span>
            ) : (
              <button
                onClick={() => setConfirm(true)}
                data-testid="delete-corpus"
                className="text-xs text-slate-400 hover:text-red-400"
              >
                Delete
              </button>
            )}
          </div>
        </div>
        <nav className="mt-4 flex gap-1" data-testid="workspace-nav">
          {NAV.map((n) => {
            const active = pathname.endsWith(`/${n.seg}`);
            return (
              <Link
                key={n.seg}
                href={`/corpus/${id}/${n.seg}`}
                className={cn(
                  "border-b-2 px-4 py-2 text-sm font-medium transition",
                  active
                    ? "border-emerald-400 text-slate-100"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                )}
              >
                {n.label}
              </Link>
            );
          })}
        </nav>
      </div>
      {/* Chat manages its own full-height scroll + composer, so it gets a bare pane; other tabs
          keep the padded, scrollable container. */}
      {pathname.endsWith("/chat") ? (
        <div className="min-h-0 flex-1">{children}</div>
      ) : (
        <div className="flex-1 overflow-y-auto p-8">{children}</div>
      )}
    </div>
  );
}
