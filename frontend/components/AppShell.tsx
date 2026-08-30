"use client";
// Global app chrome (Gemini-style): a fixed top bar + a collapsible left sidebar
// that lists corpora and lets you create / switch / delete them. Wraps the whole
// app from the root layout; renders bare (no chrome) on /login and /demo.
import { useCallback, useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import Link from "next/link";
import { api, getToken, clearToken, notifyCorporaChanged, CORPORA_CHANGED } from "@/lib/api";
import { Button, cn } from "@/components/ui";

const BARE = ["/login", "/demo", "/about"];
const STATUS_DOT: Record<string, string> = {
  new: "bg-slate-300",
  training: "bg-amber-400 animate-pulse",
  ready: "bg-emerald-500",
  failed: "bg-red-400",
};

function hrefFor(c: any) {
  return c.status === "ready" ? `/corpus/${c.id}/chat` : `/corpus/${c.id}/setup`;
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname() || "/";
  const isBare = BARE.some((r) => pathname === r || pathname.startsWith(r + "/"));
  if (isBare) return <>{children}</>;
  return <Shell pathname={pathname}>{children}</Shell>;
}

function Shell({ pathname, children }: { pathname: string; children: React.ReactNode }) {
  const router = useRouter();
  const [corpora, setCorpora] = useState<any[]>([]);
  const [me, setMe] = useState<any>(null);
  const [open, setOpen] = useState(true);

  const refresh = useCallback(async () => {
    if (!getToken()) return;
    try {
      const [cs, m] = await Promise.all([api.listCorpora(), api.me()]);
      setCorpora(cs);
      setMe(m);
    } catch {
      /* not authed yet / transient */
    }
  }, []);

  // Refresh the sidebar on every navigation (covers create / train / delete).
  useEffect(() => {
    refresh();
  }, [pathname, refresh]);

  // Refresh when any view mutates the corpus list (e.g. delete from a card).
  useEffect(() => {
    window.addEventListener(CORPORA_CHANGED, refresh);
    return () => window.removeEventListener(CORPORA_CHANGED, refresh);
  }, [refresh]);

  // While anything is training, keep statuses live.
  useEffect(() => {
    if (!corpora.some((c) => c.status === "training")) return;
    const iv = setInterval(refresh, 4000);
    return () => clearInterval(iv);
  }, [corpora, refresh]);

  function logout() {
    clearToken();
    router.push("/login");
  }

  const activeId = pathname.startsWith("/corpus/") ? pathname.split("/")[2] : null;

  return (
    <div className="flex h-screen flex-col">
      {/* top bar */}
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-slate-800 bg-slate-900 px-3">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setOpen((o) => !o)}
            aria-label="Toggle sidebar"
            data-testid="sidebar-toggle"
            className="rounded-md p-2 text-slate-400 hover:bg-slate-800"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18M3 12h18M3 18h18" strokeLinecap="round" />
            </svg>
          </button>
          <Link href="/" className="flex items-center gap-2 font-semibold">
            <img src="/logo.svg" alt="Engram Smart CAG" width={28} height={28} className="h-7 w-7" />
            <span>Engram</span>
          </Link>
        </div>
        <div className="flex items-center gap-3 text-sm">
          {me && <span className="hidden text-slate-500 sm:block" data-testid="user-email">{me.email}</span>}
          <Button variant="outline" onClick={logout} data-testid="logout-btn">Sign out</Button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* sidebar */}
        {open && (
          <aside
            className="flex w-64 shrink-0 flex-col border-r border-slate-800 bg-slate-900"
            data-testid="sidebar"
          >
            <div className="p-3">
              <Link href="/corpus/new">
                <Button data-testid="new-corpus" className="w-full gap-2">
                  <span className="text-base leading-none">+</span> New Corpus
                </Button>
              </Link>
            </div>
            <div className="px-4 pb-1 text-xs font-medium uppercase tracking-wide text-slate-400">
              Corpora
            </div>
            <nav className="flex-1 space-y-0.5 overflow-y-auto px-2" data-testid="sidebar-corpora">
              {corpora.map((c) => (
                <SidebarItem key={c.id} c={c} active={c.id === activeId} onChange={refresh} />
              ))}
              {corpora.length === 0 && (
                <p className="px-3 py-2 text-sm text-slate-400">No corpora yet.</p>
              )}
            </nav>
            <div className="space-y-0.5 border-t border-slate-800 p-2">
              <Link
                href="/about"
                className="block rounded-md px-3 py-2 text-sm text-slate-300 hover:bg-slate-800"
                data-testid="about-link"
              >
                About
              </Link>
              <Link
                href="/demo"
                className="block rounded-md px-3 py-2 text-sm text-slate-300 hover:bg-slate-800"
                data-testid="demo-link"
              >
                Cost demo
              </Link>
            </div>
          </aside>
        )}

        {/* main content fills the rest of the screen */}
        <main className="min-w-0 flex-1 overflow-y-auto bg-slate-950">{children}</main>
      </div>
    </div>
  );
}

function SidebarItem({ c, active, onChange }: { c: any; active: boolean; onChange: () => void }) {
  const router = useRouter();
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);

  async function del(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    setBusy(true);
    try {
      await api.deleteCorpus(c.id);
      if (active) router.push("/");
      notifyCorporaChanged();
      await onChange();
    } finally {
      setBusy(false);
      setConfirm(false);
    }
  }

  return (
    <div
      className={cn(
        "group flex items-center rounded-md",
        active ? "bg-slate-800" : "hover:bg-slate-800/60"
      )}
    >
      <Link
        href={hrefFor(c)}
        data-testid="sidebar-item"
        className="flex min-w-0 flex-1 items-center gap-2 px-3 py-2 text-sm"
      >
        <span className={cn("h-2 w-2 shrink-0 rounded-full", STATUS_DOT[c.status] || "bg-slate-300")} />
        <span className="truncate" title={c.name}>{c.name}</span>
      </Link>
      {confirm ? (
        <span className="flex items-center gap-1 pr-2 text-xs">
          <button
            onClick={del}
            disabled={busy}
            data-testid="confirm-delete"
            className="font-medium text-red-600 hover:underline"
          >
            {busy ? "…" : "Delete"}
          </button>
          <button
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setConfirm(false);
            }}
            className="text-slate-400 hover:underline"
          >
            Cancel
          </button>
        </span>
      ) : (
        <button
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setConfirm(true);
          }}
          data-testid="delete-corpus"
          aria-label={`Delete ${c.name}`}
          className="mr-1 rounded p-1 text-slate-400 opacity-0 transition hover:text-red-600 focus:opacity-100 group-hover:opacity-100"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      )}
    </div>
  );
}
