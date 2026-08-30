import Link from "next/link";
import { Card, CardBody } from "@/components/ui";

export const metadata = {
  title: "About — Engram Smart CAG",
  description: "Read once, infer many: a KV-cache platform that turns a document corpus into cheap, composable inference.",
};

const STEPS = [
  {
    n: "1",
    title: "Read once",
    body: "Point Engram Smart CAG at a corpus — upload files or whole folders. We process it a single time, offline.",
  },
  {
    n: "2",
    title: "Train cartridges",
    body: "Each document is compressed into a small, trainable KV-cache “cartridge” (5–100× smaller than its text) that a frozen base model can attend to.",
  },
  {
    n: "3",
    title: "Infer many",
    body: "At query time we load just the relevant cartridges. You pay the heavy document-reading cost once, then amortize it over every future question.",
  },
];

const DIFFS = [
  {
    title: "vs. RAG",
    body: "RAG re-reads the retrieved chunks through the model on every query. Engram Smart CAG retrieves the cartridge, not the text — the same answer at a fraction of the prompt tokens, and the adaptive router skips the documents entirely when the cartridge is already confident.",
  },
  {
    title: "vs. fine-tuning",
    body: "Fine-tuning bakes data into the weights and breaks multi-tenancy. Cartridges are per-document, per-tenant artifacts that compose with one shared frozen model.",
  },
];

export default function AboutPage() {
  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      {/* bare top bar */}
      <header className="mx-auto flex max-w-5xl items-center justify-between px-6 py-5">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <img src="/logo.svg" alt="Engram Smart CAG" width={28} height={28} className="h-7 w-7" />
          <span>Engram</span>
        </Link>
        <nav className="flex items-center gap-5 text-sm text-slate-400">
          <Link href="/demo" className="hover:text-slate-100">Cost demo</Link>
          <Link href="/login" className="hover:text-slate-100">Sign in</Link>
        </nav>
      </header>

      {/* hero */}
      <section className="mx-auto max-w-3xl px-6 pb-10 pt-12 text-center">
        <p className="text-sm font-medium uppercase tracking-widest text-emerald-400">Read once · infer many</p>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">
          Spend GPU once.<br className="hidden sm:block" /> Make every future query cheap.
        </h1>
        <p className="mx-auto mt-5 max-w-2xl text-lg text-slate-400">
          Engram Smart CAG processes your document corpus a single time and turns each document into a compact,
          composable KV-cache “cartridge.” Your apps then query it for a fraction of the usual token cost —
          over chat or as an MCP server your own LLM can call.
        </p>
        <div className="mt-7 flex items-center justify-center gap-3">
          <Link
            href="/login"
            className="rounded-md bg-slate-100 px-5 py-2.5 text-sm font-medium text-slate-900 transition hover:bg-white"
          >
            Get started
          </Link>
          <Link
            href="/demo"
            className="rounded-md border border-slate-700 px-5 py-2.5 text-sm font-medium text-slate-200 transition hover:bg-slate-800"
          >
            See the cost comparison
          </Link>
        </div>
      </section>

      {/* how it works */}
      <section className="mx-auto max-w-5xl px-6 py-10">
        <h2 className="mb-6 text-center text-sm font-semibold uppercase tracking-widest text-slate-500">
          How it works
        </h2>
        <div className="grid gap-4 sm:grid-cols-3">
          {STEPS.map((s) => (
            <Card key={s.n}>
              <CardBody className="space-y-2">
                <span className="grid h-8 w-8 place-items-center rounded-full bg-emerald-500/15 text-sm font-semibold text-emerald-300">
                  {s.n}
                </span>
                <h3 className="font-medium text-slate-100">{s.title}</h3>
                <p className="text-sm text-slate-400">{s.body}</p>
              </CardBody>
            </Card>
          ))}
        </div>
      </section>

      {/* why it's different */}
      <section className="mx-auto max-w-5xl px-6 py-10">
        <h2 className="mb-6 text-center text-sm font-semibold uppercase tracking-widest text-slate-500">
          Why it&apos;s different
        </h2>
        <div className="grid gap-4 sm:grid-cols-3">
          {DIFFS.map((d) => (
            <Card key={d.title}>
              <CardBody className="space-y-2">
                <h3 className="font-medium text-slate-100">{d.title}</h3>
                <p className="text-sm text-slate-400">{d.body}</p>
              </CardBody>
            </Card>
          ))}
        </div>
      </section>

      {/* close */}
      <section className="mx-auto max-w-3xl px-6 py-12 text-center">
        <h2 className="text-2xl font-semibold">Built for corpora that change slowly and get queried a lot.</h2>
        <p className="mx-auto mt-3 max-w-xl text-slate-400">
          Knowledge bases, code, legal, clinical, financial filings — large, stable, and asked about
          thousands of times. That&apos;s where reading once and inferring many pays off.
        </p>
        <div className="mt-7">
          <Link
            href="/login"
            className="rounded-md bg-slate-100 px-5 py-2.5 text-sm font-medium text-slate-900 transition hover:bg-white"
          >
            Create your workspace
          </Link>
        </div>
      </section>

      <footer className="border-t border-slate-800 py-6 text-center text-xs text-slate-500">
        Engram Smart CAG — Cartridge KV Platform
      </footer>
    </main>
  );
}
