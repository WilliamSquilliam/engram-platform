"use client";
// Resumable onboarding wizard — the 5-step flow over the onboarding endpoints:
//   1 Name       rename the corpus (created in /corpus/new)
//   2 Documents  upload files; Google Drive / SharePoint render as disabled "coming soon"
//   3 Model      GET /models — tiers; unavailable tiers render but aren't selectable ("coming soon")
//   4 Review     GET /corpora/{id}/estimate — doc count, file types, size, est. time + cost
//   5 Onboard    POST /corpora/{id}/onboard — dispatches server-side; handles the 409 no_serving_engine
//                gate ("starts once a model is enabled") and shows live progress when it runs
//
// Resumable: entering a not-ready corpus loads GET /corpora/{id}/onboarding and jumps to its
// onboarding_step. Each step PATCHes the cursor so a reload / exit resumes at the right place. The
// step cursor (where the USER is) is distinct from corpus.status (where the WORK is).
import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import { api, getToken } from "@/lib/api";
import { useTrainingJob, fmtClock } from "@/lib/useTrainingJob";
import { Badge, Button, Card, CardBody, CardHeader, Input, Stepper, cn } from "@/components/ui";
import type {
  Connector,
  Document,
  ModelTier,
  OnboardEstimate,
  OnboardingStep,
  ParseStatus,
} from "@/lib/types";

// Accepted upload types (shared contract with the backend's parsers).
const ACCEPTED_EXTS = [".txt", ".md", ".pdf", ".docx", ".html", ".htm"];
const DOC_RE = /\.(txt|md|pdf|docx|html?)$/i;
// The dropzone `accept` map (MIME -> extensions) react-dropzone filters against.
const DROPZONE_ACCEPT: Record<string, string[]> = {
  "text/plain": [".txt"],
  "text/markdown": [".md"],
  "application/pdf": [".pdf"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
  "text/html": [".html", ".htm"],
};
// The wizard's user-facing steps (the backend also has a terminal "ready").
const STEP_LABELS = ["Name", "Documents", "Model", "Review", "Onboard"];
const STEP_ORDER: OnboardingStep[] = ["name", "documents", "model", "review", "onboarding"];
const stepIndex = (s: OnboardingStep) => Math.max(0, STEP_ORDER.indexOf(s === "ready" ? "onboarding" : s));

// Poll the document list while any file is still parsing so rows transition
// parsing -> parsed/failed without a manual refresh.
const PARSE_POLL_MS = 2000;

// Per-document parse-status badge: label + the Badge palette color to use.
const PARSE_BADGE: Record<ParseStatus, { label: string; color: string }> = {
  pending: { label: "Queued", color: "slate" },
  parsing: { label: "Parsing…", color: "amber" },
  parsed: { label: "Ready", color: "green" },
  failed: { label: "Failed", color: "red" },
};

const fmtBytes = (n: number) => {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
};

export default function OnboardingWizard() {
  const { id } = useParams() as { id: string };
  const router = useRouter();

  const [corpus, setCorpus] = useState<any>(null);
  const [step, setStep] = useState<OnboardingStep>("name");
  const [docs, setDocs] = useState<Document[]>([]);
  const [connectors, setConnectors] = useState<Connector[]>([]);
  const [tiers, setTiers] = useState<ModelTier[]>([]);
  const [defaultTier, setDefaultTier] = useState<string>("");
  const [selectedTier, setSelectedTier] = useState<string | null>(null);
  const [estimate, setEstimate] = useState<OnboardEstimate | null>(null);
  const [name, setName] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [gated, setGated] = useState(false); // 409 no_serving_engine — onboarding is queued for a model
  const folderRef = useRef<HTMLInputElement>(null);

  const train = useTrainingJob(id, (job) => {
    // Onboarding finished (or stopped). Success -> the corpus is ready; go to chat.
    if (job.status === "succeeded") {
      router.replace(`/corpus/${id}/chat`);
    } else if (job.status === "failed") {
      setNote("Onboarding failed: " + (job.detail || "unknown error"));
      goto("review");
    } else if (job.status === "canceled") {
      setNote("Onboarding canceled.");
      goto("review");
    }
  });

  // Load the resumable snapshot + the model registry, and jump to the persisted step.
  const load = useCallback(async () => {
    const [c, st, mt, cx] = await Promise.all([
      api.getCorpus(id),
      api.getOnboarding(id),
      api.modelTiers(),
      // Connectors are a static registry; a failure shouldn't block the wizard.
      api.connectors().catch(() => ({ connectors: [] })),
    ]);
    setCorpus(c);
    setName(c.name);
    setDocs(st.documents || []);
    setConnectors(cx.connectors || []);
    setTiers(mt.tiers);
    setDefaultTier(mt.default_tier);
    setSelectedTier(st.model_tier ?? c.model_tier ?? null);
    return { c, st };
  }, [id]);

  useEffect(() => {
    if (!getToken()) {
      router.push("/login");
      return;
    }
    (async () => {
      try {
        const { c, st } = await load();
        if (c.status === "ready") {
          router.replace(`/corpus/${id}/chat`);
          return;
        }
        if (c.status === "training") {
          // A run is in flight — resume the live progress view on the Onboard step.
          setStep("onboarding");
          train.resume();
        } else {
          setStep(st.onboarding_step);
        }
      } catch {
        router.push("/login");
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  // Persist the cursor and move to a step.
  const goto = useCallback(
    async (s: OnboardingStep) => {
      setStep(s);
      setNote("");
      try {
        await api.patchOnboarding(id, { onboarding_step: s });
      } catch {
        /* cursor is best-effort; the step still advances locally */
      }
    },
    [id]
  );

  // ---- Step 1: name ----
  async function saveName() {
    // Rename isn't exposed by the API surface here; the name was set at creation. We only advance
    // the cursor. (Kept as a step so the flow reads 1..5 and resumes cleanly.)
    await goto("documents");
  }

  // ---- Step 2: documents ----
  async function handleFiles(files: File[]) {
    const supported = files.filter((f) => DOC_RE.test((f as any).path || f.name));
    const skipped = files.length - supported.length;
    if (!supported.length) {
      setNote(`No supported files found${skipped ? ` (skipped ${skipped})` : ""}. Accepts ${ACCEPTED_EXTS.join(" ")}.`);
      return;
    }
    setUploading(true);
    setNote("");
    try {
      await api.uploadDocuments(id, supported);
      setDocs(await api.listDocuments(id));
      setNote(`Added ${supported.length} file${supported.length === 1 ? "" : "s"}${skipped ? `, skipped ${skipped} unsupported` : ""}.`);
    } catch (err: any) {
      setNote("Upload failed: " + err.message);
    } finally {
      setUploading(false);
    }
  }

  // Refresh the doc list while any file is still pending/parsing so its badge
  // transitions to Ready/Failed on its own. Stops once every file has settled.
  const parsing = docs.some((d) => d.parse_status === "pending" || d.parse_status === "parsing");
  useEffect(() => {
    if (step !== "documents" || !parsing) return;
    const iv = setInterval(async () => {
      try {
        setDocs(await api.listDocuments(id));
      } catch {
        /* transient — keep polling */
      }
    }, PARSE_POLL_MS);
    return () => clearInterval(iv);
  }, [id, step, parsing]);

  const { getRootProps, getInputProps, open, isDragActive } = useDropzone({
    noClick: true,
    noKeyboard: true,
    multiple: true,
    accept: DROPZONE_ACCEPT,
    onDrop: (accepted) => handleFiles(accepted),
  });

  // ---- Step 3: model ----
  async function chooseTier(tierId: string) {
    setSelectedTier(tierId);
    try {
      await api.patchOnboarding(id, { model_tier: tierId });
    } catch (e: any) {
      setNote(e.message || "Could not select model");
    }
  }

  // ---- Step 4: review ----
  async function loadReview() {
    await goto("review");
    try {
      setEstimate(await api.estimate(id));
    } catch (e: any) {
      setNote(e.message || "Could not load estimate");
    }
  }

  // ---- Step 5: onboard ----
  async function startOnboard() {
    setBusy(true);
    setNote("");
    setGated(false);
    try {
      const res = await api.onboard(id);
      if ("no_serving_engine" in res) {
        // 409 gate: no live model yet. Stay on the Onboard step showing the queued state; the cursor
        // is left at "review" server-side (nothing dispatched).
        setGated(true);
        setStep("onboarding");
      } else {
        // Dispatched: a training run is now in flight. Show live progress.
        setStep("onboarding");
        setCorpus((c: any) => ({ ...(c || {}), status: "training" }));
        train.resume();
      }
    } catch (e: any) {
      setNote(e.message || "Could not start onboarding");
    } finally {
      setBusy(false);
    }
  }

  if (!corpus) return <main className="p-8 text-slate-400">Loading…</main>;

  const idx = stepIndex(step);
  const selectedTierObj = tiers.find((t) => t.id === selectedTier) || null;
  const canOnboard = docs.length > 0 && !!selectedTier;

  return (
    <main className="mx-auto max-w-2xl p-8">
      <button
        onClick={() => router.push("/")}
        className="text-sm text-slate-400 hover:text-slate-200"
        data-testid="setup-exit"
      >
        ← All Corpora
      </button>
      <h1 className="mt-2 mb-4 text-2xl font-semibold" data-testid="setup-title">{corpus.name}</h1>
      <div className="mb-6">
        <Stepper steps={STEP_LABELS} current={idx} />
      </div>

      {/* ---------------- Step 1: Name ---------------- */}
      {step === "name" && (
        <Card>
          <CardHeader>
            <h2 className="font-medium">Name your corpus</h2>
            <p className="text-xs text-slate-400">One knowledge base — e.g. a handbook or support KB.</p>
          </CardHeader>
          <CardBody className="space-y-4">
            <Input data-testid="wizard-name" value={name} onChange={(e: any) => setName(e.target.value)} />
            <div className="flex justify-end">
              <Button onClick={saveName} data-testid="step-name-next">Continue</Button>
            </div>
          </CardBody>
        </Card>
      )}

      {/* ---------------- Step 2: Documents ---------------- */}
      {step === "documents" && (
        <Card>
          <CardHeader className="flex items-center justify-between">
            <h2 className="font-medium">Add documents</h2>
            <span className="text-xs text-slate-400">{docs.length} file(s)</span>
          </CardHeader>
          <CardBody className="space-y-4">
            <div
              {...getRootProps()}
              data-testid="dropzone"
              className={cn(
                "rounded-xl border-2 border-dashed px-6 py-7 text-center transition",
                isDragActive ? "border-slate-400 bg-slate-800/50" : "border-slate-700 bg-slate-800/20"
              )}
            >
              <input {...getInputProps()} />
              <input
                ref={folderRef}
                type="file"
                multiple
                hidden
                onChange={(e) => e.target.files && handleFiles(Array.from(e.target.files))}
                {...({ webkitdirectory: "", directory: "" } as any)}
              />
              <p className="text-sm text-slate-300">{isDragActive ? "Drop to add" : "Drag files & folders here"}</p>
              <div className="mt-3">
                <Button type="button" data-testid="select-files" onClick={open} disabled={uploading}>
                  {uploading ? "Uploading…" : "Select files"}
                </Button>
              </div>
              <p className="mt-2 text-xs text-slate-500">
                Files or folders ·{" "}
                <button
                  type="button"
                  data-testid="select-folder"
                  className="underline hover:text-slate-300"
                  onClick={() => folderRef.current?.click()}
                  disabled={uploading}
                >
                  Select a folder
                </button>{" "}
                · {ACCEPTED_EXTS.join(" ")}
              </p>
              {note && <p data-testid="upload-note" className="mt-2 text-xs text-slate-400">{note}</p>}
            </div>

            {/* Document-source connectors. filesystem is the upload above; external
                connectors render as Connect buttons, disabled + "coming soon" until
                available. Same styling as the unavailable model tiers for consistency. */}
            {connectors.filter((c) => c.id !== "filesystem").length > 0 && (
              <div className="grid grid-cols-2 gap-2" data-testid="connectors">
                {connectors
                  .filter((c) => c.id !== "filesystem")
                  .map((c) => (
                    <button
                      key={c.id}
                      type="button"
                      disabled={!c.available}
                      data-testid={`connector-${c.id}`}
                      title={c.available ? c.description : `${c.description} — coming soon`}
                      className={cn(
                        "flex items-center justify-between gap-2 rounded-lg border px-3 py-2 text-sm transition",
                        c.available
                          ? "border-slate-700 bg-slate-900 text-slate-200 hover:border-slate-600"
                          : "cursor-not-allowed border-slate-800 bg-slate-900/50 text-slate-400 opacity-60"
                      )}
                    >
                      <span>Connect {c.label}</span>
                      {!c.available && (
                        <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-500">
                          Coming soon
                        </span>
                      )}
                    </button>
                  ))}
              </div>
            )}

            {docs.length > 0 && (
              <ul data-testid="doc-list" className="max-h-48 divide-y divide-slate-800 overflow-y-auto text-sm">
                {docs.map((d) => {
                  const badge = d.parse_status ? PARSE_BADGE[d.parse_status] : null;
                  return (
                    <li key={d.id} className="py-1.5" data-testid={`doc-row-${d.id}`}>
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate" title={d.filename}>{d.filename}</span>
                        <div className="flex shrink-0 items-center gap-2">
                          {badge && (
                            <span data-testid={`doc-status-${d.id}`}>
                              <Badge color={badge.color}>{badge.label}</Badge>
                            </span>
                          )}
                          <span className="text-slate-500">{fmtBytes(d.size)}</span>
                        </div>
                      </div>
                      {d.parse_status === "failed" && d.parse_error && (
                        <p data-testid={`doc-error-${d.id}`} className="mt-0.5 truncate text-xs text-red-400" title={d.parse_error}>
                          {d.parse_error}
                        </p>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}

            <div className="flex justify-between">
              <Button variant="outline" onClick={() => goto("name")} data-testid="step-back">Back</Button>
              <Button onClick={() => goto("model")} disabled={docs.length === 0} data-testid="step-docs-next">
                Continue
              </Button>
            </div>
          </CardBody>
        </Card>
      )}

      {/* ---------------- Step 3: Model ---------------- */}
      {step === "model" && (
        <Card>
          <CardHeader>
            <h2 className="font-medium">Choose a model</h2>
            <p className="text-xs text-slate-400">Tiers marked “coming soon” are not selectable yet.</p>
          </CardHeader>
          <CardBody className="space-y-4">
            <div className="space-y-2" data-testid="tier-list">
              {tiers.map((t) => {
                const selected = selectedTier === t.id;
                return (
                  <button
                    key={t.id}
                    type="button"
                    disabled={!t.available}
                    onClick={() => t.available && chooseTier(t.id)}
                    data-testid={`tier-${t.id}`}
                    className={cn(
                      "flex w-full items-center justify-between rounded-lg border px-4 py-3 text-left transition",
                      selected ? "border-emerald-400 bg-emerald-500/5" : "border-slate-800 bg-slate-900",
                      t.available ? "hover:border-slate-700" : "cursor-not-allowed opacity-60"
                    )}
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-slate-100">{t.label}</span>
                        {t.id === defaultTier && t.available && (
                          <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-400">Default</span>
                        )}
                        {!t.available && (
                          <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-500">Coming soon</span>
                        )}
                      </div>
                      <p className="mt-0.5 truncate text-xs text-slate-400">
                        {t.description}
                        {t.precision ? ` · ${t.precision}` : ""}
                        {t.context_tokens ? ` · ${t.context_tokens.toLocaleString()} ctx` : ""}
                      </p>
                    </div>
                    <span
                      className={cn(
                        "ml-3 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border",
                        selected ? "border-emerald-400 bg-emerald-400 text-slate-950" : "border-slate-700"
                      )}
                    >
                      {selected ? "✓" : ""}
                    </span>
                  </button>
                );
              })}
            </div>
            {note && <p className="text-sm text-red-400">{note}</p>}
            <div className="flex justify-between">
              <Button variant="outline" onClick={() => goto("documents")} data-testid="step-back">Back</Button>
              <Button onClick={loadReview} disabled={!selectedTier} data-testid="step-model-next">Continue</Button>
            </div>
          </CardBody>
        </Card>
      )}

      {/* ---------------- Step 4: Review ---------------- */}
      {step === "review" && (
        <Card>
          <CardHeader>
            <h2 className="font-medium">Review</h2>
            <p className="text-xs text-slate-400">A rough pre-run estimate. Real figures land after onboarding.</p>
          </CardHeader>
          <CardBody className="space-y-4">
            <dl className="grid grid-cols-2 gap-3 text-sm" data-testid="review-summary">
              <Stat label="Documents" value={String(estimate?.n_documents ?? docs.length)} />
              <Stat label="Total size" value={fmtBytes(estimate?.total_bytes ?? docs.reduce((a, d) => a + d.size, 0))} />
              <Stat label="Model" value={selectedTierObj?.label ?? selectedTier ?? "—"} />
              <Stat
                label="File types"
                value={
                  estimate
                    ? Object.entries(estimate.file_types).map(([k, v]) => `${k} ×${v}`).join(", ") || "—"
                    : "—"
                }
              />
              <Stat label="Est. onboarding time" value={estimate ? `${fmtClock(estimate.est_seconds)}` : "—"} />
              <Stat label="Est. cost" value={estimate ? `$${estimate.est_cost_ondemand.toFixed(2)}` : "—"} />
            </dl>
            {note && <p className="text-sm text-red-400">{note}</p>}
            <div className="flex justify-between">
              <Button variant="outline" onClick={() => goto("model")} data-testid="step-back">Back</Button>
              <Button onClick={startOnboard} disabled={!canOnboard || busy} data-testid="step-onboard">
                {busy ? "Starting…" : "Start onboarding"}
              </Button>
            </div>
          </CardBody>
        </Card>
      )}

      {/* ---------------- Step 5: Onboard ---------------- */}
      {step === "onboarding" && (
        <>
          {gated ? (
            // 409 gate: no serving engine enabled yet.
            <Card className="border-amber-500/30 bg-amber-500/5">
              <CardBody className="space-y-3">
                <div className="text-lg font-semibold" data-testid="onboard-gated">Queued for a model</div>
                <p className="text-sm text-slate-300">
                  Your documents are ready to onboard. Onboarding starts once a model is enabled for your
                  account. We will pick this up automatically — nothing else to do here.
                </p>
                <div className="flex gap-2">
                  <Button variant="outline" onClick={() => goto("review")} data-testid="gated-back">Back to review</Button>
                  <Button variant="outline" onClick={() => router.push("/")}>All Corpora</Button>
                </div>
              </CardBody>
            </Card>
          ) : (
            // Live progress (a run is in flight).
            <Card>
              <CardHeader><h2 className="font-medium">Onboarding</h2></CardHeader>
              <CardBody className="space-y-4">
                <div data-testid="onboard-progress" className="space-y-2">
                  <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-slate-100 transition-[width] duration-500 ease-out"
                      style={{ width: `${Math.max(4, train.pct)}%` }}
                    />
                  </div>
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span data-testid="onboard-phase">{train.phase || "Starting…"}</span>
                    <span className="tabular-nums">
                      <span data-testid="onboard-elapsed">{fmtClock(train.elapsed)}</span>
                      {" · "}
                      <span data-testid="onboard-pct">{train.pct}%</span>
                      {" · "}
                      <span>{train.eta != null ? `-${fmtClock(train.eta)}` : "--:--"}</span>
                    </span>
                  </div>
                </div>
                <p className="text-xs text-slate-500">
                  You can leave this page — onboarding keeps running and the corpus stays under
                  “Corpora” as <b>Training</b>. Click it there to return here.
                </p>
                <div className="flex justify-between">
                  <Button variant="danger" onClick={() => train.cancel()} data-testid="cancel-onboard">
                    Cancel
                  </Button>
                  <Button variant="outline" onClick={() => router.push("/")} data-testid="onboard-exit">Exit</Button>
                </div>
              </CardBody>
            </Card>
          )}
        </>
      )}
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2">
      <dt className="text-xs text-slate-500">{label}</dt>
      <dd className="mt-0.5 truncate font-medium text-slate-100" title={value}>{value}</dd>
    </div>
  );
}
