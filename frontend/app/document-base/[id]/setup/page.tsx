"use client";
// Resumable onboarding wizard — the 5-step flow over the onboarding endpoints:
//   1 Name       rename the corpus (created in /document-base/new)
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
import { fmtBytes, PARSE_BADGE } from "@/lib/format";
import { Badge, Button, Card, CardBody, CardHeader, Input, Stepper, cn } from "@/components/ui";
import type {
  BrowseFolder,
  BrowseResult,
  Connector,
  ConnectorConnection,
  Document,
  ImportStatus,
  ModelTier,
  OnboardEstimate,
  OnboardingStep,
} from "@/lib/types";

// Accepted upload types (shared contract with the backend's parsers — see backend SUPPORTED_EXTS).
const ACCEPTED_EXTS = [".txt", ".md", ".pdf", ".docx", ".doc", ".html", ".xlsx", ".xls", ".csv"];
const DOC_RE = /\.(txt|md|pdf|docx?|html?|xlsx?|csv|tsv)$/i;
// The dropzone `accept` map (MIME -> extensions) react-dropzone filters against.
const DROPZONE_ACCEPT: Record<string, string[]> = {
  "text/plain": [".txt"],
  "text/markdown": [".md"],
  "application/pdf": [".pdf"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
  "application/msword": [".doc"],
  "text/html": [".html", ".htm"],
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
  "application/vnd.ms-excel": [".xls"],
  "text/csv": [".csv"],
  "text/tab-separated-values": [".tsv"],
};
// The wizard's user-facing steps (the backend also has a terminal "ready").
const STEP_LABELS = ["Name", "Documents", "Model", "Review", "Onboard"];
const STEP_ORDER: OnboardingStep[] = ["name", "documents", "model", "review", "onboarding"];
const stepIndex = (s: OnboardingStep) => Math.max(0, STEP_ORDER.indexOf(s === "ready" ? "onboarding" : s));

// Poll the document list while any file is still parsing so rows transition
// parsing -> parsed/failed without a manual refresh.
const PARSE_POLL_MS = 2000;
// Poll the folder-import status on this cadence while an import is running, so the inline progress
// line (added/skipped) ticks alongside the doc list's own parse-badge polling.
const IMPORT_POLL_MS = 3000;

// Human labels for the two external connectors, so copy ("Google Drive connected.") reads right even
// before the registry loads. Falls back to the registry label / a title-cased id for anything else.
const PROVIDER_LABEL: Record<string, string> = {
  google_drive: "Google Drive",
  sharepoint: "SharePoint",
};
const providerLabel = (id: string, connectors: Connector[]) =>
  connectors.find((c) => c.id === id)?.label || PROVIDER_LABEL[id] || id;
// Re-fetch the review estimate on this interval while the review step is open, so a fresh estimate
// and the serving_up gate (Start onboarding re-enables when the GPU comes back) stay current.
const REVIEW_POLL_MS = 15000;
// Shown on the review step when the GPU onboard plane is down: Start onboarding is disabled but
// nothing is lost, and it unlocks automatically once serving is back (the 15s poll flips the gate).
const SERVING_OFFLINE_NOTE =
  "The GPU is offline right now. Everything here is saved — Start onboarding unlocks the moment it is back.";

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

  // ---- External-source connectors (Google Drive / SharePoint) ----
  // A friendly note shown at the top of step 2 after returning from the provider consent screen
  // (success or failure). Separate from `note` so it survives the upload/import notes.
  const [connectNote, setConnectNote] = useState("");
  const [connectError, setConnectError] = useState(false);
  // Which provider is mid-connect (button spinner), and the folder-picker target once opened.
  const [connectingProvider, setConnectingProvider] = useState<string | null>(null);
  const [picker, setPicker] = useState<{ connectionId: string; provider: string } | null>(null);
  // Live folder-import status. Polled every 3s while running; the terminal line stays until the user
  // starts another import. null = never started this session (nothing to show).
  const [importStatus, setImportStatus] = useState<ImportStatus | null>(null);

  const train = useTrainingJob(id, (job) => {
    // Onboarding finished (or stopped). Success -> the corpus is ready; go to chat.
    if (job.status === "succeeded") {
      router.replace(`/document-base/${id}/chat`);
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
          router.replace(`/document-base/${id}/chat`);
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

  // Open the folder picker for the NEWEST connection of a provider (used after the OAuth return and
  // when a connection already exists). Returns false if none is found yet.
  const openPickerForProvider = useCallback(async (provider: string): Promise<boolean> => {
    const conns = await api.connectorConnections().catch(() => [] as ConnectorConnection[]);
    const mine = conns
      .filter((c) => c.provider === provider)
      .sort((a, b) => (a.created_at < b.created_at ? 1 : -1)); // newest first
    if (!mine.length) return false;
    setPicker({ connectionId: mine[0].id, provider });
    return true;
  }, []);

  // Connect button: if this workspace already linked this provider, jump straight to the folder
  // picker; otherwise start OAuth and full-page redirect to the provider's consent screen.
  async function connectProvider(provider: string) {
    setConnectingProvider(provider);
    setConnectNote("");
    setConnectError(false);
    try {
      if (await openPickerForProvider(provider)) return; // existing connection — skip OAuth
      const { url } = await api.connectorAuthorize(provider, id);
      window.location.assign(url); // leaves the app for the provider consent screen
    } catch (e: any) {
      setConnectError(true);
      setConnectNote(
        `We could not connect ${providerLabel(provider, connectors)}. Nothing was changed. ` +
          `Try again or use file upload.`
      );
    } finally {
      setConnectingProvider(null);
    }
  }

  // Handle the return from the provider consent screen. The provider redirects back to
  //   /document-base/{id}/setup?connected={provider}   (success)  or  ?connector_error={provider}.
  // On success we note it and AUTO-OPEN the folder picker for that provider's newest connection; on
  // failure we show a friendly inline note. Either way we scrub the query params from the URL (same
  // history.replaceState approach used for #token scrubbing) so a reload doesn't re-fire this.
  const oauthReturnHandled = useRef(false);
  useEffect(() => {
    if (oauthReturnHandled.current) return;
    const params = new URLSearchParams(window.location.search);
    const connected = params.get("connected");
    const errored = params.get("connector_error");
    if (!connected && !errored) return;
    oauthReturnHandled.current = true;
    // Scrub the connector params but keep any others (defensive) + the path.
    params.delete("connected");
    params.delete("connector_error");
    const qs = params.toString();
    window.history.replaceState(null, "", window.location.pathname + (qs ? `?${qs}` : ""));

    if (connected) {
      setConnectError(false);
      setConnectNote(`${providerLabel(connected, connectors)} connected.`);
      // Make sure the user is on the Documents step, then auto-open the picker for the fresh account.
      setStep("documents");
      openPickerForProvider(connected).then((found) => {
        if (!found) {
          // The connection row hasn't materialized yet; nudge the user rather than silently doing nothing.
          setConnectNote(
            `${providerLabel(connected, connectors)} connected. Click Connect ${providerLabel(
              connected,
              connectors
            )} to pick a folder.`
          );
        }
      });
    } else if (errored) {
      setConnectError(true);
      setConnectNote(
        `We could not connect ${providerLabel(errored, connectors)}. Nothing was changed. ` +
          `Try again or use file upload.`
      );
      setStep("documents");
    }
    // connectors may load a tick later; providerLabel falls back to a built-in label so copy is fine
    // even on first paint. Re-run once connectors arrive to upgrade any fallback label.
  }, [connectors, openPickerForProvider]);

  // Kick off a folder import from the picker, then poll import-status until it settles. The doc list's
  // own parse-badge polling (above) shows files as they land; this drives the compact progress line.
  const startImport = useCallback(
    async (connectionId: string, folder: BrowseFolder) => {
      const res = await api.corpusImport(id, {
        connection_id: connectionId,
        folder_id: folder.id,
        folder_name: folder.name,
      });
      setPicker(null);
      if ("already_running" in res) {
        setImportStatus({
          state: "running",
          imported: 0,
          skipped: 0,
          failed: 0,
          folder_name: folder.name,
          error: "An import is already running for this document base.",
        });
        return;
      }
      // Optimistic running state so the line appears immediately; the poll overwrites it.
      setImportStatus({ state: "running", imported: 0, skipped: 0, failed: 0, folder_name: folder.name, error: null });
    },
    [id]
  );

  // Poll import-status every 3s while an import is running. Refresh the doc list on each tick too so
  // imported files appear in the list without waiting for the separate parse poll to notice them.
  const importing = importStatus?.state === "running";
  useEffect(() => {
    if (step !== "documents" || !importing) return;
    const iv = setInterval(async () => {
      try {
        const [s] = await Promise.all([
          api.importStatus(id),
          api.listDocuments(id).then(setDocs).catch(() => {}),
        ]);
        setImportStatus(s);
      } catch {
        /* transient — keep polling */
      }
    }, IMPORT_POLL_MS);
    return () => clearInterval(iv);
  }, [id, step, importing]);

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
  // Continue-from-Model just moves the cursor; the estimate is fetched by the effect below, which
  // also covers every OTHER way of entering review (resume at the "review" step, or the training-job
  // callback bouncing back here on failure) — those used to leave the estimate null (all dashes).
  async function loadReview() {
    await goto("review");
  }

  // Fetch the estimate whenever the review step is active, and re-fetch on each entry (cheap, picks
  // up new uploads). While ON review, poll every 15s so the estimate stays fresh AND the serving_up
  // gate (Fix 2) lights the Start onboarding button back up the moment the GPU is reachable again.
  useEffect(() => {
    if (step !== "review") return;
    let alive = true;
    const refresh = async () => {
      try {
        const est = await api.estimate(id);
        if (alive) setEstimate(est);
      } catch (e: any) {
        if (alive) setNote(e.message || "Could not load estimate");
      }
    };
    refresh();
    const iv = setInterval(refresh, REVIEW_POLL_MS);
    return () => {
      alive = false;
      clearInterval(iv);
    };
  }, [id, step]);

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
      } else if ("serving_offline" in res) {
        // 503: the model is wired but the GPU plane went down between the poll and this click (a
        // race). Nothing dispatched — stay on review; the note + the 15s poll's gate handle re-entry.
        setNote(SERVING_OFFLINE_NOTE);
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
  // The GPU onboard plane is down (estimate says so): Start onboarding is gated until it's back. Only
  // treat an ARRIVED estimate as down — while it's still loading (null) don't flash the gate.
  const servingDown = estimate?.serving_up === false;

  return (
    <main className="mx-auto max-w-2xl p-8">
      <button
        onClick={() => router.push("/")}
        className="text-sm text-slate-400 hover:text-slate-200"
        data-testid="setup-exit"
      >
        ← All Document Bases
      </button>
      <h1 className="mt-2 mb-4 text-2xl font-semibold" data-testid="setup-title">{corpus.name}</h1>
      <div className="mb-6">
        <Stepper steps={STEP_LABELS} current={idx} />
      </div>

      {/* ---------------- Step 1: Name ---------------- */}
      {step === "name" && (
        <Card>
          <CardHeader>
            <h2 className="font-medium">Name your document base</h2>
            <p className="text-xs text-slate-400">A document base is also called a knowledge base, it is everything you want the AI to know.</p>
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

            {/* Document-source connectors. filesystem is the upload above; external connectors that
                are available render as enabled "Connect <label>" buttons that start OAuth (or jump
                straight to the folder picker when a connection already exists). Unavailable ones keep
                today's disabled "Coming soon" treatment. Connectors add to, never replace, upload. */}
            {connectors.filter((c) => c.id !== "filesystem").length > 0 && (
              <div className="grid grid-cols-2 gap-2" data-testid="connectors">
                {connectors
                  .filter((c) => c.id !== "filesystem")
                  .map((c) => {
                    const connecting = connectingProvider === c.id;
                    return (
                      <button
                        key={c.id}
                        type="button"
                        disabled={!c.available || connecting}
                        onClick={c.available ? () => connectProvider(c.id) : undefined}
                        data-testid={`connector-${c.id}`}
                        title={c.available ? c.description : `${c.description} — coming soon`}
                        className={cn(
                          "flex items-center justify-between gap-2 rounded-lg border px-3 py-2 text-sm transition",
                          c.available
                            ? "border-slate-700 bg-slate-900 text-slate-200 hover:border-slate-600"
                            : "cursor-not-allowed border-slate-800 bg-slate-900/50 text-slate-400 opacity-60"
                        )}
                      >
                        <span>{connecting ? `Connecting ${c.label}…` : `Connect ${c.label}`}</span>
                        {!c.available && (
                          <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-500">
                            Coming soon
                          </span>
                        )}
                      </button>
                    );
                  })}
              </div>
            )}

            {/* Return-from-OAuth note + a friendly failure line. Green on success, amber on failure —
                never red (a failed connect changed nothing). */}
            {connectNote && (
              <p
                data-testid="connect-note"
                className={cn("text-xs", connectError ? "text-amber-400" : "text-emerald-400")}
              >
                {connectNote}
              </p>
            )}

            {/* Compact import progress line, shown alongside the live doc list. */}
            <ImportProgressLine status={importStatus} />

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
            <p className="text-xs text-slate-400">
              Tiers marked “coming soon” aren’t live yet — you can still pick one, finish setup,
              and onboarding will start automatically once it’s enabled.
            </p>
          </CardHeader>
          <CardBody className="space-y-4">
            <div className="space-y-2" data-testid="tier-list">
              {tiers.map((t) => {
                const selected = selectedTier === t.id;
                return (
                  // Placeholder ("coming soon") tiers stay SELECTABLE: a tier choice is just a
                  // wizard selection — the serving gate lives at POST /onboard (409
                  // no_serving_engine -> the queued state). Disabling them dead-ended the whole
                  // wizard while no model was wired (caught in the e2e run-up).
                  <button
                    key={t.id}
                    type="button"
                    onClick={() => chooseTier(t.id)}
                    data-testid={`tier-${t.id}`}
                    className={cn(
                      "flex w-full items-center justify-between rounded-lg border px-4 py-3 text-left transition hover:border-slate-700",
                      selected ? "border-emerald-400 bg-emerald-500/5" : "border-slate-800 bg-slate-900",
                      !t.available && "opacity-75"
                    )}
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        {/* Prefer the public model name; when present the tier label moves into the
                            small line below (e.g. "Best tier · <description>"). No name -> today's UI. */}
                        <span className="font-medium text-slate-100">{t.display_name || t.label}</span>
                        {t.id === defaultTier && t.available && (
                          <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-400">Default</span>
                        )}
                        {!t.available && (
                          <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-500">Coming soon</span>
                        )}
                      </div>
                      <p className="mt-0.5 truncate text-xs text-slate-400">
                        {t.display_name ? `${t.label} tier · ${t.description}` : t.description}
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
              <Stat
                label="Model"
                value={selectedTierObj?.display_name || selectedTierObj?.label || selectedTier || "—"}
                sub={selectedTierObj?.description}
              />
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
            {servingDown && (
              <p data-testid="serving-offline-note" className="text-sm text-amber-400">{SERVING_OFFLINE_NOTE}</p>
            )}
            {note && <p className="text-sm text-red-400">{note}</p>}
            <div className="flex justify-between">
              <Button variant="outline" onClick={() => goto("model")} data-testid="step-back">Back</Button>
              <Button onClick={startOnboard} disabled={!canOnboard || busy || servingDown} data-testid="step-onboard">
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
                  <Button variant="outline" onClick={() => router.push("/")}>All Document Bases</Button>
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
                  You can leave this page — onboarding keeps running and the document base stays under
                  “Document Bases” as <b>Training</b>. Click it there to return here.
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

      {/* Folder picker for a linked connection. Rendered outside the step blocks so it can open from
          the OAuth return (auto-open) or a Connect click. Import starts from inside it. */}
      <ConnectorFolderPicker
        open={!!picker}
        connectionId={picker?.connectionId ?? null}
        providerName={picker ? providerLabel(picker.provider, connectors) : ""}
        onClose={() => setPicker(null)}
        onImport={(folder) => picker && startImport(picker.connectionId, folder)}
      />
    </main>
  );
}

// Compact import progress / terminal line for step 2, shown alongside the live doc list. Mirrors the
// import-status contract: running -> "N added, M skipped"; terminal states get their own honest copy.
// The "limited" line deliberately does NOT restate the global beta-limit modal's wording (which also
// fires) — it only says the import stopped and what is saved.
function ImportProgressLine({ status }: { status: ImportStatus | null }) {
  if (!status || status.state === "none") return null;
  const { state, imported, skipped, failed, folder_name, error } = status;

  if (state === "running") {
    // The 409 case reuses "running" with an error string carrying the already-running copy.
    if (error) {
      return (
        <p data-testid="import-line" className="text-xs text-amber-400">
          {error}
        </p>
      );
    }
    return (
      <p data-testid="import-line" className="flex items-center gap-2 text-xs text-slate-300">
        <span
          aria-hidden
          className="h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent"
        />
        Importing from {folder_name}: {imported} added
        {skipped ? `, ${skipped} skipped` : ""}
        {failed ? `, ${failed} failed` : ""}
      </p>
    );
  }
  if (state === "done") {
    return (
      <p data-testid="import-line" className="text-xs text-emerald-400">
        Import complete: {imported} added
        {skipped ? `, ${skipped} skipped` : ""}
        {failed ? `, ${failed} failed` : ""}.
      </p>
    );
  }
  if (state === "limited") {
    return (
      <p data-testid="import-line" className="text-xs text-amber-400">
        Import stopped at your beta document limit. Everything imported so far is saved.
      </p>
    );
  }
  // failed — show the error honestly.
  return (
    <p data-testid="import-line" className="text-xs text-red-400">
      Import failed{error ? `: ${error}` : "."}
    </p>
  );
}

// daisyUI modal folder picker. Browses a connection one level at a time (folders are opaque ids we
// just pass back), tracks a visited-folder stack for breadcrumb navigation, shows how many supported
// files sit in the current folder, and imports the current folder (with an in-modal confirm) — its
// supported files and every subfolder's are added to the document base.
function ConnectorFolderPicker({
  open,
  connectionId,
  providerName,
  onClose,
  onImport,
}: {
  open: boolean;
  connectionId: string | null;
  providerName: string;
  onClose: () => void;
  onImport: (folder: BrowseFolder) => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  // The path back to root: each entry is the folder we drilled INTO (id + name). Empty = top level.
  const [stack, setStack] = useState<BrowseFolder[]>([]);
  const [level, setLevel] = useState<BrowseResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  // Which folder the user asked to import — drives the in-modal confirm step.
  const [confirming, setConfirming] = useState<BrowseFolder | null>(null);
  const [importingNow, setImportingNow] = useState(false);

  // Keep the native <dialog> in sync (Escape + backdrop close), same as GpuConfirmDialog/BetaLimitNotice.
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  // Load a level of the tree. folderId omitted -> top level. Resets the confirm step on navigation.
  const loadLevel = useCallback(
    async (folderId?: string) => {
      if (!connectionId) return;
      setLoading(true);
      setError("");
      setConfirming(null);
      try {
        setLevel(await api.connectorBrowse(connectionId, folderId));
      } catch (e: any) {
        setError(e.message || "Could not load folders");
        setLevel(null);
      } finally {
        setLoading(false);
      }
    },
    [connectionId]
  );

  // On open (or connection change) reset to the top level.
  useEffect(() => {
    if (!open || !connectionId) return;
    setStack([]);
    loadLevel();
  }, [open, connectionId, loadLevel]);

  // Drill into a folder: push it on the stack and load its children.
  function drillInto(folder: BrowseFolder) {
    setStack((s) => [...s, folder]);
    loadLevel(folder.id);
  }
  // Jump to a breadcrumb: truncate the stack to that depth and reload. depth 0 = root.
  function jumpTo(depth: number) {
    const next = stack.slice(0, depth);
    setStack(next);
    loadLevel(next.length ? next[next.length - 1].id : undefined);
  }

  // The folder the "Import this folder" button imports: the current one (top of stack), or a virtual
  // root when we're at the top level (backend imports the whole connection from an empty folder_id).
  const current: BrowseFolder =
    stack.length > 0
      ? stack[stack.length - 1]
      : { id: "", name: level?.path_hint || providerName || "root" };

  async function confirmImport() {
    if (!confirming) return;
    setImportingNow(true);
    onImport(confirming);
    // The parent closes the modal and takes over the progress line; reset local state for next open.
    setImportingNow(false);
    setConfirming(null);
  }

  return (
    <dialog ref={ref} data-theme="engram" className="modal" data-testid="folder-picker" onClose={onClose}>
      <div className="modal-box max-w-lg border border-slate-800 bg-slate-900 text-slate-100">
        <h3 className="text-lg font-semibold">Choose a folder in {providerName}</h3>

        {/* Breadcrumb from path_hint + the visited-folder stack. Click a crumb to jump back up. */}
        <nav data-testid="picker-breadcrumb" className="mt-2 flex flex-wrap items-center gap-1 text-xs text-slate-400">
          <button type="button" className="hover:text-slate-200" onClick={() => jumpTo(0)}>
            {providerName || "Top"}
          </button>
          {stack.map((f, i) => (
            <span key={f.id} className="flex items-center gap-1">
              <span className="text-slate-600">/</span>
              <button type="button" className="hover:text-slate-200" onClick={() => jumpTo(i + 1)}>
                {f.name}
              </button>
            </span>
          ))}
        </nav>
        {level?.path_hint && (
          <p className="mt-1 truncate text-[11px] text-slate-500" title={level.path_hint}>
            {level.path_hint}
          </p>
        )}

        {/* Folder list. Click a row to drill in. */}
        <div className="mt-3 max-h-64 overflow-y-auto rounded-lg border border-slate-800" data-testid="picker-list">
          {loading ? (
            <p className="px-3 py-6 text-center text-sm text-slate-400">Loading…</p>
          ) : error ? (
            <p className="px-3 py-6 text-center text-sm text-red-400">{error}</p>
          ) : level && level.folders.length > 0 ? (
            <ul className="divide-y divide-slate-800 text-sm">
              {level.folders.map((f) => (
                <li key={f.id}>
                  <button
                    type="button"
                    onClick={() => drillInto(f)}
                    data-testid={`picker-folder-${f.id}`}
                    className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-slate-800/60"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="shrink-0 text-slate-500">
                      <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" strokeLinejoin="round" />
                    </svg>
                    <span className="truncate">{f.name}</span>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="px-3 py-6 text-center text-sm text-slate-500">No folders here.</p>
          )}
        </div>

        {/* Supported-file count for the current folder. */}
        {level && !loading && !error && (
          <p className="mt-2 text-xs text-slate-400" data-testid="picker-count">
            {level.supported_files} supported file{level.supported_files === 1 ? "" : "s"} here
          </p>
        )}

        {/* In-modal confirm before importing. */}
        {confirming && (
          <div className="mt-3 rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm text-slate-300" data-testid="picker-confirm">
            Import &lsquo;{confirming.name}&rsquo; — supported files in this folder and its subfolders
            will be added to your document base.
          </div>
        )}

        <div className="modal-action">
          <Button type="button" variant="outline" onClick={onClose}>
            {confirming ? "Cancel" : "Close"}
          </Button>
          {confirming ? (
            <Button type="button" variant="default" onClick={confirmImport} disabled={importingNow} data-testid="picker-confirm-import">
              {importingNow ? "Starting…" : "Import"}
            </Button>
          ) : (
            <Button
              type="button"
              variant="default"
              onClick={() => setConfirming(current)}
              disabled={loading || !!error}
              data-testid="picker-import"
            >
              Import this folder
            </Button>
          )}
        </div>
      </div>
      {/* Backdrop click closes. */}
      <form method="dialog" className="modal-backdrop">
        <button aria-label="Close" onClick={onClose}>
          close
        </button>
      </form>
    </dialog>
  );
}

// `sub` renders a small muted line under the value — used only by the Model stat to show the tier's
// description (whatever copy the registry carries), nothing hardcoded.
function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2">
      <dt className="text-xs text-slate-500">{label}</dt>
      <dd className="mt-0.5 truncate font-medium text-slate-100" title={value}>{value}</dd>
      {sub && <p className="mt-0.5 truncate text-xs text-slate-500" title={sub}>{sub}</p>}
    </div>
  );
}
