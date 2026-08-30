"use client";
// New Corpus wizard — Steps 2 & 3: Upload then Train. Keyed by corpus id so it is
// fully resumable: if the user exits mid-training, the dashboard routes them back
// here and we pick the in-flight run back up (useTrainingJob.resume). Controls:
// Back (to dashboard), Cancel training, Exit (leave; training keeps running).
import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import { api, getToken } from "@/lib/api";
import { useTrainingJob, fmtClock } from "@/lib/useTrainingJob";
import { Button, Card, CardBody, CardHeader, Stepper, cn } from "@/components/ui";

const TEXT_RE = /\.(txt|md|markdown|text)$/i;
type Step = "upload" | "train" | "done";

// Training runs in three stages; the ranges mirror ml_service (_PREP_FRAC = 0.15
// and the 0.97 train/save split) so each stage's bar fills off the overall
// progress. The timers below stay for the whole run, not per stage.
const TRAIN_STAGES = [
  { key: "analyze", label: "Analyze Documents", lo: 0, hi: 0.15 },
  { key: "train", label: "Train Cartridges", lo: 0.15, hi: 0.97 },
  { key: "save", label: "Save", lo: 0.97, hi: 1 },
];
const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

export default function CorpusSetupPage() {
  const { id } = useParams() as { id: string };
  const router = useRouter();

  const [corpus, setCorpus] = useState<any>(null);
  const [docs, setDocs] = useState<any[]>([]);
  const [step, setStep] = useState<Step>("upload");
  const [uploading, setUploading] = useState(false);
  const [note, setNote] = useState("");
  const [canceling, setCanceling] = useState(false);
  const folderRef = useRef<HTMLInputElement>(null);

  const train = useTrainingJob(id, (job) => {
    // Fired when the run leaves "running": advance or fall back to upload.
    load();
    if (job.status === "succeeded") setStep("done");
    else if (job.status === "canceled") {
      setStep("upload");
      setNote("Training canceled.");
    } else if (job.status === "failed") {
      setStep("upload");
      setNote("Training failed: " + (job.detail || "unknown error"));
    }
    setCanceling(false);
  });

  async function load() {
    const c = await api.getCorpus(id);
    setCorpus(c);
    setDocs(await api.listDocuments(id));
    return c;
  }

  useEffect(() => {
    if (!getToken()) {
      router.push("/login");
      return;
    }
    (async () => {
      let c;
      try {
        c = await load();
      } catch {
        router.push("/login");
        return;
      }
      if (c.status === "training") {
        setStep("train");
        train.resume();
      } else if (c.status === "ready") {
        setStep("done");
      } else {
        setStep("upload");
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleFiles(files: File[]) {
    const texts = files.filter((f) => TEXT_RE.test((f as any).path || f.name));
    const skipped = files.length - texts.length;
    if (!texts.length) {
      setNote(`No .txt/.md files found${skipped ? ` (skipped ${skipped})` : ""}.`);
      return;
    }
    setUploading(true);
    setNote("");
    try {
      await api.uploadDocuments(id, texts);
      setNote(`Added ${texts.length} file${texts.length === 1 ? "" : "s"}${skipped ? `, skipped ${skipped} non-text` : ""}.`);
      await load();
    } catch (err: any) {
      setNote("Upload failed: " + err.message);
    } finally {
      setUploading(false);
    }
  }

  const { getRootProps, getInputProps, open, isDragActive } = useDropzone({
    noClick: true,
    noKeyboard: true,
    multiple: true,
    onDrop: (accepted) => handleFiles(accepted),
  });

  async function startTrain() {
    setNote("");
    setStep("train");
    try {
      await train.start();
    } catch (err: any) {
      setStep("upload");
      setNote("Failed to start training: " + err.message);
    }
  }

  async function cancelTrain() {
    setCanceling(true);
    try {
      await train.cancel();
    } catch {
      setCanceling(false);
    }
  }

  if (!corpus) return <main className="p-8 text-slate-400">Loading…</main>;
  const stepIndex = step === "upload" ? 1 : 2;

  return (
    <main className="mx-auto max-w-2xl p-8">
      <button
        onClick={() => router.push("/")}
        className="text-sm text-slate-400 hover:text-slate-200"
        data-testid="setup-exit"
      >
        ← All Corpora
      </button>
      <h1 className="mt-2 mb-4 text-2xl font-semibold" data-testid="setup-title">
        {corpus.name}
      </h1>
      <div className="mb-6">
        <Stepper steps={["Name", "Upload", "Train"]} current={stepIndex} />
      </div>

      {/* ---------------- Upload step ---------------- */}
      {step === "upload" && (
        <Card>
          <CardHeader className="flex items-center justify-between">
            <h2 className="font-medium">Add Documents</h2>
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
              <p className="text-sm text-slate-300">
                {isDragActive ? "Drop to add" : "Drag files & folders here"}
              </p>
              <div className="mt-3">
                <Button type="button" data-testid="select-corpus" onClick={open} disabled={uploading}>
                  {uploading ? "Uploading…" : "Select Corpus"}
                </Button>
              </div>
              <p className="mt-2 text-xs text-slate-500">
                Files or folders, any combination ·{" "}
                <button
                  type="button"
                  data-testid="select-folder"
                  className="underline hover:text-slate-300"
                  onClick={() => folderRef.current?.click()}
                  disabled={uploading}
                >
                  Select a folder
                </button>{" "}
                · .txt / .md
              </p>
              {note && <p data-testid="upload-note" className="mt-2 text-xs text-slate-400">{note}</p>}
            </div>

            {docs.length > 0 && (
              <ul data-testid="doc-list" className="max-h-40 divide-y divide-slate-800 overflow-y-auto text-sm">
                {docs.map((d) => (
                  <li key={d.id} className="flex justify-between gap-2 py-1">
                    <span className="truncate" title={d.filename}>{d.filename}</span>
                    <span className="shrink-0 text-slate-500">{d.size} B</span>
                  </li>
                ))}
              </ul>
            )}

            <div className="flex justify-between">
              <Button variant="outline" onClick={() => router.push("/")} data-testid="setup-back">
                Back
              </Button>
              <Button onClick={startTrain} disabled={uploading || docs.length === 0} data-testid="train-btn">
                Train Cartridges
              </Button>
            </div>
          </CardBody>
        </Card>
      )}

      {/* ---------------- Train step ---------------- */}
      {step === "train" && (
        <Card>
          <CardHeader>
            <h2 className="font-medium">Training Cartridges</h2>
          </CardHeader>
          <CardBody className="space-y-4">
            <div data-testid="train-progress" className="space-y-3">
              {/* Stage stepper: the bar is split into the stages of the run. */}
              <div className="flex gap-2">
                {TRAIN_STAGES.map((s, i) => {
                  const fill = clamp01((train.progress - s.lo) / (s.hi - s.lo));
                  const done = train.progress >= s.hi;
                  const active = !done && train.progress >= s.lo;
                  return (
                    <div key={s.key} className="flex-1 space-y-1.5" data-testid={`stage-${s.key}`}>
                      <div className="flex items-center gap-1.5">
                        <span
                          className={cn(
                            "flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-semibold",
                            done
                              ? "bg-emerald-500 text-slate-950"
                              : active
                              ? "bg-slate-100 text-slate-900"
                              : "bg-slate-800 text-slate-400"
                          )}
                        >
                          {done ? "✓" : i + 1}
                        </span>
                        <span
                          className={cn(
                            "text-xs",
                            active ? "font-medium text-slate-100" : done ? "text-slate-400" : "text-slate-500"
                          )}
                        >
                          {s.label}
                        </span>
                      </div>
                      <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
                        <div
                          className={cn(
                            "h-full rounded-full transition-[width] duration-500 ease-out",
                            done ? "bg-emerald-500" : "bg-slate-100"
                          )}
                          style={{ width: `${fill * 100}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
              {/* Detail + overall timers (whole run, not per stage). */}
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span data-testid="train-phase">{train.phase || "Starting…"}</span>
                <span className="tabular-nums">
                  <span data-testid="train-elapsed">{fmtClock(train.elapsed)}</span>
                  {" · "}
                  <span data-testid="train-pct">{train.progress ? `${train.pct}%` : "0%"}</span>
                  {" · "}
                  <span data-testid="train-remaining">
                    {train.eta != null ? `-${fmtClock(train.eta)}` : "--:--"}
                  </span>
                </span>
              </div>
            </div>
            <p className="text-xs text-slate-500">
              You can leave this page — training keeps running and the corpus stays under
              “Corpora” as <b>Training</b>. Click it there to return here.
            </p>
            <div className="flex justify-between">
              <Button variant="danger" onClick={cancelTrain} disabled={canceling} data-testid="cancel-train">
                {canceling ? "Canceling…" : "Cancel Training"}
              </Button>
              <Button variant="outline" onClick={() => router.push("/")} data-testid="train-exit">
                Exit
              </Button>
            </div>
          </CardBody>
        </Card>
      )}

      {/* ---------------- Done step ---------------- */}
      {step === "done" && (
        <Card className="border-emerald-500/30 bg-emerald-500/10">
          <CardBody className="space-y-4">
            <div>
              <div className="text-lg font-semibold" data-testid="setup-done">Training Complete 🎉</div>
              <p className="text-sm text-slate-300">
                {corpus.n_cartridges ?? docs.length} cartridge
                {(corpus.n_cartridges ?? docs.length) === 1 ? "" : "s"} ready. Open the corpus to
                compare strategies, expose it as an MCP server, and view costs.
              </p>
            </div>
            <div className="flex gap-2">
              <Button onClick={() => router.push(`/corpus/${id}/chat`)} data-testid="open-corpus">
                Open Corpus
              </Button>
              <Button variant="outline" onClick={() => setStep("upload")} data-testid="retrain">
                Add Documents / Re-train
              </Button>
            </div>
          </CardBody>
        </Card>
      )}
    </main>
  );
}
