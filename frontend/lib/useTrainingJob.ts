"use client";
// Shared training-job controller: start / cancel / resume a corpus training run
// and expose live progress. Used by the New Corpus wizard (setup) and the
// dashboard so the progress mechanism lives in exactly one place.
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";

const POLL_MS = 1500;

export function useTrainingJob(corpusId: string, onComplete?: (job: any) => void) {
  const [job, setJob] = useState<any>(null);
  const [training, setTraining] = useState(false);
  const pollRef = useRef<any>(null);
  // Keep the latest onComplete without re-subscribing the interval each render.
  const cbRef = useRef(onComplete);
  cbRef.current = onComplete;

  const stop = useCallback(() => {
    clearInterval(pollRef.current);
    pollRef.current = null;
  }, []);

  const poll = useCallback(
    (jobId: string) => {
      stop();
      pollRef.current = setInterval(async () => {
        try {
          const j = await api.getJob(jobId);
          setJob(j);
          if (j.status !== "running") {
            stop();
            setTraining(false);
            cbRef.current?.(j);
          }
        } catch {
          /* transient network error — keep polling */
        }
      }, POLL_MS);
    },
    [stop]
  );

  // Pick up a run that was already in flight (e.g. user navigated back to it).
  const resume = useCallback(async (): Promise<boolean> => {
    try {
      const jobs = await api.listJobs(corpusId);
      const running = jobs.find((j: any) => j.status === "running");
      if (running) {
        setJob(running);
        setTraining(true);
        poll(running.id);
        return true;
      }
    } catch {
      /* ignore */
    }
    return false;
  }, [corpusId, poll]);

  const start = useCallback(async () => {
    setTraining(true);
    setJob(null);
    const j = await api.train(corpusId);
    setJob(j);
    poll(j.id);
    return j;
  }, [corpusId, poll]);

  // Request cancellation; the worker aborts at its next heartbeat and the poll
  // observes the status flip to "canceled" (which fires onComplete).
  const cancel = useCallback(async () => {
    await api.cancelTraining(corpusId);
  }, [corpusId]);

  useEffect(() => () => stop(), [stop]);

  // Live timers. Anchor to the server clock: (updated_at - created_at) is the
  // time at the last heartbeat, and because both timestamps are parsed in the
  // same (browser) timezone their difference is correct regardless of offset.
  // We then interpolate forward with a 1s ticker so the readout counts smoothly
  // between heartbeats. Re-anchoring on each poll corrects any drift, and it
  // stays accurate across a resume (the timestamps are server truth).
  const anchorRef = useRef<{ at: number; elapsed: number }>({ at: Date.now(), elapsed: 0 });
  const [, forceTick] = useState(0);

  useEffect(() => {
    if (job?.created_at && job?.updated_at) {
      const base = (Date.parse(job.updated_at) - Date.parse(job.created_at)) / 1000;
      anchorRef.current = { at: Date.now(), elapsed: Math.max(0, base) };
    }
  }, [job?.created_at, job?.updated_at]);

  useEffect(() => {
    if (!training) return;
    const iv = setInterval(() => forceTick((t) => t + 1), 1000);
    return () => clearInterval(iv);
  }, [training]);

  const sinceAnchor = training ? (Date.now() - anchorRef.current.at) / 1000 : 0;
  const elapsed = anchorRef.current.elapsed + sinceAnchor;
  const etaBase = (job?.eta_seconds as number | null) ?? null;
  const remaining = etaBase == null ? null : Math.max(0, etaBase - sinceAnchor);

  const progress: number = job?.progress || 0;
  return {
    job,
    training,
    progress,
    pct: Math.round(progress * 100),
    phase: (job?.detail as string) || "",
    elapsed, // seconds, counting up
    eta: remaining, // seconds, counting down
    start,
    cancel,
    resume,
    stop,
  };
}

// Clock-style m:ss (or h:mm:ss) — no "elapsed"/"remaining"/"left" wording.
export function fmtClock(s?: number | null): string {
  if (s == null || !isFinite(s) || s < 0) return "0:00";
  const t = Math.floor(s);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const sec = t % 60;
  const mm = h > 0 ? String(m).padStart(2, "0") : String(m);
  return (h > 0 ? `${h}:` : "") + `${mm}:${String(sec).padStart(2, "0")}`;
}
