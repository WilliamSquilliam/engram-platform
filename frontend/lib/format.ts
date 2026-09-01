// Shared display formatters so the same value renders the same way everywhere (single source of
// truth — the onboarding wizard and the documents tab both pull from here).
import type { ParseStatus } from "./types";

// Human-readable file size (B / KB / MB).
export const fmtBytes = (n: number) => {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
};

// Per-document parse-status badge: label + the Badge palette color to use.
export const PARSE_BADGE: Record<ParseStatus, { label: string; color: string }> = {
  pending: { label: "Queued", color: "slate" },
  parsing: { label: "Parsing…", color: "amber" },
  parsed: { label: "Ready", color: "green" },
  failed: { label: "Failed", color: "red" },
};
