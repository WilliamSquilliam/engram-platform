"use client";
// Global beta-limit modal. Mounted once from the root layout. Listens for the "beta-limit-hit"
// window event that lib/api.ts dispatches when any endpoint returns 429 {error:"beta_limit"}, and
// shows one friendly modal carrying the server's message verbatim.
//
// This is deliberately NOT a paywall or an error state: beta caps stay invisible until a user hits
// one, and the hit should read as a friendly "ask us and we'll raise it", styled with the sky/info
// accent (never red). No usage bars, no limit numbers live anywhere else in the UI.
import { useEffect, useRef, useState } from "react";
import { BETA_LIMIT_HIT, type BetaLimitDetail } from "@/lib/api";
import { Button } from "@/components/ui";

export function BetaLimitNotice() {
  const [open, setOpen] = useState(false);
  const [message, setMessage] = useState("");
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    // Debounce bursts: concurrent uploads/queries can each throw a 429 in the same tick. Once the
    // modal is open we ignore further hits until it's dismissed, so a burst shows ONE modal, never a
    // stack. We read `open` from the dialog element (source of truth) to avoid a stale closure.
    function onHit(e: Event) {
      const detail = (e as CustomEvent<BetaLimitDetail>).detail;
      if (ref.current?.open) return; // already showing — swallow the rest of the burst
      setMessage(detail?.message || "You have reached a beta limit.");
      setOpen(true);
    }
    window.addEventListener(BETA_LIMIT_HIT, onHit);
    return () => window.removeEventListener(BETA_LIMIT_HIT, onHit);
  }, []);

  // Keep the native <dialog> in sync so Escape and the backdrop close it too (same pattern as the
  // platform-admin GpuConfirmDialog).
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  // Render nothing while closed so the dialog isn't in the DOM/a11y tree — a CSS-hidden <dialog> was
  // still announced by screen readers as a phantom "You have reached a beta limit" on every page.
  if (!open) return null;

  return (
    <dialog
      ref={ref}
      data-theme="engram"
      className="modal"
      data-testid="beta-limit-modal"
      onClose={() => setOpen(false)}
    >
      <div className="modal-box border border-sky-500/30 bg-slate-900 text-slate-100">
        <div className="flex items-start gap-3">
          {/* Friendly info glyph — sky, not error red. */}
          <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-sky-500/15 text-sky-300">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 16v-4M12 8h.01" strokeLinecap="round" />
            </svg>
          </div>
          <div className="min-w-0">
            <h3 className="text-lg font-semibold">You have reached a beta limit</h3>
            {/* The server writes the message; render it verbatim so copy stays in one place. */}
            <p className="mt-2 text-sm text-slate-300" data-testid="beta-limit-message">
              {message}
            </p>
            <p className="mt-3 text-xs text-slate-500">
              Beta limits exist to keep the platform fast for everyone. Ask us and we will raise yours.
            </p>
          </div>
        </div>
        <div className="modal-action">
          <Button
            type="button"
            variant="default"
            data-testid="beta-limit-dismiss"
            onClick={() => setOpen(false)}
          >
            Got it
          </Button>
        </div>
      </div>
      {/* Backdrop click closes (daisyUI form-method=dialog pattern). */}
      <form method="dialog" className="modal-backdrop">
        <button aria-label="Close">close</button>
      </form>
    </dialog>
  );
}
