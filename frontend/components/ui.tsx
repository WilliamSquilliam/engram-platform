// Minimal shadcn-style primitives (vendored Tailwind components). The production
// target is shadcn/ui proper; these keep the same shape so a swap is mechanical.
// Dark theme: surfaces are slate-900 on a slate-950 page, text is slate-100.
import * as React from "react";

export function cn(...c: (string | false | undefined | null)[]) {
  return c.filter(Boolean).join(" ");
}

export function Button({ className, variant = "default", ...props }: any) {
  const base =
    "inline-flex items-center justify-center rounded-md text-sm font-medium px-4 py-2 transition disabled:opacity-50 disabled:pointer-events-none";
  const variants: Record<string, string> = {
    default: "bg-slate-100 text-slate-900 hover:bg-white",
    outline: "border border-slate-700 bg-transparent text-slate-200 hover:bg-slate-800",
    ghost: "text-slate-200 hover:bg-slate-800",
    danger: "bg-red-600 text-white hover:bg-red-500",
  };
  return <button className={cn(base, variants[variant], className)} {...props} />;
}

export function Input({ className, ...props }: any) {
  return (
    <input
      className={cn(
        "w-full rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none focus:ring-2 focus:ring-slate-600",
        className
      )}
      {...props}
    />
  );
}

export function Label({ className, ...props }: any) {
  return <label className={cn("block text-sm font-medium mb-1 text-slate-300", className)} {...props} />;
}

export function Card({ className, ...props }: any) {
  return <div className={cn("rounded-lg border border-slate-800 bg-slate-900 shadow-sm", className)} {...props} />;
}
export function CardHeader({ className, ...props }: any) {
  return <div className={cn("p-4 border-b border-slate-800", className)} {...props} />;
}
export function CardBody({ className, ...props }: any) {
  return <div className={cn("p-4", className)} {...props} />;
}

export function Badge({ children, color = "slate" }: any) {
  const c: Record<string, string> = {
    slate: "bg-slate-800 text-slate-300",
    green: "bg-emerald-500/15 text-emerald-300",
    amber: "bg-amber-500/15 text-amber-300",
    red: "bg-red-500/15 text-red-300",
    blue: "bg-blue-500/15 text-blue-300",
    violet: "bg-violet-500/15 text-violet-300",
  };
  return <span className={cn("inline-flex rounded-full px-2 py-0.5 text-xs font-medium", c[color])}>{children}</span>;
}

// Determinate (value 0..1) or indeterminate (animated) progress bar.
export function ProgressBar({
  value,
  indeterminate,
  className,
}: {
  value?: number | null;
  indeterminate?: boolean;
  className?: string;
}) {
  if (indeterminate) {
    return (
      <div className={cn("h-2 w-full overflow-hidden rounded-full bg-slate-800", className)}>
        <div className="progress-indeterminate h-full w-2/5 rounded-full bg-slate-100" />
      </div>
    );
  }
  const pct = Math.max(0, Math.min(100, Math.round((value ?? 0) * 100)));
  return (
    <div className={cn("h-2 w-full overflow-hidden rounded-full bg-slate-800", className)}>
      <div
        className="h-full rounded-full bg-slate-100 transition-[width] duration-500 ease-out"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

// Horizontal wizard stepper — daisyUI's `steps` component (library CSS, not bespoke
// drawing): equal-width steps, connected progress line, labels under the circles.
// `current` is 0-based; done + current steps take the brand primary, done ones a check.
export function Stepper({ steps, current }: { steps: string[]; current: number }) {
  return (
    <ul data-theme="engram" className="steps w-full !bg-transparent text-xs" data-testid="stepper">
      {steps.map((label, i) => (
        <li
          key={label}
          data-content={i < current ? "✓" : String(i + 1)}
          className={cn(
            "step",
            i <= current && "step-primary",
            i === current ? "font-medium text-slate-100" : "text-slate-400"
          )}
        >
          {label}
        </li>
      ))}
    </ul>
  );
}
