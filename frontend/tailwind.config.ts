import type { Config } from "tailwindcss";
import daisyui from "daisyui";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: { extend: {} },
  // daisyUI supplies component classes (we use its `steps` stepper). Scoped tight:
  // no base-style injection (our slate theme stays untouched); the one custom theme
  // carries the brand indigo and is applied per-element via data-theme="engram".
  plugins: [daisyui],
  daisyui: {
    base: false,
    logs: false,
    themes: [
      {
        engram: {
          primary: "#5b54ee",          // ribbons indigo — done/current steps
          "primary-content": "#ffffff",
          secondary: "#34d399",
          accent: "#38bdf8",
          neutral: "#334155",          // undone step circles/lines (slate-700)
          "neutral-content": "#94a3b8",
          "base-100": "#0f172a",       // page background (slate-900)
          "base-200": "#1e293b",
          "base-300": "#334155",
          "base-content": "#e2e8f0",
        },
      },
    ],
  },
};
export default config;
