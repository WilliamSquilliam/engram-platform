import "./globals.css";
import type { ReactNode } from "react";
import { AppShell } from "@/components/AppShell";

export const metadata = {
  title: "Engram Smart CAG — Cartridge KV Platform",
  description: "Read-once / infer-many — onboard a corpus, chat, expose as MCP.",
  // The deployed brand mark (indigo gradient "E") — same logo the landing site uses.
  icons: { icon: "/logo.svg", shortcut: "/logo.svg", apple: "/logo.svg" },
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
