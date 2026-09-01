import "./globals.css";
import type { ReactNode } from "react";
import { AppShell } from "@/components/AppShell";

export const metadata = {
  title: "Engram Smart CAG — Cartridge KV Platform",
  description: "Read-once / infer-many — onboard a corpus, chat, expose as MCP.",
  // Browser-tab icon = the ribbons mark alone; the in-app header uses the stacked wordmark.
  icons: { icon: "/icon.svg", shortcut: "/icon.svg", apple: "/icon.svg" },
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
