"use client";
// /document-base/[id] is a router: ready corpora go to the workspace (Chat), others
// resume the onboarding wizard. Keeps deep links and old bookmarks working.
import { useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { api, getToken } from "@/lib/api";

export default function CorpusIndex() {
  const { id } = useParams() as { id: string };
  const router = useRouter();

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
      return;
    }
    (async () => {
      try {
        const c = await api.getCorpus(id);
        router.replace(c.status === "ready" ? `/document-base/${id}/chat` : `/document-base/${id}/setup`);
      } catch {
        router.replace("/login");
      }
    })();
  }, [id, router]);

  return <main className="p-6 text-slate-500">Loading…</main>;
}
