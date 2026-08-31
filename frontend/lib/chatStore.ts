"use client";
// Client-side conversation history for the chat surface. Persists per corpus to localStorage so a
// reload keeps the thread and the sidebar of past conversations. No server store needed for the MVP
// (the backend chat endpoint is stateless — history rides in each request), so this is the source of
// truth for "what has been said" purely on the client.
import { useCallback, useEffect, useState } from "react";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  // Sources the assistant turn cited (set on the head event of the stream).
  sources?: { id: string; title: string }[];
  // The doc_ids the first turn retrieved; pinned so follow-ups skip re-retrieval.
  usedDocs?: string[];
  // The routing tier the answer came from (cartridge | rag-backup), for a small badge.
  tier?: string;
  // True while this assistant turn is still streaming.
  streaming?: boolean;
  // True if this turn errored (rendered muted).
  error?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  messages: ChatMessage[];
  // The doc_ids the conversation is pinned to (from the first answered turn), reused on every
  // follow-up so retrieval runs once and the thread stays on the same evidence.
  pinnedDocs?: string[];
}

const KEY = (corpusId: string) => `engram:chat:${corpusId}`;

function load(corpusId: string): Conversation[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(KEY(corpusId));
    return raw ? (JSON.parse(raw) as Conversation[]) : [];
  } catch {
    return [];
  }
}

function save(corpusId: string, convos: Conversation[]) {
  try {
    localStorage.setItem(KEY(corpusId), JSON.stringify(convos));
  } catch {
    /* quota / private mode — history is best-effort */
  }
}

export function newConversation(): Conversation {
  return { id: Math.random().toString(36).slice(2), title: "New chat", createdAt: Date.now(), messages: [] };
}

// Derive a short title from the first user message.
export function titleFrom(text: string): string {
  const t = text.trim().replace(/\s+/g, " ");
  return t.length > 40 ? t.slice(0, 40) + "…" : t || "New chat";
}

export function useConversations(corpusId: string) {
  const [convos, setConvos] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  // Hydrate once on mount (localStorage is client-only).
  useEffect(() => {
    const list = load(corpusId);
    setConvos(list);
    setActiveId(list[0]?.id ?? null);
  }, [corpusId]);

  const persist = useCallback(
    (next: Conversation[]) => {
      setConvos(next);
      save(corpusId, next);
    },
    [corpusId]
  );

  const startNew = useCallback(() => {
    const c = newConversation();
    persist([c, ...convos]);
    setActiveId(c.id);
    return c.id;
  }, [convos, persist]);

  const remove = useCallback(
    (id: string) => {
      const next = convos.filter((c) => c.id !== id);
      persist(next);
      if (activeId === id) setActiveId(next[0]?.id ?? null);
    },
    [convos, activeId, persist]
  );

  // Update one conversation by id via an updater. Also lifts it to the top on activity.
  const update = useCallback(
    (id: string, fn: (c: Conversation) => Conversation, bump = false) => {
      setConvos((prev) => {
        const idx = prev.findIndex((c) => c.id === id);
        if (idx < 0) return prev;
        const updated = fn(prev[idx]);
        let next = [...prev];
        next[idx] = updated;
        if (bump && idx > 0) {
          next.splice(idx, 1);
          next = [updated, ...next];
        }
        save(corpusId, next);
        return next;
      });
    },
    [corpusId]
  );

  const active = convos.find((c) => c.id === activeId) ?? null;
  return { convos, active, activeId, setActiveId, startNew, remove, update };
}
