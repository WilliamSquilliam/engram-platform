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
  // Reasoning text the model streamed on the {thinking} channel BEFORE its answer deltas. Shown as a
  // collapsible aside above the answer; NEVER sent back as history content and never copied. Persisting
  // it here (the store already persists messages) is fine — it just must not enter request history.
  thinking?: string;
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

// Caps so localStorage can't grow without bound: keep the newest N conversations per corpus, and
// the newest M messages per conversation. Convos are stored newest-first, messages oldest-first.
const MAX_CONVERSATIONS = 50;
const MAX_MESSAGES = 200;

function load(corpusId: string): Conversation[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(KEY(corpusId));
    return raw ? (JSON.parse(raw) as Conversation[]) : [];
  } catch {
    return [];
  }
}

// Enforce the history caps: drop the oldest conversations past the limit and, within each, trim the
// oldest messages past the limit.
function trim(convos: Conversation[]): Conversation[] {
  return convos.slice(0, MAX_CONVERSATIONS).map((c) =>
    c.messages.length > MAX_MESSAGES ? { ...c, messages: c.messages.slice(-MAX_MESSAGES) } : c
  );
}

function save(corpusId: string, convos: Conversation[]) {
  const capped = trim(convos);
  try {
    localStorage.setItem(KEY(corpusId), JSON.stringify(capped));
  } catch (err) {
    // Over quota (or private mode): evict the oldest half of conversations and retry once. If it
    // still fails, warn rather than silently losing history so the failure is at least visible.
    if (err instanceof DOMException && err.name === "QuotaExceededError") {
      const evicted = capped.slice(0, Math.max(1, Math.floor(capped.length / 2)));
      try {
        localStorage.setItem(KEY(corpusId), JSON.stringify(evicted));
        return;
      } catch (retryErr) {
        console.warn("chatStore: localStorage quota exceeded; history not saved", retryErr);
        return;
      }
    }
    console.warn("chatStore: could not persist history", err);
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
