"use client";
// Conversational chat (E3) — the primary product surface. A Claude/Gemini-style thread over a ready
// corpus: streamed answers (SSE via /chat/stream), markdown + code rendering, citations to the source
// docs, client-side conversation history, a corpus picker, and stop / regenerate / copy. The corpus
// KV is resident on the serving engine, so this feels like chatting with the documents directly.
import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { Markdown } from "@/components/Markdown";
import { Button, cn } from "@/components/ui";
import {
  useConversations,
  titleFrom,
  type ChatMessage,
  type Conversation,
} from "@/lib/chatStore";
import type { Corpus } from "@/lib/types";

const SUGGESTIONS = [
  "Summarize the key points across these documents.",
  "What are the main risks or open questions?",
  "Find where this document base talks about pricing.",
];

export default function ChatPage() {
  const { id } = useParams() as { id: string };
  const router = useRouter();
  const { convos, active, activeId, setActiveId, startNew, remove, update } = useConversations(id);

  const [corpora, setCorpora] = useState<Corpus[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const streamRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const taRef = useRef<HTMLTextAreaElement | null>(null);

  // Corpus picker: only READY corpora can be chatted; switching navigates to that corpus's chat.
  useEffect(() => {
    api.listCorpora().then(setCorpora).catch(() => {});
  }, []);

  // Abort any in-flight stream on unmount so a fetch reader isn't left running after we leave.
  useEffect(() => () => streamRef.current?.abort(), []);

  // Switching corpora (id changes) points chat at a different resident KV — abort whatever is
  // streaming from the old corpus so its tokens don't land in the new thread.
  useEffect(() => {
    return () => streamRef.current?.abort();
  }, [id]);

  // Keep the thread pinned to the newest message as tokens stream in.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [active?.messages, busy]);

  const autosize = useCallback(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }, []);
  useEffect(autosize, [input, autosize]);

  // Core send: append the user turn + a streaming assistant turn, then consume /chat/stream. On a
  // regenerate the trailing assistant turn is dropped first (see regenerate) and the same history is
  // re-sent. History for the request = all prior turns of THIS conversation (excludes the new user
  // turn, which rides as `question`).
  const send = useCallback(
    async (question: string, opts: { convoId?: string; priorMessages?: ChatMessage[] } = {}) => {
      const q = question.trim();
      if (!q || busy) return;
      // Abort any stream still in flight before starting a new one (belt-and-suspenders: busy
      // usually gates this, but a regenerate or fast resubmit shouldn't leave a dangling reader).
      streamRef.current?.abort();
      let convoId = opts.convoId ?? activeId;
      if (!convoId) convoId = startNew();

      // History sent to the backend = prior turns (role/content only), oldest first, capped to
      // the LAST 20 messages (ChatReq.history max_length=20 — sending the full transcript 422'd
      // every conversation at its 11th question; the model keeps the docs resident regardless,
      // so dropping the oldest turns costs only distant chat context). Long individual messages
      // are truncated to the schema's 4k-char cap the same way.
      const prior = opts.priorMessages ?? active?.messages ?? [];
      const history = prior
        .filter((m) => !m.error && m.content)
        .map((m) => ({ role: m.role, content: m.content.slice(0, 4000) }))
        .slice(-20);
      const pinnedDocs = active?.pinnedDocs;

      // Append the user turn + an empty streaming assistant turn.
      update(
        convoId!,
        (c) => ({
          ...c,
          title: c.messages.length === 0 ? titleFrom(q) : c.title,
          messages: [
            ...(opts.priorMessages ? opts.priorMessages : c.messages),
            { role: "user", content: q },
            { role: "assistant", content: "", streaming: true },
          ],
        }),
        true
      );
      setBusy(true);

      const patchLast = (fn: (m: ChatMessage) => ChatMessage) =>
        update(convoId!, (c) => {
          const msgs = [...c.messages];
          msgs[msgs.length - 1] = fn(msgs[msgs.length - 1]);
          return { ...c, messages: msgs };
        });

      const { controller, done } = api.chatStream(
        id,
        q,
        (e: any) => {
          if (e.head) {
            const sources = e.sources || [];
            const used = e.used_docs || [];
            patchLast((m) => ({ ...m, sources, usedDocs: used }));
            // Re-pin the conversation to THIS turn's retrieval on every head frame, not just the first
            // unpinned turn. The server refreshes the pin on a topic shift (UPGRADE 2) and reports the
            // new used_docs here — updating pinnedDocs each turn propagates that shift to the next turn
            // (a stale pin would keep re-sending the old docs and undo the server-side refresh).
            if (used.length) {
              update(convoId!, (c) => ({ ...c, pinnedDocs: used }));
            }
          } else if (e.thinking) {
            // Reasoning frames arrive before the answer deltas. Accumulate them into the turn's
            // `thinking` field so the aside can render them and show activity while only thinking
            // has streamed (this doubles as the slow-first-token affordance).
            patchLast((m) => ({ ...m, thinking: (m.thinking || "") + e.thinking }));
          } else if (e.delta) {
            patchLast((m) => ({ ...m, content: m.content + e.delta }));
          } else if (e.escalate) {
            patchLast((m) => ({ ...m, tier: "rag-backup" }));
          } else if (e.done) {
            patchLast((m) => ({ ...m, tier: e.metrics?.tier || m.tier }));
          } else if (e.error) {
            patchLast((m) => ({
              ...m,
              content: m.content + (m.content ? "\n\n" : "") + `_${e.error}_`,
              error: true,
            }));
          }
        },
        { history, docIds: pinnedDocs }
      );
      streamRef.current = controller;

      try {
        await done;
      } catch (err: any) {
        if (err?.name !== "AbortError") {
          patchLast((m) => ({
            ...m,
            content: m.content + (m.content ? "\n\n" : "") + `_${err.message || "chat failed"}_`,
            error: true,
          }));
        }
      } finally {
        patchLast((m) => ({ ...m, streaming: false }));
        setBusy(false);
        streamRef.current = null;
      }
    },
    [busy, activeId, active, id, startNew, update]
  );

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const q = input;
    setInput("");
    send(q);
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const q = input;
      setInput("");
      send(q);
    }
  }

  function stop() {
    streamRef.current?.abort();
  }

  // Regenerate: re-ask the last user turn, dropping the assistant answer after it.
  function regenerate() {
    if (!active || busy) return;
    const msgs = active.messages;
    let lastUserIdx = -1;
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "user") { lastUserIdx = i; break; }
    }
    if (lastUserIdx < 0) return;
    const q = msgs[lastUserIdx].content;
    const prior = msgs.slice(0, lastUserIdx); // everything before the user turn
    send(q, { convoId: active.id, priorMessages: prior });
  }

  async function copy(text: string, key: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedKey(key);
      setTimeout(() => setCopiedKey(null), 1200);
    } catch {
      /* clipboard unavailable */
    }
  }

  const readyCorpora = corpora.filter((c) => c.status === "ready");
  const messages = active?.messages ?? [];
  const empty = messages.length === 0;

  return (
    <div className="flex h-full min-h-0">
      {/* Conversation history rail */}
      <aside className="hidden w-56 shrink-0 flex-col border-r border-slate-800 bg-slate-900/40 md:flex">
        <div className="p-3">
          <Button className="w-full gap-2" onClick={() => { startNew(); setInput(""); }} data-testid="chat-new">
            <span className="text-base leading-none">+</span> New chat
          </Button>
        </div>
        <div className="px-4 pb-1 text-xs font-medium uppercase tracking-wide text-slate-500">History</div>
        <nav className="flex-1 space-y-0.5 overflow-y-auto px-2" data-testid="chat-history">
          {convos.length === 0 && <p className="px-3 py-2 text-sm text-slate-500">No chats yet.</p>}
          {convos.map((c) => (
            <ConvoItem
              key={c.id}
              c={c}
              active={c.id === activeId}
              onOpen={() => setActiveId(c.id)}
              onDelete={() => remove(c.id)}
            />
          ))}
        </nav>
      </aside>

      {/* Thread */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Corpus picker */}
        <div className="flex items-center gap-2 border-b border-slate-800 px-4 py-2">
          <span className="text-xs text-slate-500">Document Base</span>
          <select
            data-testid="corpus-picker"
            value={id}
            onChange={(e) => router.push(`/document-base/${e.target.value}/chat`)}
            className="rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-sm text-slate-100 outline-none focus:ring-2 focus:ring-slate-600"
          >
            {readyCorpora.map((c) => (
              <option key={c.id} value={c.id}>{c.name}</option>
            ))}
            {!readyCorpora.some((c) => c.id === id) && <option value={id}>This document base</option>}
          </select>
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto" data-testid="chat-thread">
          {empty ? (
            <div className="mx-auto flex h-full max-w-2xl flex-col items-center justify-center px-6 text-center">
              <h2 className="text-2xl font-semibold text-slate-100">Chat with your document base</h2>
              <p className="mt-2 text-sm text-slate-400">
                Ask anything — answers stream straight from the resident documents, with sources.
              </p>
              <div className="mt-6 grid w-full gap-2 sm:grid-cols-3">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="rounded-lg border border-slate-800 bg-slate-900 p-3 text-left text-sm text-slate-300 transition hover:border-slate-700 hover:bg-slate-800/60"
                    data-testid="chat-suggestion"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
              {messages.map((m, i) => (
                <Message
                  key={i}
                  m={m}
                  onCopy={() => copy(m.content, `${activeId}-${i}`)}
                  copied={copiedKey === `${activeId}-${i}`}
                />
              ))}
            </div>
          )}
        </div>

        {/* Composer */}
        <div className="border-t border-slate-800 bg-slate-950 px-4 py-3">
          <form onSubmit={onSubmit} className="mx-auto max-w-3xl">
            <div className="flex items-end gap-2 rounded-2xl border border-slate-700 bg-slate-900 px-3 py-2 focus-within:ring-2 focus-within:ring-slate-600">
              <textarea
                ref={taRef}
                data-testid="chat-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                rows={1}
                placeholder="Message your document base…"
                className="max-h-52 min-h-[1.5rem] flex-1 resize-none bg-transparent text-sm text-slate-100 placeholder-slate-500 outline-none"
              />
              {busy ? (
                <Button type="button" variant="outline" onClick={stop} data-testid="chat-stop">
                  Stop
                </Button>
              ) : (
                <Button type="submit" disabled={!input.trim()} data-testid="chat-send">
                  Send
                </Button>
              )}
            </div>
            <div className="mt-1.5 flex items-center justify-between px-1 text-[11px] text-slate-500">
              <span>Enter to send · Shift+Enter for a new line</span>
              {!busy && messages.length > 0 && (
                <button
                  type="button"
                  onClick={regenerate}
                  className="hover:text-slate-300"
                  data-testid="chat-regenerate"
                >
                  Regenerate
                </button>
              )}
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

function Message({ m, onCopy, copied }: { m: ChatMessage; onCopy: () => void; copied: boolean }) {
  if (m.role === "user") {
    return (
      <div className="msg-in flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl bg-slate-800 px-4 py-2.5 text-[15px] text-slate-100">
          {m.content}
        </div>
      </div>
    );
  }
  return (
    <div className="msg-in group">
      <div className="mb-1 flex items-center gap-2 text-xs text-slate-500">
        <span className="font-medium text-slate-400">Engram</span>
        {m.tier === "rag-backup" && (
          <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] text-amber-300">
            verified with documents
          </span>
        )}
      </div>
      {/* Reasoning aside: shows above the answer. While only thinking has streamed (no answer delta
          yet) it's expanded with a shimmering "Thinking…" label so the turn visibly works; once the
          first delta lands it auto-collapses to a "Thought for a moment ▸" toggle. */}
      {m.thinking && <ThinkingAside thinking={m.thinking} answered={!!m.content} />}

      {m.content ? (
        <div className={cn(m.error && "text-slate-400")}>
          <Markdown text={m.content} className={m.streaming ? "stream-caret" : undefined} />
        </div>
      ) : m.thinking ? null : (
        <div className="flex gap-1 py-1" aria-label="Thinking">
          <Dot /> <Dot delay="0.15s" /> <Dot delay="0.3s" />
        </div>
      )}

      {m.sources && m.sources.length > 0 && (
        <div className="mt-3 text-xs" data-testid="chat-sources">
          <span className="text-slate-500">Sources</span>
          <div className="mt-1 flex flex-wrap gap-1.5">
            {m.sources.map((s) => (
              <span
                key={s.id}
                title={s.id}
                className="rounded-md border border-slate-800 bg-slate-900 px-2 py-1 text-slate-300"
              >
                {s.title}
              </span>
            ))}
          </div>
        </div>
      )}

      {!m.streaming && m.content && (
        <div className="mt-2 opacity-0 transition group-hover:opacity-100">
          <button onClick={onCopy} className="text-xs text-slate-500 hover:text-slate-300" data-testid="chat-copy">
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      )}
    </div>
  );
}

// Collapsible reasoning aside rendered above the answer. `answered` is true once the first answer
// delta has landed. Before then it's forced open with a shimmering "Thinking…" label (activity +
// slow-first-token affordance); after, it collapses to a small toggle the user can re-expand to read
// the muted thinking text. Local open state so a user who expands it after the answer stays expanded.
function ThinkingAside({ thinking, answered }: { thinking: string; answered: boolean }) {
  const [open, setOpen] = useState(false);
  // While still thinking, force it open; once answered, honor the user's toggle (default collapsed).
  const expanded = answered ? open : true;
  return (
    <div className="mb-2" data-testid="chat-thinking">
      <button
        type="button"
        onClick={() => answered && setOpen((o) => !o)}
        disabled={!answered}
        className={cn(
          "flex items-center gap-1.5 text-xs text-slate-500",
          answered ? "hover:text-slate-300" : "cursor-default"
        )}
      >
        {answered ? (
          <>
            <span className={cn("transition-transform", expanded && "rotate-90")}>▸</span>
            <span>Thought for a moment</span>
          </>
        ) : (
          <span className="animate-pulse">Thinking…</span>
        )}
      </button>
      {expanded && (
        <div className="mt-1 whitespace-pre-wrap border-l border-slate-800 pl-3 text-xs text-slate-500">
          {thinking}
        </div>
      )}
    </div>
  );
}

function Dot({ delay }: { delay?: string }) {
  return (
    <span
      className="h-2 w-2 animate-bounce rounded-full bg-slate-600"
      style={delay ? { animationDelay: delay } : undefined}
    />
  );
}

function ConvoItem({
  c, active, onOpen, onDelete,
}: { c: Conversation; active: boolean; onOpen: () => void; onDelete: () => void }) {
  return (
    <div
      className={cn(
        "group flex items-center rounded-md",
        active ? "bg-slate-800" : "hover:bg-slate-800/60"
      )}
    >
      <button
        onClick={onOpen}
        className="flex min-w-0 flex-1 items-center px-3 py-2 text-left text-sm"
        data-testid="chat-history-item"
      >
        <span className="truncate text-slate-200" title={c.title}>{c.title}</span>
      </button>
      <button
        onClick={onDelete}
        aria-label="Delete conversation"
        className="mr-1 rounded p-1 text-slate-500 opacity-0 transition hover:text-red-400 group-hover:opacity-100"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
    </div>
  );
}
