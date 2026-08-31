"use client";
// Minimal, dependency-free markdown renderer for chat answers. Builds React nodes (never sets
// innerHTML), so there is no XSS surface — model output is data, rendered as text with formatting.
// Handles: fenced code blocks (with a copy button + language label), inline code, bold, italic,
// links, headings, unordered/ordered lists, blockquotes, and paragraphs. Good enough for the chat
// surface; the intended production swap is react-markdown, but this keeps the build lean and safe.
import * as React from "react";

// --- inline: escape-free tokenizer over `code`, **bold**, *italic*, [text](url) --------------------
// We split on inline code first (its content is literal), then apply the other spans to the rest.
function renderInline(text: string, keyBase: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  const codeRe = /`([^`]+)`/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = codeRe.exec(text))) {
    if (m.index > last) out.push(...renderSpans(text.slice(last, m.index), `${keyBase}-t${i}`));
    out.push(
      <code key={`${keyBase}-c${i}`} className="rounded bg-slate-950/60 px-1 py-0.5 font-mono text-slate-200">
        {m[1]}
      </code>
    );
    last = m.index + m[0].length;
    i++;
  }
  if (last < text.length) out.push(...renderSpans(text.slice(last), `${keyBase}-t${i}`));
  return out;
}

// Bold / italic / links inside a plain (non-code) run.
function renderSpans(text: string, keyBase: string): React.ReactNode[] {
  const re = /(\*\*([^*]+)\*\*)|(\*([^*]+)\*)|(\[([^\]]+)\]\((https?:\/\/[^\s)]+)\))/g;
  const out: React.ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[2] !== undefined) {
      out.push(<strong key={`${keyBase}-b${i}`}>{m[2]}</strong>);
    } else if (m[4] !== undefined) {
      out.push(<em key={`${keyBase}-i${i}`}>{m[4]}</em>);
    } else if (m[6] !== undefined) {
      out.push(
        <a key={`${keyBase}-a${i}`} href={m[7]} target="_blank" rel="noopener noreferrer">
          {m[6]}
        </a>
      );
    }
    last = m.index + m[0].length;
    i++;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function CodeBlock({ code, lang }: { code: string; lang?: string }) {
  const [copied, setCopied] = React.useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      /* clipboard unavailable */
    }
  };
  return (
    <div className="group relative overflow-hidden rounded-md border border-slate-800 bg-slate-950/70">
      <div className="flex items-center justify-between border-b border-slate-800 px-3 py-1 text-[11px] text-slate-500">
        <span>{lang || "code"}</span>
        <button onClick={copy} className="rounded px-1.5 py-0.5 text-slate-400 hover:bg-slate-800 hover:text-slate-200">
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto p-3 text-[13px] leading-relaxed">
        <code className="font-mono text-slate-200">{code}</code>
      </pre>
    </div>
  );
}

// --- block parser: split the text into fenced-code and non-code segments, then parse blocks --------
type Block =
  | { t: "code"; code: string; lang?: string }
  | { t: "h"; level: number; text: string }
  | { t: "ul"; items: string[] }
  | { t: "ol"; items: string[] }
  | { t: "quote"; text: string }
  | { t: "p"; text: string };

function parseBlocks(src: string): Block[] {
  const blocks: Block[] = [];
  // First pull fenced code blocks out so their contents are never treated as markdown.
  const fence = /```(\w+)?\n?([\s\S]*?)```/g;
  let last = 0;
  let m: RegExpExecArray | null;
  const pushText = (text: string) => {
    for (const b of parseTextBlocks(text)) blocks.push(b);
  };
  while ((m = fence.exec(src))) {
    if (m.index > last) pushText(src.slice(last, m.index));
    blocks.push({ t: "code", code: m[2].replace(/\n$/, ""), lang: m[1] });
    last = m.index + m[0].length;
  }
  // An unterminated fence (still streaming): render the remainder as an open code block.
  if (last < src.length) {
    const rest = src.slice(last);
    const open = rest.match(/```(\w+)?\n?([\s\S]*)$/);
    if (open) {
      pushText(rest.slice(0, open.index));
      blocks.push({ t: "code", code: open[2], lang: open[1] });
    } else {
      pushText(rest);
    }
  }
  return blocks;
}

function parseTextBlocks(text: string): Block[] {
  const out: Block[] = [];
  const lines = text.split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    const h = line.match(/^(#{1,3})\s+(.*)$/);
    if (h) { out.push({ t: "h", level: h[1].length, text: h[2] }); i++; continue; }
    if (/^>\s?/.test(line)) {
      const buf: string[] = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) { buf.push(lines[i].replace(/^>\s?/, "")); i++; }
      out.push({ t: "quote", text: buf.join("\n") });
      continue;
    }
    if (/^[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i])) { items.push(lines[i].replace(/^[-*]\s+/, "")); i++; }
      out.push({ t: "ul", items });
      continue;
    }
    if (/^\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])) { items.push(lines[i].replace(/^\d+\.\s+/, "")); i++; }
      out.push({ t: "ol", items });
      continue;
    }
    // Paragraph: consume until a blank line or a block-starting line.
    const buf: string[] = [];
    while (
      i < lines.length && lines[i].trim() &&
      !/^(#{1,3})\s+/.test(lines[i]) && !/^>\s?/.test(lines[i]) &&
      !/^[-*]\s+/.test(lines[i]) && !/^\d+\.\s+/.test(lines[i])
    ) { buf.push(lines[i]); i++; }
    out.push({ t: "p", text: buf.join("\n") });
  }
  return out;
}

export function Markdown({ text, className }: { text: string; className?: string }) {
  const blocks = React.useMemo(() => parseBlocks(text), [text]);
  return (
    <div className={"md text-[15px] leading-relaxed text-slate-100" + (className ? " " + className : "")}>
      {blocks.map((b, i) => {
        const k = `b${i}`;
        switch (b.t) {
          case "code":
            return <CodeBlock key={k} code={b.code} lang={b.lang} />;
          case "h": {
            const Tag = (`h${b.level}` as unknown) as keyof JSX.IntrinsicElements;
            return <Tag key={k}>{renderInline(b.text, k)}</Tag>;
          }
          case "ul":
            return <ul key={k}>{b.items.map((it, j) => <li key={j}>{renderInline(it, `${k}-${j}`)}</li>)}</ul>;
          case "ol":
            return <ol key={k}>{b.items.map((it, j) => <li key={j}>{renderInline(it, `${k}-${j}`)}</li>)}</ol>;
          case "quote":
            return <blockquote key={k}>{renderInline(b.text, k)}</blockquote>;
          default:
            return <p key={k} className="whitespace-pre-wrap">{renderInline(b.text, k)}</p>;
        }
      })}
    </div>
  );
}
