"use client";
// Markdown renderer for chat answers. Uses react-markdown (safe by default — it does NOT render
// raw HTML, so there is no XSS surface) + remark-gfm (tables / strikethrough / task lists) +
// react-syntax-highlighter (fenced-code highlighting, theme bundled as JS so there's no CSS import
// to wire). Standard, maintained stack — replaces the previous hand-rolled parser.
import * as React from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import remarkGfm from "remark-gfm";

// A fenced code block with a language label + copy button.
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
        <button
          onClick={copy}
          className="rounded px-1.5 py-0.5 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <SyntaxHighlighter
        language={lang}
        style={oneDark}
        PreTag="pre"
        customStyle={{ margin: 0, background: "transparent", padding: "0.75rem", fontSize: "13px" }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

export function Markdown({ text, className }: { text: string; className?: string }) {
  return (
    <div className={"md text-[15px] leading-relaxed text-slate-100" + (className ? " " + className : "")}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
          // react-markdown wraps fenced code in <pre><code>; render our CodeBlock instead and
          // flatten the default <pre> so we never nest <pre> inside <pre>.
          pre: ({ children }) => <>{children}</>,
          code: ({ node, className: cls, children, ...props }) => {
            const raw = String(children ?? "").replace(/\n$/, "");
            const lang = /language-(\w+)/.exec(cls || "")?.[1];
            const isBlock = !!lang || raw.includes("\n");
            if (!isBlock) {
              return (
                <code
                  className="rounded bg-slate-950/60 px-1 py-0.5 font-mono text-slate-200"
                  {...props}
                >
                  {children}
                </code>
              );
            }
            return <CodeBlock code={raw} lang={lang} />;
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
