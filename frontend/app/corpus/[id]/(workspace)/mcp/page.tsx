"use client";
// MCP Server section: expose this corpus as a tool for the tenant's own LLM.
// (Moved verbatim out of the old single-page corpus view.)
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { api, API_URL } from "@/lib/api";
import { Card, CardBody, CardHeader } from "@/components/ui";

export default function McpPage() {
  const { id } = useParams() as { id: string };
  const [corpus, setCorpus] = useState<any>(null);

  useEffect(() => {
    api.getCorpus(id).then(setCorpus).catch(() => setCorpus(null));
  }, [id]);

  if (!corpus) return <p className="text-sm text-slate-400">Loading…</p>;
  const mcpUrl = `${API_URL}/mcp/${id}/query`;
  const serverName = corpus.name.replace(/[^a-zA-Z0-9]+/g, "-").toLowerCase();

  return (
    <Card>
      <CardHeader>
        <h2 className="font-medium">Expose as MCP server</h2>
        <p className="text-xs text-slate-400">
          Point any MCP-capable LLM (e.g. Claude) at this corpus as a tool — it queries the
          expert at a fraction of the token cost of ingesting the documents directly.
        </p>
      </CardHeader>
      <CardBody className="space-y-2 text-sm">
        {!corpus.mcp_token ? (
          <p className="text-slate-400">Train the corpus to generate an MCP access token.</p>
        ) : (
          <>
            <div>
              <span className="text-slate-400">Endpoint:</span>{" "}
              <code data-testid="mcp-url" className="rounded bg-slate-800 px-1">{mcpUrl}</code>
            </div>
            <div>
              <span className="text-slate-400">Token:</span>{" "}
              <code data-testid="mcp-token" className="rounded bg-slate-800 px-1">{corpus.mcp_token}</code>
            </div>
            <pre className="overflow-x-auto rounded-md bg-slate-950 p-3 text-xs text-slate-100">{`{
  "mcpServers": {
    "${serverName}": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "env": {
        "BACKEND_URL": "${API_URL}",
        "CORPUS_ID": "${id}",
        "MCP_TOKEN": "${corpus.mcp_token}"
      }
    }
  }
}`}</pre>
          </>
        )}
      </CardBody>
    </Card>
  );
}
