"""MCP server — exposes one onboarded corpus as a tool the tenant's own LLM can
call (Claude, etc.). Built on the official `mcp` SDK (FastMCP). Stateless: it is
configured per-corpus by env and proxies to the control plane's token-authenticated
/mcp/{corpus}/query endpoint, so it carries no secrets beyond the corpus token.

Run (e.g. from a Claude Desktop mcpServers entry):
    BACKEND_URL=http://localhost:8000 \
    CORPUS_ID=<corpus id> MCP_TOKEN=<token> \
    python -m mcp_server.server
"""
import os

import httpx
from mcp.server.fastmcp import FastMCP

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
CORPUS_ID = os.environ.get("CORPUS_ID", "")
MCP_TOKEN = os.environ.get("MCP_TOKEN", "")

mcp = FastMCP("cartridge-corpus")


@mcp.tool()
def query_corpus(question: str, k: int = 3) -> str:
    """Ask a natural-language question against the onboarded corpus.

    Returns a grounded answer synthesized from the corpus's trained KV cartridges,
    plus the source documents used. Use this whenever the user asks about content
    that lives in this organization's knowledge base.
    """
    if not CORPUS_ID or not MCP_TOKEN:
        return "Error: this MCP server needs CORPUS_ID and MCP_TOKEN environment variables."
    resp = httpx.post(
        f"{BACKEND_URL}/mcp/{CORPUS_ID}/query",
        headers={"X-MCP-Token": MCP_TOKEN},
        json={"question": question, "k": k},
        timeout=120.0,
    )
    if resp.status_code == 401:
        return "Error: invalid MCP token for this corpus."
    resp.raise_for_status()
    data = resp.json()
    sources = ", ".join(data.get("used_docs", [])) or "—"
    return f"{data['answer']}\n\n(sources: {sources})"


if __name__ == "__main__":
    mcp.run()  # stdio transport
