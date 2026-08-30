# MCP server — stdio transport, spawned per-corpus by the tenant's LLM client.
# Build context = repo root.  Configure with BACKEND_URL / CORPUS_ID / MCP_TOKEN.
FROM python:3.12-slim
WORKDIR /app
COPY mcp_server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY mcp_server ./mcp_server
CMD ["python", "-m", "mcp_server.server"]
