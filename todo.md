# TODO

## High priority

- [x] **Windows start script** — `scripts/start.ps1` added.
- [ ] **MCP tool descriptions** — `load_mcp_tools()` sets `description="MCP tool: {name}"`. Fetch real descriptions + input schemas from the `/health` endpoint so the LLM knows exactly which arguments each tool accepts (reduces hallucinated args).
- [ ] **RAG not tested** — `agent/rag.py` has no test coverage. Add unit tests (mock ChromaDB + embeddings).
- [ ] **mcp_adapter not tested** — `agent/mcp_adapter.py` has no tests. Add tests for `MCPTool._arun` and `load_mcp_tools`.

## Medium priority

- [ ] **Persistent conversation history** — the agent currently receives only the last user message. Add multi-turn context by passing the full message list.
- [ ] **RAG ingestion entrypoint** — there is no CLI or script to add documents to ChromaDB. Add `scripts/ingest.py`.
- [ ] **Graceful MCP reconnect** — if the MCP server restarts, the agent holds stale tool handles. Add a retry / reload mechanism.
- [ ] **Token usage tracking** — the `/v1/chat/completions` response returns `0` for all token counts. Wire up actual counts from LangChain callbacks.

## Low priority / nice to have

- [ ] **Docker Compose** — optional alternative to the start scripts: single `docker-compose.yml` with Ollama sidecar + MCP server + agent API.
- [ ] **Structured logging** — switch from `logging.basicConfig` to JSON-structured logs for easier parsing.
- [ ] **Streaming from the LLM** — current streaming is simulated (word-by-word split). Wire real token streaming from `ChatOllama`.
- [ ] **Model hot-swap** — allow changing `OLLAMA_MODEL` without restarting the agent (reload on `/v1/models` selection).
