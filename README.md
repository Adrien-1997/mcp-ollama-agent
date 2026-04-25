# mcp-ollama-agent

**A fully local AI agent — private, offline, and production-grade.**

Combines Ollama, LangChain, MCP (Model Context Protocol), ChromaDB RAG, and Open WebUI into a single deployable stack. The agent reasons step by step, calls tools when needed, and searches a knowledge base built from 12 real open-source AI project repositories — no cloud, no API keys, no data leaving your machine.

```
User (Open WebUI · :3000)
        │
        ▼
  FastAPI  ──  OpenAI-compatible  POST /v1/chat/completions
               Swagger UI         GET  /docs
        │
        ▼
  LangChain ReAct Agent ──► Ollama (qwen2.5:3b · :11434)
        │
        ▼
  MCP Client  ──►  MCP Server (:8001)
                     ├── web_search            DuckDuckGo, live results
                     ├── file_read/write/list  sandboxed workspace
                     ├── code_exec             Python subprocess
                     └── query_knowledge_base  ChromaDB · 3,700+ chunks
                                │
                                ▼
                          ChromaDB + nomic-embed-text
                          (Ollama, llama.cpp, LocalAI,
                           Open WebUI, LiteLLM, Tabby …)
```

## Why this project

Most "local AI" demos connect a chat UI to a model and stop there. This project goes further:

- **Explicit tool use via MCP** — tools are registered with typed JSON schemas; the agent decides when to call them.
- **RAG over real documentation** — not toy data. 3,726 chunks ingested from 12 production GitHub repos covering the local AI ecosystem.
- **OpenAI-compatible API** — swap Open WebUI for any client that speaks `/v1/chat/completions`.
- **Fully reproducible** — one script starts every service. No manual setup.

## Stack

| Layer | Technology |
|---|---|
| LLM runtime | [Ollama](https://ollama.com) · qwen2.5:3b (default) · qwen2.5:7b (high quality) |
| Agent orchestration | [LangChain](https://python.langchain.com) · ReAct loop |
| Tool protocol | [MCP](https://modelcontextprotocol.io) (SSE transport) |
| Vector store | [ChromaDB](https://www.trychroma.com) + nomic-embed-text embeddings |
| API gateway | [FastAPI](https://fastapi.tiangolo.com) (OpenAI-compatible) |
| Chat UI | [Open WebUI](https://openwebui.com) (Docker · proxied to agent) |

## Quick start

### Prerequisites

| Tool | Notes |
|---|---|
| [Ollama](https://ollama.com/download) | Models pulled automatically on first run |
| Python 3.11+ | venv and deps created automatically |
| [Docker](https://docs.docker.com/get-docker/) | Required for Open WebUI |

### Run

**Linux / macOS**
```bash
git clone https://github.com/Adrien-1997/mcp-ollama-agent
cd mcp-ollama-agent
./scripts/start.sh
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/Adrien-1997/mcp-ollama-agent
cd mcp-ollama-agent
.\scripts\start.ps1
```

The start script handles everything: creates the venv, installs deps, starts Ollama, pulls models on first run, launches the MCP server, agent API, and Open WebUI.

| Service | URL |
|---|---|
| Chat UI (Open WebUI) | http://localhost:3000 |
| Agent API (Swagger) | http://localhost:8000/docs |
| MCP Server | http://localhost:8001/health |

### Build the knowledge base (first time)

```bash
source .venv/bin/activate
python scripts/ingest.py --reset
```

This fetches documentation from 12 GitHub repos, embeds ~3,700 chunks into ChromaDB using `nomic-embed-text`, and persists them to `.chroma/`. Takes ~20–30 min on CPU.

```
=== Done: 3726 chunks from 164 files across 13 repos ===
```

## Usage

### Chat (Open WebUI)

Open **http://localhost:3000**, select the `local-agent` model, and start chatting. The agent automatically routes knowledge base questions, web searches, and code execution through the right tools.

**Example — knowledge base query:**

> **User:** What quantization formats does llama.cpp support, and what's the difference between Q4_K_M and Q8_0?
>
> **Agent:** LLaMA supports several quantization formats, including Q4_K_M and Q8_0.
>
> - **Q4_K_M** — 4-bit integer quantization with mixed precision on attention/feed-forward layers. Low memory usage, small accuracy loss.
> - **Q8_0** — 8-bit floating-point quantization. Higher precision than Q4_K_M but roughly twice the memory footprint.
>
> Q4_K_M is the recommended default for most machines. Use Q8_0 when accuracy matters more than RAM budget.

The agent retrieved this answer from the ingested llama.cpp and LocalAI documentation via `query_knowledge_base`.

### Agent API

The agent exposes an OpenAI-compatible endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-agent",
    "messages": [{"role": "user", "content": "What quantization formats does llama.cpp support?"}],
    "stream": false
  }'
```

### Query the knowledge base directly

The knowledge base is also accessible as a standalone REST tool — useful for testing retrieval or building your own clients:

```bash
curl http://localhost:8001/call \
  -X POST -H "Content-Type: application/json" \
  -d '{
    "tool": "query_knowledge_base",
    "arguments": {
      "query": "llama.cpp GGUF quantization formats Q4 Q8",
      "k": 3
    }
  }'
```

```json
{
  "query": "llama.cpp GGUF quantization formats Q4 Q8",
  "results": [
    {
      "content": "| Backend | Quantization Types |\n|---------|-------------------|\n| llama-cpp-quantization | q2_k, q3_k_s, q3_k_m, q4_0, q4_k_s, q4_k_m, q5_0, q5_k_s, q5_k_m, q6_k, q8_0 |",
      "source": "github/mudler/LocalAI/docs/content/features/quantization.md",
      "url": "https://github.com/mudler/LocalAI/blob/master/docs/content/features/quantization.md"
    },
    {
      "content": "ik_llama.cpp ships additional quantization types (IQK quants), custom quantization mixes, Multi-head Latent Attention (MLA) for superior CPU and hybrid GPU/CPU performance.",
      "source": "github/mudler/LocalAI/docs/content/features/text-generation.md",
      "url": "https://github.com/mudler/LocalAI/blob/master/docs/content/features/text-generation.md"
    },
    {
      "content": "LLGuidance is a library for constrained decoding for Large Language Models — structured outputs and constrained sampling.",
      "source": "github/ggerganov/llama.cpp/docs/llguidance.md",
      "url": "https://github.com/ggerganov/llama.cpp/blob/master/docs/llguidance.md"
    }
  ]
}
```

### Inspect available tools

```bash
curl http://localhost:8001/tools | python3 -m json.tool
```

Returns the full JSON Schema for each tool — the same payload the agent reads to build its typed tool registry.

## Knowledge base

### Covered projects

| Project | What's indexed |
|---|---|
| [Ollama](https://github.com/ollama/ollama) | API reference, development guide, model examples |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | Build docs, backends (CUDA, SYCL, Vulkan), quantization, multimodal |
| [LocalAI](https://github.com/mudler/LocalAI) | Full feature docs: embeddings, image gen, audio, MCP, distributed inference |
| [Open WebUI](https://github.com/open-webui/open-webui) | README, security |
| [Jan](https://github.com/janhq/jan) | README, docs |
| [text-generation-webui](https://github.com/oobabooga/text-generation-webui) | All 18 doc pages: parameters, training, OpenAI API, extensions |
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) | README, bare metal setup |
| [PrivateGPT](https://github.com/zylon-ai/private-gpt) | README |
| [LiteLLM](https://github.com/BerriAI/litellm) | Architecture, proxy setup |
| [GPT4All](https://github.com/nomic-ai/gpt4all) | README |
| [Continue](https://github.com/continuedev/continue) | README, docs |
| [Tabby](https://github.com/TabbyML/tabby) | Model spec, README |

### Re-ingesting

```bash
# Wipe and re-ingest everything
python scripts/ingest.py --reset

# Preview what would be ingested (no DB writes)
python scripts/ingest.py --dry-run
```

## Project structure

```
mcp-ollama-agent/
├── agent/
│   ├── main.py          # FastAPI gateway (OpenAI-compatible API)
│   ├── agent.py         # LangChain ReAct agent + system prompt
│   ├── mcp_adapter.py   # MCP → LangChain StructuredTool adapter
│   ├── rag.py           # ChromaDB helpers (get, add, reset, search)
│   └── config.py        # Settings via pydantic-settings / .env
├── mcp_server/
│   ├── main.py          # MCP server entry point (FastAPI + SSE)
│   ├── server.py        # Tool registration and dispatch
│   ├── config.py        # MCP server settings
│   └── tools/
│       ├── web_search.py       # DuckDuckGo search
│       ├── file_ops.py         # Sandboxed file read/write/list
│       ├── code_exec.py        # Python subprocess with timeout
│       └── kb_search.py        # ChromaDB knowledge base query
├── scripts/
│   ├── start.sh         # Linux / macOS launcher
│   ├── start.ps1        # Windows launcher
│   └── ingest.py        # GitHub → ChromaDB ingestion pipeline
├── tests/
│   ├── test_agent.py
│   ├── test_mcp_adapter.py
│   └── test_mcp_tools.py
├── .env.example
├── requirements.txt
└── docker-compose.yml
```

## Configuration

Copy `.env.example` to `.env` to override defaults:

```env
# Model
OLLAMA_MODEL=qwen2.5:3b        # default — good tool use, ~2 GB RAM
OLLAMA_EMBED_MODEL=nomic-embed-text

# Ports
AGENT_API_PORT=8000
MCP_SERVER_PORT=8001

# Storage
CHROMA_PERSIST_DIR=.chroma
FILE_OPS_ROOT=./workspace
```

## Development

```bash
source .venv/bin/activate

# Run all tests
pytest -v

# Run a single suite
pytest tests/test_mcp_tools.py -v
```

Tests mock all external dependencies (Ollama, ChromaDB, DuckDuckGo). 22 tests, no live services required.

## Model recommendations

| Model | RAM | Tool use | Best for |
|---|---|---|---|
| `qwen2.5:1.5b` | ~1 GB | Limited | Fast answers, conversational only |
| `qwen2.5:3b` | ~2 GB | Good | **Default** — reliable tool calls, fits most machines |
| `qwen2.5:7b` | ~5 GB | Excellent | Best quality, needs 8+ GB free RAM |

To change model: set `OLLAMA_MODEL=qwen2.5:7b` in `.env` and re-run `start.sh` (model is pulled automatically).

## Design notes

**MCP over stdio vs SSE** — This project uses SSE transport so the MCP server runs as an independent process, restartable without restarting the agent.

**Explicit RAG via tool, not silent injection** — Early versions silently prepended retrieved chunks into every prompt. The current design exposes `query_knowledge_base` as a proper MCP tool: the agent decides when retrieval is useful, which is more transparent and avoids polluting prompts on irrelevant queries.

**OpenAI-compatible API** — Swap `ChatOllama` for `ChatAnthropic` or `ChatOpenAI` in `agent/agent.py` with one line. Any client that speaks `/v1/chat/completions` (Continue, Open WebUI, LangChain) works out of the box.

**ReAct vs function-calling** — ReAct works with any model, including smaller local ones that lack native function-calling. If you upgrade to a model with tool-call support, change `create_react_agent` to `create_openai_tools_agent` in `agent/agent.py`.
