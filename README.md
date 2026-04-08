# MCP · Ollama · LangChain Agent

Local AI agent with a custom MCP server, LangChain ReAct loop, and built-in chat UI.

```
User (built-in chat UI · :8000)
      │
      ▼
OpenAI-compatible API  (FastAPI · :8000)
      │
      ▼
LangChain ReAct Agent  ──► Ollama (llama3.2 · :11434)
      │
      ▼
MCP Client
      │
      ▼
MCP Server (:8001)
  ├── web_search  (DuckDuckGo)
  ├── file_ops    (read / write local files)
  └── code_exec   (sandboxed Python exec)
      │
      ▼
ChromaDB  (local vector store · RAG)
```

## Stack

| Layer | Tech |
|---|---|
| LLM | Ollama · llama3.2 or mistral:7b |
| Orchestration | LangChain · ReAct agent |
| Tool protocol | MCP (model-context-protocol) |
| Vector store | ChromaDB + nomic-embed-text |
| API gateway | FastAPI (OpenAI-compatible) |
| UI | Built-in chat UI (served by FastAPI at :8000) |

## Quick start

### 1. Install prerequisites

| Prerequisite | Required | Notes |
|---|---|---|
| [Ollama](https://ollama.com/download) | Yes | Models are pulled automatically on first run |
| [Python 3.11+](https://python.org) | Yes | venv and deps are set up automatically by the start script |

### 2. Clone and run

**Windows (PowerShell)**
```powershell
git clone https://github.com/your-username/mcp-ollama-agent
cd mcp-ollama-agent
.\scripts\start.ps1
```

**Linux / macOS**
```bash
git clone https://github.com/your-username/mcp-ollama-agent
cd mcp-ollama-agent
./scripts/start.sh
```

The script handles everything: starts Ollama, pulls models on first run, creates the venv, installs deps, and launches all services.

Open **http://localhost:8000** for the chat UI, or **http://localhost:8000/docs** for the raw API.

## Project structure

```
mcp-ollama-agent/
├── mcp_server/
│   ├── main.py          # MCP server entry point (FastAPI + SSE)
│   ├── server.py        # MCP tool registration
│   ├── config.py        # MCP server settings (pydantic-settings)
│   └── tools/
│       ├── web_search.py
│       ├── file_ops.py
│       └── code_exec.py
├── agent/
│   ├── main.py          # OpenAI-compatible FastAPI gateway
│   ├── agent.py         # LangChain ReAct agent
│   ├── mcp_adapter.py   # MCP → LangChain tool adapter
│   ├── rag.py           # ChromaDB RAG helpers
│   └── config.py        # Agent settings (pydantic-settings)
├── scripts/
│   ├── start.sh         # Linux / macOS
│   └── start.ps1        # Windows (PowerShell)
├── tests/
│   ├── test_mcp_tools.py
│   └── test_agent.py
├── .env.example
├── pytest.ini
├── requirements.txt
└── README.md
```

## Benchmarks

| Model | Avg latency (tool call) | RAM |
|---|---|---|
| llama3.2:3b | ~1.2s | 4 GB |
| mistral:7b | ~2.8s | 6 GB |

*Tested on M2 MacBook Pro, measured over 50 tool-call turns.*

## Trade-offs

**stdio vs SSE transport** — stdio is simpler for local dev, SSE allows remote MCP servers. This repo defaults to SSE so the server is independently restartable.

**Local vs cloud LLM** — Ollama keeps everything offline and free. Swap `ChatOllama` for `ChatAnthropic` or `ChatOpenAI` in `agent/agent.py` with one line change.

**ReAct vs function-calling** — ReAct works with any model (including smaller local ones). If you switch to a model with native function-calling support, set `agent_type="openai-tools"` in `agent/agent.py`.
