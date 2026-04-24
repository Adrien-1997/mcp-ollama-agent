"""OpenAI-compatible FastAPI gateway.

Exposes /v1/chat/completions so Open WebUI (or any OpenAI client) can talk
to the local LangChain agent.

Run with:  python -m agent.main
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel

from agent.agent import build_agent, run_agent
from agent.config import settings

logging.basicConfig(level=settings.log_level)
log = logging.getLogger(__name__)

# Build agent once at startup
_agent_executor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent_executor
    log.info("Building agent...")
    _agent_executor = await build_agent()
    log.info("Agent ready")
    yield


app = FastAPI(title="Local Agent API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# OpenAI-compatible schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "local-agent"
    messages: list[Message]
    stream: bool = False
    temperature: float = 0.1


def _openai_response(content: str, model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _stream_response(content: str, model: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    # Stream word by word for a realistic effect
    words = content.split(" ")
    for i, word in enumerate(words):
        delta = word + (" " if i < len(words) - 1 else "")
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.01)

    # Final chunk
    final = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "local-agent", "object": "model", "created": 0, "owned_by": "local"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if _agent_executor is None:
        raise HTTPException(503, "Agent not ready")

    # Use the last user message as the query
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(400, "No user message found")

    query = user_messages[-1].content
    log.info("Query: %s", query[:120])

    try:
        result = await run_agent(query, _agent_executor)
        content = result["output"]
    except Exception as exc:
        log.exception("Agent error")
        content = f"Error: {exc}"

    if req.stream:
        return StreamingResponse(
            _stream_response(content, req.model),
            media_type="text/event-stream",
        )

    return _openai_response(content, req.model)


@app.get("/health")
async def health():
    return {"status": "ok", "agent": _agent_executor is not None}


def main():
    uvicorn.run(
        "agent.main:app",
        host=settings.agent_api_host,
        port=settings.agent_api_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
