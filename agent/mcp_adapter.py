"""MCP → LangChain tool adapter.

Fetches the tool list from the MCP server and wraps each as a LangChain BaseTool.
"""

import json
import logging
from typing import Any

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agent.config import settings

log = logging.getLogger(__name__)


class MCPTool(BaseTool):
    """Wraps a single MCP tool as a LangChain tool."""

    name: str
    description: str
    mcp_server_url: str = Field(default_factory=lambda: settings.mcp_server_url)

    # Suppress schema validation for arbitrary args
    class Config:
        arbitrary_types_allowed = True

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("MCPTool is async-only; use _arun")

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        # LangChain ReAct passes tool input as a positional string; parse it into kwargs.
        if args:
            raw = args[0]
            if isinstance(raw, str):
                try:
                    kwargs = json.loads(raw)
                except json.JSONDecodeError:
                    kwargs = {"input": raw}
            elif isinstance(raw, dict):
                kwargs = raw

        payload = {"tool": self.name, "arguments": kwargs}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{self.mcp_server_url}/call", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return json.dumps(data, ensure_ascii=False)


async def load_mcp_tools() -> list[BaseTool]:
    """Fetch tool definitions from MCP server health endpoint and build adapters."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(f"{settings.mcp_server_url}/health")
            resp.raise_for_status()
            tool_names: list[str] = resp.json().get("tools", [])
        except Exception as exc:
            log.warning("Could not reach MCP server: %s — using empty tool list", exc)
            return []

    # Also fetch full schema from /tools if available
    tools = []
    for name in tool_names:
        tools.append(
            MCPTool(
                name=name,
                description=f"MCP tool: {name}",
                mcp_server_url=settings.mcp_server_url,
            )
        )
    log.info("Loaded %d MCP tools: %s", len(tools), tool_names)
    return tools
