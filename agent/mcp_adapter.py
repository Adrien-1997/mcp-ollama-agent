"""MCP → LangChain tool adapter.

Fetches full tool schemas from the MCP server (/tools) and wraps each as a
typed LangChain StructuredTool so the agent knows exactly what args each tool
expects.
"""

import json
import logging
from typing import Any, Optional

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from agent.config import settings

log = logging.getLogger(__name__)

_TYPE_MAP = {"integer": int, "number": float, "boolean": bool, "string": str}


def _build_args_schema(input_schema: dict) -> type[BaseModel]:
    """Build a Pydantic model from a JSON Schema object so LangChain can
    validate and document tool arguments."""
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    fields: dict[str, Any] = {}
    for name, prop in properties.items():
        py_type = _TYPE_MAP.get(prop.get("type", "string"), str)
        desc = prop.get("description", "")
        if name in required:
            fields[name] = (py_type, FieldInfo(description=desc))
        else:
            default = prop.get("default", None)
            fields[name] = (Optional[py_type], FieldInfo(default=default, description=desc))
    return create_model("ArgsSchema", **fields)


def _make_tool(tool_def: dict, server_url: str) -> StructuredTool:
    name: str = tool_def["name"]
    description: str = tool_def.get("description", f"MCP tool: {name}")
    input_schema: dict = tool_def.get("inputSchema", {})
    args_schema = _build_args_schema(input_schema)

    async def _call(*args: Any, **kwargs: Any) -> str:
        # LangChain may pass input as a positional arg (raw string or dict)
        if args:
            raw = args[0]
            if isinstance(raw, str):
                try:
                    kwargs.update(json.loads(raw))
                except json.JSONDecodeError:
                    kwargs["input"] = raw
            elif isinstance(raw, dict):
                kwargs.update(raw)
        payload = {"tool": name, "arguments": kwargs}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{server_url}/call", json=payload)
            resp.raise_for_status()
            return json.dumps(resp.json(), ensure_ascii=False)

    return StructuredTool.from_function(
        coroutine=_call,
        name=name,
        description=description,
        args_schema=args_schema,
    )


async def load_mcp_tools() -> list[StructuredTool]:
    """Fetch full tool definitions from /tools and build typed LangChain tools."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(f"{settings.mcp_server_url}/tools")
            resp.raise_for_status()
            tool_defs: list[dict] = resp.json()
        except Exception as exc:
            log.warning("Could not reach MCP server: %s — using empty tool list", exc)
            return []

    tools = [_make_tool(t, settings.mcp_server_url) for t in tool_defs]
    log.info("Loaded %d MCP tools: %s", len(tools), [t.name for t in tools])
    return tools
