"""Tests for agent/mcp_adapter.py — schema building and tool dispatch."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# _build_args_schema
# ---------------------------------------------------------------------------

def test_build_args_schema_required_field():
    from agent.mcp_adapter import _build_args_schema

    schema = _build_args_schema({
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    })
    fields = schema.model_fields
    assert "query" in fields
    assert fields["query"].is_required()
    assert fields["query"].description == "Search query"


def test_build_args_schema_optional_field_has_default():
    from agent.mcp_adapter import _build_args_schema

    schema = _build_args_schema({
        "type": "object",
        "properties": {
            "max_results": {"type": "integer", "default": 5, "description": "Number of results"},
        },
    })
    fields = schema.model_fields
    assert "max_results" in fields
    assert fields["max_results"].default == 5


def test_build_args_schema_type_mapping():
    from agent.mcp_adapter import _build_args_schema
    import typing

    schema = _build_args_schema({
        "type": "object",
        "properties": {
            "flag": {"type": "boolean"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "label": {"type": "string"},
            "unknown": {"type": "exotic"},
        },
    })
    fields = schema.model_fields
    # Optional wraps non-required fields; get the inner type
    def inner(f):
        ann = f.annotation
        args = getattr(ann, "__args__", None)
        return args[0] if args else ann

    assert inner(fields["flag"]) is bool
    assert inner(fields["count"]) is int
    assert inner(fields["ratio"]) is float
    assert inner(fields["label"]) is str
    assert inner(fields["unknown"]) is str  # unknown falls back to str


def test_build_args_schema_empty():
    from agent.mcp_adapter import _build_args_schema

    schema = _build_args_schema({})
    assert schema.model_fields == {}


# ---------------------------------------------------------------------------
# load_mcp_tools
# ---------------------------------------------------------------------------

def _mock_client(get_return=None, get_side_effect=None):
    """Return a patched httpx.AsyncClient that yields a mock instance."""
    mock_instance = AsyncMock()
    if get_side_effect:
        mock_instance.get.side_effect = get_side_effect
    else:
        mock_resp = MagicMock()
        mock_resp.json.return_value = get_return
        mock_resp.raise_for_status = MagicMock()
        mock_instance.get.return_value = mock_resp
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_instance)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


@pytest.mark.asyncio
async def test_load_mcp_tools_builds_tools():
    fake_defs = [
        {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "code_exec",
            "description": "Execute a Python snippet.",
            "inputSchema": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    ]

    with patch("agent.mcp_adapter.httpx.AsyncClient", return_value=_mock_client(get_return=fake_defs)):
        from agent.mcp_adapter import load_mcp_tools
        tools = await load_mcp_tools()

    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {"web_search", "code_exec"}

    ws = next(t for t in tools if t.name == "web_search")
    assert ws.description == "Search the web using DuckDuckGo."
    assert "query" in ws.args_schema.model_fields


@pytest.mark.asyncio
async def test_load_mcp_tools_server_unreachable():
    with patch("agent.mcp_adapter.httpx.AsyncClient",
               return_value=_mock_client(get_side_effect=Exception("Connection refused"))):
        from agent.mcp_adapter import load_mcp_tools
        tools = await load_mcp_tools()

    assert tools == []


# ---------------------------------------------------------------------------
# _make_tool dispatch
# ---------------------------------------------------------------------------

def _mock_post_client(response_body: dict):
    mock_instance = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = response_body
    mock_resp.raise_for_status = MagicMock()
    mock_instance.post.return_value = mock_resp
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_instance)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx, mock_instance


@pytest.mark.asyncio
async def test_make_tool_posts_to_call_endpoint():
    tool_def = {
        "name": "code_exec",
        "description": "Execute Python",
        "inputSchema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code"}},
            "required": ["code"],
        },
    }
    ctx, mock_instance = _mock_post_client({"stdout": "hello\n", "returncode": 0})

    with patch("agent.mcp_adapter.httpx.AsyncClient", return_value=ctx):
        from agent.mcp_adapter import _make_tool
        tool = _make_tool(tool_def, "http://localhost:8001")
        result = await tool.ainvoke({"code": "print('hello')"})

    mock_instance.post.assert_called_once()
    url, = mock_instance.post.call_args.args
    assert url == "http://localhost:8001/call"
    payload = mock_instance.post.call_args.kwargs["json"]
    assert payload["tool"] == "code_exec"
    assert payload["arguments"]["code"] == "print('hello')"


@pytest.mark.asyncio
async def test_make_tool_uses_description_from_server():
    tool_def = {
        "name": "file_read",
        "description": "Read a file from the workspace.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
    from agent.mcp_adapter import _make_tool
    tool = _make_tool(tool_def, "http://localhost:8001")
    assert tool.description == "Read a file from the workspace."


@pytest.mark.asyncio
async def test_make_tool_fallback_description():
    tool_def = {
        "name": "mystery_tool",
        "inputSchema": {},
    }
    from agent.mcp_adapter import _make_tool
    tool = _make_tool(tool_def, "http://localhost:8001")
    assert tool.description == "MCP tool: mystery_tool"
