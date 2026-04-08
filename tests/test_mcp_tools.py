"""Tests for MCP tools."""

import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock


# --- web_search ---

@pytest.mark.asyncio
async def test_web_search_returns_results():
    mock_results = [
        {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
        {"title": "Result 2", "href": "https://example.com/2", "body": "Snippet 2"},
    ]
    with patch("mcp_server.tools.web_search.DDGS") as MockDDGS:
        instance = MockDDGS.return_value.__enter__.return_value
        instance.text.return_value = mock_results

        from mcp_server.tools.web_search import web_search
        result = await web_search("python MCP", max_results=2)

    assert result["query"] == "python MCP"
    assert len(result["results"]) == 2
    assert result["results"][0]["title"] == "Result 1"


# --- file_ops ---

@pytest.mark.asyncio
async def test_file_write_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr("mcp_server.tools.file_ops.WORKSPACE", tmp_path)

    from mcp_server.tools.file_ops import file_write, file_read

    write_result = await file_write("test.txt", "hello world")
    assert write_result["bytes_written"] == len("hello world".encode())

    read_result = await file_read("test.txt")
    assert read_result["content"] == "hello world"


@pytest.mark.asyncio
async def test_file_read_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("mcp_server.tools.file_ops.WORKSPACE", tmp_path)

    from mcp_server.tools.file_ops import file_read
    result = await file_read("nonexistent.txt")
    assert "error" in result


@pytest.mark.asyncio
async def test_file_ops_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr("mcp_server.tools.file_ops.WORKSPACE", tmp_path)

    from mcp_server.tools.file_ops import file_read
    with pytest.raises(ValueError, match="Path traversal denied"):
        await file_read("../../etc/passwd")


# --- code_exec ---

@pytest.mark.asyncio
async def test_code_exec_simple():
    from mcp_server.tools.code_exec import code_exec
    result = await code_exec("print('hello')")
    assert result["returncode"] == 0
    assert "hello" in result["stdout"]


@pytest.mark.asyncio
async def test_code_exec_stderr():
    from mcp_server.tools.code_exec import code_exec
    result = await code_exec("import sys; sys.stderr.write('err')")
    assert "err" in result["stderr"]


@pytest.mark.asyncio
async def test_code_exec_timeout():
    from mcp_server.tools.code_exec import code_exec
    result = await code_exec("import time; time.sleep(99)", timeout=1)
    assert "error" in result
    assert "timed out" in result["error"]
