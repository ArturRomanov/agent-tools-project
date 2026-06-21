"""Tests for the Web Search MCP server using FastMCP direct call methods."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

ROOT_DIR = Path(__file__).resolve().parents[3]
WEB_SEARCH_DIR = ROOT_DIR / "mcp-servers" / "web-search"
if str(WEB_SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_SEARCH_DIR))

from web_search_mcp.search import SearchResult  # noqa: E402
from web_search_mcp.server import mcp  # noqa: E402


@pytest.fixture
def fake_search_results():
    return [
        SearchResult(title="Result 1", url="https://example.com/1", snippet="Snippet 1"),
        SearchResult(title="Result 2", url="https://example.com/2", snippet="Snippet 2"),
        SearchResult(title="Result 3", url="https://example.com/3", snippet="Snippet 3"),
    ]


def _extract_text(result) -> str:
    """Extract text from call_tool result (tuple of content_blocks, metadata)."""
    content_blocks, _ = result
    for block in content_blocks:
        if hasattr(block, "text"):
            return block.text
    return str(result)


@pytest.mark.asyncio
async def test_web_search_tool_returns_json(fake_search_results):
    with patch(
        "web_search_mcp.server.do_web_search",
        new_callable=AsyncMock,
        return_value=fake_search_results,
    ):
        result = await mcp.call_tool("web_search", {"query": "test query", "max_results": 3})

    text = _extract_text(result)
    parsed = json.loads(text)
    assert "summary" in parsed
    assert "sources" in parsed
    assert len(parsed["sources"]) <= 3
    for source in parsed["sources"]:
        assert "title" in source
        assert "url" in source
        assert "snippet" in source


@pytest.mark.asyncio
async def test_web_search_tool_with_timelimit(fake_search_results):
    mock_search = AsyncMock(return_value=fake_search_results)
    with patch("web_search_mcp.server.do_web_search", mock_search):
        result = await mcp.call_tool(
            "web_search", {"query": "today news", "max_results": 2, "timelimit": "d"}
        )

    text = _extract_text(result)
    parsed = json.loads(text)
    assert "sources" in parsed

    # Verify timelimit was passed through
    call_args = mock_search.call_args
    assert call_args.kwargs.get("timelimit") == "d" or (
        len(call_args.args) >= 3 and call_args.args[2] == "d"
    )


@pytest.mark.asyncio
async def test_web_search_tool_empty_results():
    with patch(
        "web_search_mcp.server.do_web_search",
        new_callable=AsyncMock,
        return_value=[],
    ):
        result = await mcp.call_tool("web_search", {"query": "obscure query"})

    text = _extract_text(result)
    parsed = json.loads(text)
    assert parsed["sources"] == []


@pytest.mark.asyncio
async def test_web_search_tool_listed():
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "web_search" in tool_names


@pytest.mark.asyncio
async def test_web_search_tool_schema():
    tools = await mcp.list_tools()
    web_search_tool = next(t for t in tools if t.name == "web_search")
    schema = web_search_tool.inputSchema
    assert "query" in schema.get("properties", {})
    assert "query" in schema.get("required", [])
