"""Tests for MCPToolRegistry and MCPToolProxy."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.mcp.client import MCPClient  # noqa: E402
from backend.app.mcp.tool_registry import MCPToolProxy, MCPToolRegistry  # noqa: E402
from backend.app.schemas.tools import ToolSpec  # noqa: E402


def _make_mock_client(name: str, tool_specs: list[ToolSpec]) -> MCPClient:
    """Create a mock MCPClient with given specs."""
    client = MagicMock(spec=MCPClient)
    client.name = name
    client.specs.return_value = tool_specs
    return client


def test_registry_aggregates_tools_from_multiple_clients():
    web_specs = [ToolSpec(name="web_search", description="Search web", input_hint="query")]
    rag_specs = [
        ToolSpec(name="rag_retrieve", description="Retrieve docs", input_hint="query"),
        ToolSpec(name="ingest_document", description="Ingest doc", input_hint="base64 content"),
    ]

    web_client = _make_mock_client("web-search", web_specs)
    rag_client = _make_mock_client("rag", rag_specs)

    registry = MCPToolRegistry(clients=[web_client, rag_client])

    all_specs = registry.specs()
    assert len(all_specs) == 3
    names = {s.name for s in all_specs}
    assert names == {"web_search", "rag_retrieve", "ingest_document"}


def test_registry_get_returns_correct_proxy():
    web_specs = [ToolSpec(name="web_search", description="Search web", input_hint="query")]
    rag_specs = [ToolSpec(name="rag_retrieve", description="Retrieve docs", input_hint="query")]

    web_client = _make_mock_client("web-search", web_specs)
    rag_client = _make_mock_client("rag", rag_specs)

    registry = MCPToolRegistry(clients=[web_client, rag_client])

    web_proxy = registry.get("web_search")
    assert web_proxy is not None
    assert web_proxy.name == "web_search"
    assert web_proxy._client is web_client

    rag_proxy = registry.get("rag_retrieve")
    assert rag_proxy is not None
    assert rag_proxy.name == "rag_retrieve"
    assert rag_proxy._client is rag_client


def test_registry_get_returns_none_for_unknown_tool():
    web_specs = [ToolSpec(name="web_search", description="Search web", input_hint="query")]
    client = _make_mock_client("web-search", web_specs)
    registry = MCPToolRegistry(clients=[client])

    assert registry.get("nonexistent") is None


def test_registry_first_tool_name():
    specs = [ToolSpec(name="web_search", description="Search web", input_hint="query")]
    client = _make_mock_client("web-search", specs)
    registry = MCPToolRegistry(clients=[client])

    assert registry.first_tool_name() == "web_search"


def test_registry_first_tool_name_empty():
    registry = MCPToolRegistry(clients=[])
    assert registry.first_tool_name() is None


def test_proxy_spec():
    proxy = MCPToolProxy(
        name="web_search",
        description="Search web",
        input_hint="query",
        _client=MagicMock(),
    )
    spec = proxy.spec()
    assert spec.name == "web_search"
    assert spec.description == "Search web"
    assert spec.input_hint == "query"


async def _run_proxy_test():
    """Test MCPToolProxy.run() parsing MCP JSON response."""
    from mcp.types import CallToolResult, TextContent

    mock_client = MagicMock(spec=MCPClient)
    mock_client.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "summary": "Found 2 results",
                            "sources": [
                                {
                                    "title": "Doc 1",
                                    "url": "https://example.com/1",
                                    "snippet": "Snippet 1",
                                },
                                {
                                    "title": "Doc 2",
                                    "url": "https://example.com/2",
                                    "snippet": "Snippet 2",
                                },
                            ],
                        }
                    ),
                )
            ],
            isError=False,
        )
    )

    proxy = MCPToolProxy(
        name="web_search",
        description="Search web",
        input_hint="query",
        _client=mock_client,
    )

    result = await proxy.run("test query", max_results=2)

    assert result.summary == "Found 2 results"
    assert len(result.sources) == 2
    assert result.sources[0].title == "Doc 1"
    assert result.sources[0].url == "https://example.com/1"

    mock_client.call_tool.assert_called_once_with(
        "web_search", {"query": "test query", "max_results": 2}
    )


def test_proxy_run():
    import asyncio

    asyncio.run(_run_proxy_test())


async def _run_proxy_with_timelimit_test():
    """Test MCPToolProxy.run() passes timelimit when provided."""
    from mcp.types import CallToolResult, TextContent

    mock_client = MagicMock(spec=MCPClient)
    mock_client.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps({"summary": "ok", "sources": []}),
                )
            ],
            isError=False,
        )
    )

    proxy = MCPToolProxy(
        name="web_search",
        description="Search web",
        input_hint="query",
        _client=mock_client,
    )

    await proxy.run("test", max_results=5, timelimit="d")

    mock_client.call_tool.assert_called_once_with(
        "web_search", {"query": "test", "max_results": 5, "timelimit": "d"}
    )


def test_proxy_run_with_timelimit():
    import asyncio

    asyncio.run(_run_proxy_with_timelimit_test())
