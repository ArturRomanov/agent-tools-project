"""Tests for the RAG MCP server using FastMCP direct call methods."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT_DIR = Path(__file__).resolve().parents[3]
RAG_DIR = ROOT_DIR / "mcp-servers" / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from rag_mcp.server import mcp  # noqa: E402


@pytest.fixture
def fake_retrieve_results():
    return [
        {"title": "Doc A", "url": "rag://local/a", "snippet": "Content from doc A"},
        {"title": "Doc B", "url": "rag://local/b", "snippet": "Content from doc B"},
    ]


def _extract_text(result) -> str:
    """Extract text from call_tool result (tuple of content_blocks, metadata)."""
    content_blocks, _ = result
    for block in content_blocks:
        if hasattr(block, "text"):
            return block.text
    return str(result)


@pytest.mark.asyncio
async def test_rag_retrieve_tool_returns_json(fake_retrieve_results):
    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(return_value=fake_retrieve_results)

    with patch("rag_mcp.server._get_retriever", return_value=mock_retriever):
        result = await mcp.call_tool(
            "rag_retrieve", {"query": "test question", "max_results": 5}
        )

    text = _extract_text(result)
    parsed = json.loads(text)
    assert "summary" in parsed
    assert "sources" in parsed
    assert len(parsed["sources"]) == 2
    for source in parsed["sources"]:
        assert "title" in source
        assert "url" in source
        assert "snippet" in source


@pytest.mark.asyncio
async def test_rag_retrieve_empty_results():
    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(return_value=[])

    with patch("rag_mcp.server._get_retriever", return_value=mock_retriever):
        result = await mcp.call_tool("rag_retrieve", {"query": "no results"})

    text = _extract_text(result)
    parsed = json.loads(text)
    assert parsed["sources"] == []


@pytest.mark.asyncio
async def test_ingest_document_tool_returns_json():
    import base64

    mock_ingest = MagicMock()
    mock_ingest.ingest = AsyncMock(
        return_value={"collection_name": "rag_documents", "indexed_chunks": 3, "status": "ok"}
    )

    fake_pdf = b"%PDF-1.4 fake content"
    content_b64 = base64.b64encode(fake_pdf).decode()

    mock_extract = MagicMock(
        return_value={
            "id": "abc123",
            "title": "Test Doc",
            "text": "Extracted text",
            "url": None,
            "metadata": {"source_type": "pdf_upload"},
        }
    )

    with (
        patch("rag_mcp.server._get_ingest_service", return_value=mock_ingest),
        patch("rag_mcp.server.extract_pdf_document", mock_extract),
    ):
        result = await mcp.call_tool(
            "ingest_document",
            {"content_base64": content_b64, "filename": "test.pdf"},
        )

    text = _extract_text(result)
    parsed = json.loads(text)
    assert parsed["status"] == "ok"
    assert parsed["indexed_chunks"] == 3


@pytest.mark.asyncio
async def test_ingest_document_invalid_base64():
    result = await mcp.call_tool(
        "ingest_document",
        {"content_base64": "!!!invalid!!!", "filename": "bad.pdf"},
    )

    text = _extract_text(result)
    parsed = json.loads(text)
    assert parsed["status"] == "error"


@pytest.mark.asyncio
async def test_tools_listed():
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "rag_retrieve" in tool_names
    assert "ingest_document" in tool_names


@pytest.mark.asyncio
async def test_rag_retrieve_tool_schema():
    tools = await mcp.list_tools()
    rag_tool = next(t for t in tools if t.name == "rag_retrieve")
    schema = rag_tool.inputSchema
    assert "query" in schema.get("properties", {})
    assert "query" in schema.get("required", [])
