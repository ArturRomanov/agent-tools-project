"""Tests for RAG routes via MCP path."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.main import app  # noqa: E402
from backend.app.mcp.client import MCPClient  # noqa: E402


def _make_mock_rag_client(
    ingest_response: dict | None = None,
    should_fail: bool = False,
) -> MCPClient:
    """Create a mock MCPClient that provides ingest_document tool."""
    from mcp.types import CallToolResult, TextContent

    client = MagicMock(spec=MCPClient)
    client.name = "rag"
    client.has_tool = MagicMock(side_effect=lambda name: name == "ingest_document")

    if should_fail:
        response_data = {"status": "error", "message": "provider failed"}
    elif ingest_response:
        response_data = ingest_response
    else:
        response_data = {
            "collection_name": "rag_documents",
            "indexed_chunks": 3,
            "status": "ok",
        }

    client.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response_data))],
            isError=False,
        )
    )
    return client


def test_post_rag_documents_happy_path_mcp() -> None:
    mock_client = _make_mock_rag_client()
    app.state.mcp_clients = [mock_client]

    client = TestClient(app)
    response = client.post(
        "/rag/documents",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"collection_name": "rag_documents"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["collection_name"] == "rag_documents"
    assert body["indexed_documents"] == 1
    assert body["indexed_chunks"] == 3

    app.state.mcp_clients = None


def test_post_rag_documents_happy_path_only_file_mcp() -> None:
    mock_client = _make_mock_rag_client()
    app.state.mcp_clients = [mock_client]

    client = TestClient(app)
    response = client.post(
        "/rag/documents",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["indexed_documents"] == 1

    app.state.mcp_clients = None


def test_post_rag_documents_invalid_file_type() -> None:
    mock_client = _make_mock_rag_client()
    app.state.mcp_clients = [mock_client]

    client = TestClient(app)
    response = client.post(
        "/rag/documents",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 422

    app.state.mcp_clients = None


def test_post_rag_documents_empty_file() -> None:
    mock_client = _make_mock_rag_client()
    app.state.mcp_clients = [mock_client]

    client = TestClient(app)
    response = client.post(
        "/rag/documents",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 422

    app.state.mcp_clients = None


def test_post_rag_documents_metadata_plain_text_coerced_mcp() -> None:
    mock_client = _make_mock_rag_client()
    app.state.mcp_clients = [mock_client]

    client = TestClient(app)
    response = client.post(
        "/rag/documents",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"metadata_json": "not-json"},
    )
    assert response.status_code == 200

    # Verify the metadata was passed to the MCP call
    call_args = mock_client.call_tool.call_args
    arguments = call_args[0][1]
    metadata = json.loads(arguments["metadata"])
    assert metadata == {"note": "not-json"}

    app.state.mcp_clients = None


def test_post_rag_documents_metadata_object_passthrough_mcp() -> None:
    mock_client = _make_mock_rag_client()
    app.state.mcp_clients = [mock_client]

    client = TestClient(app)
    response = client.post(
        "/rag/documents",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"metadata_json": '{"team":"platform"}'},
    )
    assert response.status_code == 200

    call_args = mock_client.call_tool.call_args
    arguments = call_args[0][1]
    metadata = json.loads(arguments["metadata"])
    assert metadata == {"team": "platform"}

    app.state.mcp_clients = None


def test_post_rag_documents_metadata_non_object_json_wrapped_mcp() -> None:
    mock_client = _make_mock_rag_client()
    app.state.mcp_clients = [mock_client]

    client = TestClient(app)
    response = client.post(
        "/rag/documents",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"metadata_json": '"abc"'},
    )
    assert response.status_code == 200

    call_args = mock_client.call_tool.call_args
    arguments = call_args[0][1]
    metadata = json.loads(arguments["metadata"])
    assert metadata == {"value": "abc"}

    app.state.mcp_clients = None


def test_post_rag_documents_provider_error_maps_to_422_mcp() -> None:
    mock_client = _make_mock_rag_client(should_fail=True)
    app.state.mcp_clients = [mock_client]

    client = TestClient(app)
    response = client.post(
        "/rag/documents",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert response.status_code == 422

    app.state.mcp_clients = None


def test_post_rag_documents_no_mcp_servers_returns_503() -> None:
    app.state.mcp_clients = None

    client = TestClient(app)
    response = client.post(
        "/rag/documents",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert response.status_code == 503

    app.state.mcp_clients = None


def test_post_rag_documents_collection_override_mcp() -> None:
    mock_client = _make_mock_rag_client(
        ingest_response={
            "collection_name": "custom_collection",
            "indexed_chunks": 5,
            "status": "ok",
        }
    )
    app.state.mcp_clients = [mock_client]

    client = TestClient(app)
    response = client.post(
        "/rag/documents",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"collection_name": "custom_collection", "url": "https://example.com/doc"},
    )
    assert response.status_code == 200
    assert response.json()["collection_name"] == "custom_collection"

    # Verify collection_name and url were passed to MCP
    call_args = mock_client.call_tool.call_args
    arguments = call_args[0][1]
    assert arguments["collection_name"] == "custom_collection"
    assert arguments["url"] == "https://example.com/doc"

    app.state.mcp_clients = None
