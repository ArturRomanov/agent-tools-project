from __future__ import annotations

import base64
import json
import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.observability.logging_utils import log_event
from app.schemas.rag import RagIngestResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/rag/documents", response_model=RagIngestResponse)
async def ingest_documents(
    request: Request,
    file: UploadFile = File(...),
    collection_name: str | None = Form(default=None),
    url: str | None = Form(default=None),
    metadata_json: str | None = Form(default=None),
) -> RagIngestResponse:
    content_type = (file.content_type or "").lower()
    filename = file.filename or "document.pdf"
    if content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF uploads are supported")
    if collection_name is not None and not collection_name.strip():
        raise HTTPException(status_code=422, detail="collection_name must not be blank")
    normalized_collection_name = collection_name.strip() if collection_name else None

    user_metadata: dict[str, Any] | None = None
    metadata_raw = (metadata_json or "").strip()
    if metadata_raw:
        try:
            parsed = json.loads(metadata_raw)
            if isinstance(parsed, dict):
                user_metadata = parsed
            else:
                user_metadata = {"value": parsed}
                log_event(
                    logger,
                    "rag.ingest.metadata.coerced",
                    mode="json_value_wrap",
                    route="/rag/documents",
                )
        except json.JSONDecodeError:
            user_metadata = {"note": metadata_raw}
            log_event(
                logger,
                "rag.ingest.metadata.coerced",
                mode="plain_text",
                route="/rag/documents",
            )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    mcp_clients = getattr(request.app.state, "mcp_clients", None)
    if not mcp_clients:
        raise HTTPException(
            status_code=503, detail="No MCP servers connected; ingestion unavailable"
        )

    return await _ingest_via_mcp(
        mcp_clients, file_bytes, filename, url, normalized_collection_name, user_metadata
    )


async def _ingest_via_mcp(
    mcp_clients: list,
    file_bytes: bytes,
    filename: str,
    url: str | None,
    collection_name: str | None,
    metadata: dict[str, Any] | None,
) -> RagIngestResponse:
    """Route ingestion through the RAG MCP server's ingest_document tool."""
    from app.mcp.client import MCPClient
    from mcp.types import TextContent

    rag_client: MCPClient | None = None
    for client in mcp_clients:
        if client.has_tool("ingest_document"):
            rag_client = client
            break

    if rag_client is None:
        raise HTTPException(
            status_code=502, detail="No MCP server provides the ingest_document tool"
        )

    content_b64 = base64.b64encode(file_bytes).decode("ascii")
    arguments: dict[str, Any] = {
        "content_base64": content_b64,
        "filename": filename,
    }
    if url:
        arguments["url"] = url
    if collection_name:
        arguments["collection_name"] = collection_name
    if metadata:
        arguments["metadata"] = json.dumps(metadata)

    result = await rag_client.call_tool("ingest_document", arguments)

    text_parts = []
    for content in result.content:
        if isinstance(content, TextContent):
            text_parts.append(content.text)
        elif hasattr(content, "text"):
            text_parts.append(content.text)

    raw_text = "\n".join(text_parts)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"Invalid response from MCP server: {raw_text}")

    if parsed.get("status") == "error":
        raise HTTPException(status_code=422, detail=parsed.get("message", "Ingestion failed"))

    return RagIngestResponse(
        collection_name=parsed.get("collection_name", "unknown"),
        indexed_documents=1,
        indexed_chunks=parsed.get("indexed_chunks", 0),
    )
