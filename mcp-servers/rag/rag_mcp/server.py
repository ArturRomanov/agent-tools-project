from __future__ import annotations

import base64
import json
import logging

from mcp.server.fastmcp import FastMCP

from rag_mcp.ingest.pdf_extract import PdfExtractionError, extract_pdf_document
from rag_mcp.ingest.service import RagIngestService
from rag_mcp.retriever import RagRetriever

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="rag",
    instructions="RAG tools for retrieving from and ingesting into the internal document store.",
    host="0.0.0.0",
    port=8002,
)

_retriever: RagRetriever | None = None
_ingest_service: RagIngestService | None = None


def _get_retriever() -> RagRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RagRetriever()
    return _retriever


def _get_ingest_service() -> RagIngestService:
    global _ingest_service
    if _ingest_service is None:
        _ingest_service = RagIngestService()
    return _ingest_service


@mcp.tool()
async def rag_retrieve(query: str, max_results: int = 5) -> str:
    """Retrieve relevant passages from indexed internal documents.

    Args:
        query: Natural-language question for the document store.
        max_results: Maximum number of results to return (1-20).

    Returns:
        JSON string with summary and sources [{title, url, snippet}].
    """
    sources = await _get_retriever().retrieve(query, max_results=max_results)
    summary = f"Retrieved {len(sources)} RAG sources for query: {query}"
    return json.dumps({"summary": summary, "sources": sources})


@mcp.tool()
async def ingest_document(
    content_base64: str,
    filename: str,
    url: str | None = None,
    collection_name: str | None = None,
    metadata: str | None = None,
) -> str:
    """Ingest a PDF document into the RAG index.

    Args:
        content_base64: Base64-encoded PDF file content.
        filename: Original filename.
        url: Optional source URL for attribution.
        collection_name: Target Qdrant collection (uses default if omitted).
        metadata: Optional JSON string with additional metadata.

    Returns:
        JSON string with chunk_count and status.
    """
    try:
        file_bytes = base64.b64decode(content_base64)
    except Exception as exc:
        return json.dumps({"status": "error", "message": f"Invalid base64 content: {exc}"})

    user_metadata = None
    if metadata:
        try:
            user_metadata = json.loads(metadata)
            if not isinstance(user_metadata, dict):
                user_metadata = {"value": user_metadata}
        except json.JSONDecodeError:
            user_metadata = {"note": metadata}

    try:
        doc = extract_pdf_document(
            file_bytes=file_bytes,
            filename=filename,
            url=url,
            metadata=user_metadata,
        )
    except PdfExtractionError as exc:
        return json.dumps({"status": "error", "message": str(exc)})

    result = await _get_ingest_service().ingest(
        document_id=doc["id"],
        title=doc["title"],
        text=doc["text"],
        url=doc.get("url"),
        metadata={k: str(v) for k, v in (doc.get("metadata") or {}).items()},
        collection_name=collection_name,
    )
    return json.dumps(result)
