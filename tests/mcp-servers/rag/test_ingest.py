"""Tests for RAG MCP server ingest service (migrated from test_rag_ingest_service.py)."""

import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
RAG_DIR = ROOT_DIR / "mcp-servers" / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from rag_mcp.ingest.chunking import ChunkInput, build_chunks  # noqa: E402
from rag_mcp.ingest.service import RagIngestError, RagIngestService  # noqa: E402


class StubEmbeddingsService:
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class StubEmptyEmbeddingsService:
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return []


class StubStore:
    def __init__(self) -> None:
        self.collection_name = "rag_documents"
        self.vector_size = 0
        self.upserted = 0

    def ensure_collection(self, vector_size: int) -> None:
        self.vector_size = vector_size

    def upsert_chunks(self, chunks, vectors) -> None:
        self.upserted = len(chunks)

    def for_collection(self, collection_name: str):
        return self


def test_rag_ingest_service_happy_path() -> None:
    service = RagIngestService(
        embeddings_service=StubEmbeddingsService(),  # type: ignore[arg-type]
        store=StubStore(),  # type: ignore[arg-type]
    )

    result = asyncio.run(
        service.ingest(
            document_id="doc-1",
            title="Doc",
            text="Paragraph one.\n\nParagraph two.",
            metadata={"source": "unit-test"},
        )
    )

    assert result["collection_name"] == "rag_documents"
    assert result["indexed_chunks"] >= 1
    assert result["status"] == "ok"


def test_rag_ingest_service_empty_vectors_error() -> None:
    service = RagIngestService(
        embeddings_service=StubEmptyEmbeddingsService(),  # type: ignore[arg-type]
        store=StubStore(),  # type: ignore[arg-type]
    )

    try:
        asyncio.run(
            service.ingest(
                document_id="doc-1",
                title="Doc",
                text="Body text",
            )
        )
        assert False, "Expected RagIngestError"
    except RagIngestError:
        assert True


def test_chunk_point_id_is_uuid_and_deterministic() -> None:
    from uuid import UUID

    chunk_input = ChunkInput(
        document_id="doc-1",
        title="Doc",
        text="First paragraph.\n\nSecond paragraph.",
        metadata={"k": "v"},
    )
    chunks_a = build_chunks(chunk_input, chunk_size=800, chunk_overlap=120)
    chunks_b = build_chunks(chunk_input, chunk_size=800, chunk_overlap=120)

    assert len(chunks_a) >= 1
    assert chunks_a[0].point_id == chunks_b[0].point_id
    UUID(chunks_a[0].point_id)
