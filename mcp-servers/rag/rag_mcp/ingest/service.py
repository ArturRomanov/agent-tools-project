from __future__ import annotations

import logging

from rag_mcp.config import RagSettings
from rag_mcp.embeddings import OllamaEmbeddingsError, OllamaEmbeddingsService
from rag_mcp.ingest.chunking import ChunkInput, build_chunks
from rag_mcp.vectorstore import QdrantStore, RagStoreError

logger = logging.getLogger(__name__)


class RagIngestError(RuntimeError):
    pass


class RagIngestService:
    def __init__(
        self,
        settings: RagSettings | None = None,
        embeddings_service: OllamaEmbeddingsService | None = None,
        store: QdrantStore | None = None,
    ) -> None:
        self._settings = settings or RagSettings()
        self._embeddings_service = embeddings_service or OllamaEmbeddingsService(self._settings)
        self._store = store or QdrantStore(self._settings)

    async def ingest(
        self,
        document_id: str,
        title: str,
        text: str,
        url: str | None = None,
        metadata: dict[str, str] | None = None,
        collection_name: str | None = None,
    ) -> dict[str, object]:
        base_store = self._store
        requested_collection = (collection_name or "").strip()
        store = (
            base_store.for_collection(requested_collection)
            if requested_collection and requested_collection != base_store.collection_name
            else base_store
        )

        try:
            chunk_input = ChunkInput(
                document_id=document_id,
                title=title,
                text=text,
                url=url,
                metadata=metadata or {},
            )
            chunks = build_chunks(
                chunk_input,
                chunk_size=self._settings.rag_chunk_size,
                chunk_overlap=self._settings.rag_chunk_overlap,
            )

            if not chunks:
                raise RagIngestError("No chunks produced from input document")

            vectors = await self._embeddings_service.embed_documents(
                [chunk.text for chunk in chunks]
            )
            vector_size = len(vectors[0]) if vectors else 0
            if vector_size == 0:
                raise RagIngestError("Embedding service returned empty vectors")

            store.ensure_collection(vector_size=vector_size)
            store.upsert_chunks(chunks=chunks, vectors=vectors)
            return {
                "collection_name": store.collection_name,
                "indexed_chunks": len(chunks),
                "status": "ok",
            }
        except (OllamaEmbeddingsError, RagStoreError, RagIngestError) as exc:
            if isinstance(exc, RagIngestError):
                raise
            raise RagIngestError("Failed to ingest document into RAG index") from exc
