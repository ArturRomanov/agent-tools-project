from __future__ import annotations

import logging
import time

from app.config.settings import get_settings
from app.llm.ollama_embeddings import OllamaEmbeddingsError, OllamaEmbeddingsService
from app.observability.logging_utils import log_event
from app.rag.ingest.chunking import ChunkInput, build_chunks
from app.rag.vectorstore.qdrant_store import QdrantStore, RagStoreError
from app.schemas.rag import RagDocumentInput, RagIngestResponse


class RagIngestError(RuntimeError):
    pass


logger = logging.getLogger(__name__)
settings = get_settings()


class RagIngestService:
    def __init__(
        self,
        embeddings_service: OllamaEmbeddingsService | None = None,
        store: QdrantStore | None = None,
    ) -> None:
        self._embeddings_service = embeddings_service
        self._store = store

    async def ingest(
        self,
        documents: list[RagDocumentInput],
        collection_name: str | None = None,
    ) -> RagIngestResponse:
        started_at = time.perf_counter()
        base_store = self._get_store()
        requested_collection = (collection_name or "").strip()
        store = (
            base_store.for_collection(requested_collection)
            if requested_collection and requested_collection != base_store.collection_name
            else base_store
        )

        log_event(
            logger,
            "rag.ingest.start",
            collection=store.collection_name,
            document_count=len(documents),
        )
        try:
            chunks = []
            for document in documents:
                chunk_input = ChunkInput(
                    document_id=document.id,
                    title=document.title,
                    text=document.text,
                    url=document.url,
                    metadata={k: str(v) for k, v in (document.metadata or {}).items()},
                )
                chunks.extend(
                    build_chunks(
                        chunk_input,
                        chunk_size=settings.rag_chunk_size,
                        chunk_overlap=settings.rag_chunk_overlap,
                    )
                )

            if not chunks:
                raise RagIngestError("No chunks produced from input documents")

            vectors = await self._get_embeddings_service().embed_documents(
                [chunk.text for chunk in chunks]
            )
            vector_size = len(vectors[0]) if vectors else 0
            if vector_size == 0:
                raise RagIngestError("Embedding service returned empty vectors")

            store.ensure_collection(vector_size=vector_size)
            store.upsert_chunks(chunks=chunks, vectors=vectors)
            response = RagIngestResponse(
                collection_name=store.collection_name,
                indexed_documents=len(documents),
                indexed_chunks=len(chunks),
            )
            log_event(
                logger,
                "rag.ingest.end",
                collection=store.collection_name,
                indexed_documents=response.indexed_documents,
                indexed_chunks=response.indexed_chunks,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            return response
        except (OllamaEmbeddingsError, RagStoreError, RagIngestError) as exc:
            log_event(
                logger,
                "rag.ingest.error",
                collection=store.collection_name,
                error_type=type(exc).__name__,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            if isinstance(exc, RagIngestError):
                raise
            raise RagIngestError("Failed to ingest documents into RAG index") from exc

    def _get_embeddings_service(self) -> OllamaEmbeddingsService:
        if self._embeddings_service is None:
            self._embeddings_service = OllamaEmbeddingsService()
        return self._embeddings_service

    def _get_store(self) -> QdrantStore:
        if self._store is None:
            self._store = QdrantStore()
        return self._store
