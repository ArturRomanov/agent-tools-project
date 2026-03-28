from __future__ import annotations

import logging
import time

from app.config.settings import get_settings
from app.llm.ollama_embeddings import OllamaEmbeddingsError, OllamaEmbeddingsService
from app.observability.logging_utils import log_event, sanitize_text
from app.rag.vectorstore.qdrant_store import QdrantStore, RagStoreError
from app.schemas.chat import SourceItem


class RagRetrievalError(RuntimeError):
    pass


logger = logging.getLogger(__name__)
settings = get_settings()


class RagRetriever:
    def __init__(
        self,
        embeddings_service: OllamaEmbeddingsService | None = None,
        store: QdrantStore | None = None,
    ) -> None:
        self._embeddings_service = embeddings_service
        self._store = store

    async def retrieve(self, query: str, max_results: int) -> list[SourceItem]:
        started_at = time.perf_counter()
        cleaned = query.strip()
        if not cleaned:
            raise RagRetrievalError("RAG query must not be blank")

        log_event(
            logger,
            "rag.retrieve.start",
            max_results=max_results,
            query_excerpt=sanitize_text(
                cleaned,
                settings.log_payload_mode,
                settings.log_payload_max_chars,
            ),
        )
        try:
            query_vector = await self._get_embeddings_service().embed_query(cleaned)
            chunks = self._get_store().search(query_vector=query_vector, limit=max_results)
            sources = [
                SourceItem(title=chunk.title, url=chunk.url, snippet=chunk.snippet)
                for chunk in chunks
            ]
            log_event(
                logger,
                "rag.retrieve.end",
                max_results=max_results,
                source_count=len(sources),
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            return sources
        except (OllamaEmbeddingsError, RagStoreError) as exc:
            log_event(
                logger,
                "rag.retrieve.error",
                error_type=type(exc).__name__,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            raise RagRetrievalError("Failed to retrieve RAG sources") from exc
        except Exception as exc:
            log_event(
                logger,
                "rag.retrieve.error",
                error_type=type(exc).__name__,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            raise RagRetrievalError("Failed to retrieve RAG sources") from exc

    def _get_embeddings_service(self) -> OllamaEmbeddingsService:
        if self._embeddings_service is None:
            self._embeddings_service = OllamaEmbeddingsService()
        return self._embeddings_service

    def _get_store(self) -> QdrantStore:
        if self._store is None:
            self._store = QdrantStore()
        return self._store
