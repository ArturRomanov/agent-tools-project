from __future__ import annotations

import logging

from rag_mcp.config import RagSettings
from rag_mcp.embeddings import OllamaEmbeddingsError, OllamaEmbeddingsService
from rag_mcp.vectorstore import QdrantStore, RagStoreError

logger = logging.getLogger(__name__)


class RagRetrievalError(RuntimeError):
    pass


class RagRetriever:
    def __init__(
        self,
        settings: RagSettings | None = None,
        embeddings_service: OllamaEmbeddingsService | None = None,
        store: QdrantStore | None = None,
    ) -> None:
        self._settings = settings or RagSettings()
        self._embeddings_service = embeddings_service or OllamaEmbeddingsService(self._settings)
        self._store = store or QdrantStore(self._settings)

    async def retrieve(self, query: str, max_results: int) -> list[dict[str, str]]:
        cleaned = query.strip()
        if not cleaned:
            raise RagRetrievalError("RAG query must not be blank")

        try:
            query_vector = await self._embeddings_service.embed_query(cleaned)
            chunks = self._store.search(query_vector=query_vector, limit=max_results)
            return [
                {"title": chunk.title, "url": chunk.url, "snippet": chunk.snippet}
                for chunk in chunks
            ]
        except (OllamaEmbeddingsError, RagStoreError) as exc:
            raise RagRetrievalError("Failed to retrieve RAG sources") from exc
        except Exception as exc:
            raise RagRetrievalError("Failed to retrieve RAG sources") from exc
