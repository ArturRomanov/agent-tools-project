from __future__ import annotations

import logging

from langchain_ollama import OllamaEmbeddings

from rag_mcp.config import RagSettings

logger = logging.getLogger(__name__)


class OllamaEmbeddingsError(RuntimeError):
    pass


class OllamaEmbeddingsService:
    def __init__(
        self,
        settings: RagSettings | None = None,
        client: OllamaEmbeddings | None = None,
    ) -> None:
        self._settings = settings or RagSettings()
        self._client = client or OllamaEmbeddings(
            base_url=self._settings.ollama_base_url,
            model=self._settings.ollama_embedding_model,
        )

    async def embed_query(self, text: str) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            raise OllamaEmbeddingsError("Embedding text must not be blank")
        try:
            return await self._client.aembed_query(cleaned)
        except Exception as exc:
            raise OllamaEmbeddingsError("Failed to generate query embeddings") from exc

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        cleaned = [text.strip() for text in texts if text.strip()]
        if not cleaned:
            raise OllamaEmbeddingsError("Embedding texts must not be blank")
        try:
            return await self._client.aembed_documents(cleaned)
        except Exception as exc:
            raise OllamaEmbeddingsError("Failed to generate document embeddings") from exc
