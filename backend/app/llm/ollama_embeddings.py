from __future__ import annotations

import logging
import time

from langchain_ollama import OllamaEmbeddings

from app.config.settings import Settings, get_settings
from app.observability.logging_utils import log_event, sanitize_text


class OllamaEmbeddingsError(RuntimeError):
    pass


logger = logging.getLogger(__name__)


class OllamaEmbeddingsService:
    def __init__(
        self,
        settings: Settings | None = None,
        client: OllamaEmbeddings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or OllamaEmbeddings(
            base_url=self._settings.ollama_base_url,
            model=self._settings.ollama_embedding_model,
        )

    async def embed_query(self, text: str) -> list[float]:
        started_at = time.perf_counter()
        cleaned = text.strip()
        if not cleaned:
            raise OllamaEmbeddingsError("Embedding text must not be blank")
        log_event(
            logger,
            "rag.embed.start",
            model=self._settings.ollama_embedding_model,
            input_type="query",
            text_excerpt=sanitize_text(
                cleaned,
                self._settings.log_payload_mode,
                self._settings.log_payload_max_chars,
            ),
        )
        try:
            vector = await self._client.aembed_query(cleaned)
            log_event(
                logger,
                "rag.embed.end",
                model=self._settings.ollama_embedding_model,
                input_type="query",
                vector_size=len(vector),
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            return vector
        except Exception as exc:
            log_event(
                logger,
                "rag.embed.error",
                model=self._settings.ollama_embedding_model,
                input_type="query",
                error_type=type(exc).__name__,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            raise OllamaEmbeddingsError("Failed to generate query embeddings") from exc

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        started_at = time.perf_counter()
        cleaned = [text.strip() for text in texts if text.strip()]
        if not cleaned:
            raise OllamaEmbeddingsError("Embedding texts must not be blank")
        log_event(
            logger,
            "rag.embed.start",
            model=self._settings.ollama_embedding_model,
            input_type="documents",
            document_count=len(cleaned),
            text_excerpt=sanitize_text(
                cleaned[0],
                self._settings.log_payload_mode,
                self._settings.log_payload_max_chars,
            ),
        )
        try:
            vectors = await self._client.aembed_documents(cleaned)
            log_event(
                logger,
                "rag.embed.end",
                model=self._settings.ollama_embedding_model,
                input_type="documents",
                document_count=len(cleaned),
                vector_size=(len(vectors[0]) if vectors else 0),
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            return vectors
        except Exception as exc:
            log_event(
                logger,
                "rag.embed.error",
                model=self._settings.ollama_embedding_model,
                input_type="documents",
                error_type=type(exc).__name__,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            raise OllamaEmbeddingsError("Failed to generate document embeddings") from exc
