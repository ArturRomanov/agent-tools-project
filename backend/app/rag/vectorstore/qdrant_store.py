from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Lock

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config.settings import Settings, get_settings
from app.observability.logging_utils import log_event, sanitize_text


class RagStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetrievedChunk:
    title: str
    snippet: str
    url: str
    score: float
    document_id: str
    metadata: dict[str, str]


logger = logging.getLogger(__name__)
_LOCAL_CLIENTS_BY_PATH: dict[str, QdrantClient] = {}
_LOCAL_CLIENTS_LOCK = Lock()


class QdrantStore:
    def __init__(
        self,
        settings: Settings | None = None,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._collection_name = collection_name or self._settings.qdrant_collection_name

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def for_collection(self, collection_name: str) -> "QdrantStore":
        return QdrantStore(
            settings=self._settings,
            client=self._client,
            collection_name=collection_name,
        )

    def ensure_collection(self, vector_size: int) -> None:
        started_at = time.perf_counter()
        resolved_metric = models.Distance.COSINE
        try:
            client = self._get_client()
            exists = client.collection_exists(collection_name=self._collection_name)
            if not exists:
                client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=resolved_metric,
                    ),
                )
            log_event(
                logger,
                "rag.store.ensure_collection.end",
                collection=self._collection_name,
                vector_size=vector_size,
                collection_created=not exists,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
        except Exception as exc:
            log_event(
                logger,
                "rag.store.ensure_collection.error",
                collection=self._collection_name,
                error_type=type(exc).__name__,
                error_message=sanitize_text(
                    str(exc),
                    self._settings.log_payload_mode,
                    self._settings.log_payload_max_chars,
                ),
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            raise RagStoreError("Failed to ensure Qdrant collection") from exc

    def search(self, query_vector: list[float], limit: int) -> list[RetrievedChunk]:
        started_at = time.perf_counter()
        try:
            client = self._get_client()
            if hasattr(client, "query_points"):
                query_response = client.query_points(
                    collection_name=self._collection_name,
                    query=query_vector,
                    limit=limit,
                    with_payload=True,
                )
                points = query_response.points
            else:
                points = client.search(
                    collection_name=self._collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                )
            results: list[RetrievedChunk] = []
            for point in points:
                payload = point.payload or {}
                text = str(payload.get("text") or "").strip()
                if not text:
                    continue
                url = str(payload.get("url") or "").strip() or "rag://local"
                title = str(payload.get("title") or "Indexed document").strip()
                metadata = payload.get("metadata")
                results.append(
                    RetrievedChunk(
                        title=title,
                        snippet=text,
                        url=url,
                        score=float(point.score or 0.0),
                        document_id=str(payload.get("document_id") or ""),
                        metadata=metadata if isinstance(metadata, dict) else {},
                    )
                )
            log_event(
                logger,
                "rag.store.search.end",
                collection=self._collection_name,
                limit=limit,
                hit_count=len(results),
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            return results
        except Exception as exc:
            log_event(
                logger,
                "rag.store.search.error",
                collection=self._collection_name,
                error_type=type(exc).__name__,
                error_message=sanitize_text(
                    str(exc),
                    self._settings.log_payload_mode,
                    self._settings.log_payload_max_chars,
                ),
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            raise RagStoreError("Failed to query Qdrant collection") from exc

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            path = self._settings.qdrant_path
            with _LOCAL_CLIENTS_LOCK:
                cached_client = _LOCAL_CLIENTS_BY_PATH.get(path)
                if cached_client is None:
                    cached_client = QdrantClient(path=path)
                    _LOCAL_CLIENTS_BY_PATH[path] = cached_client
                self._client = cached_client
        return self._client
