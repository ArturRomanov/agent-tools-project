from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Lock

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config.settings import Settings, get_settings
from app.observability.logging_utils import log_event, sanitize_text
from app.rag.ingest.chunking import ChunkRecord


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
        distance_map = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclid": models.Distance.EUCLID,
        }
        raw_metric = self._settings.rag_distance_metric
        normalized_metric = str(raw_metric).strip().lower()
        resolved_metric = distance_map.get(normalized_metric, models.Distance.COSINE)
        fallback_used = normalized_metric not in distance_map
        try:
            client = self._get_client()
            if fallback_used:
                log_event(
                    logger,
                    "rag.store.distance_metric.fallback",
                    collection=self._collection_name,
                    distance_metric_raw=str(raw_metric),
                    distance_metric_resolved=resolved_metric.name.lower(),
                )
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
                distance_metric_raw=str(raw_metric),
                distance_metric_resolved=resolved_metric.name.lower(),
                distance_metric_fallback=fallback_used,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
        except Exception as exc:
            log_event(
                logger,
                "rag.store.ensure_collection.error",
                collection=self._collection_name,
                error_type=type(exc).__name__,
                distance_metric_raw=str(raw_metric),
                distance_metric_resolved=resolved_metric.name.lower(),
                error_message=sanitize_text(
                    str(exc),
                    self._settings.log_payload_mode,
                    self._settings.log_payload_max_chars,
                ),
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            raise RagStoreError("Failed to ensure Qdrant collection") from exc

    def upsert_chunks(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
        started_at = time.perf_counter()
        if len(chunks) != len(vectors):
            raise RagStoreError("Chunks and vectors length mismatch")
        try:
            client = self._get_client()
            points = []
            for chunk, vector in zip(chunks, vectors, strict=True):
                payload = {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "title": chunk.title,
                    "text": chunk.text,
                    "url": chunk.url,
                    "metadata": chunk.metadata,
                }
                points.append(
                    models.PointStruct(
                        id=chunk.point_id,
                        vector=vector,
                        payload=payload,
                    )
                )
            client.upsert(collection_name=self._collection_name, points=points)
            log_event(
                logger,
                "rag.store.upsert.end",
                collection=self._collection_name,
                chunk_count=len(chunks),
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
        except Exception as exc:
            sample_point_id = chunks[0].point_id if chunks else None
            log_event(
                logger,
                "rag.store.upsert.error",
                collection=self._collection_name,
                error_type=type(exc).__name__,
                chunk_count=len(chunks),
                sample_point_id=sample_point_id,
                error_message=sanitize_text(
                    str(exc),
                    self._settings.log_payload_mode,
                    self._settings.log_payload_max_chars,
                ),
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            raise RagStoreError("Failed to upsert chunks into Qdrant") from exc

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
