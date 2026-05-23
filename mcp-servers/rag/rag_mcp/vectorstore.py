from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag_mcp.config import RagSettings
from rag_mcp.ingest.chunking import ChunkRecord

logger = logging.getLogger(__name__)


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


_LOCAL_CLIENTS_BY_PATH: dict[str, QdrantClient] = {}
_LOCAL_CLIENTS_LOCK = Lock()


class QdrantStore:
    def __init__(
        self,
        settings: RagSettings | None = None,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._settings = settings or RagSettings()
        self._client = client
        self._collection_name = collection_name or self._settings.qdrant_collection_name

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def for_collection(self, collection_name: str) -> QdrantStore:
        return QdrantStore(
            settings=self._settings,
            client=self._client,
            collection_name=collection_name,
        )

    def ensure_collection(self, vector_size: int) -> None:
        distance_map = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclid": models.Distance.EUCLID,
        }
        normalized_metric = str(self._settings.rag_distance_metric).strip().lower()
        resolved_metric = distance_map.get(normalized_metric, models.Distance.COSINE)
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
        except Exception as exc:
            raise RagStoreError("Failed to ensure Qdrant collection") from exc

    def upsert_chunks(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
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
        except Exception as exc:
            raise RagStoreError("Failed to upsert chunks into Qdrant") from exc

    def search(self, query_vector: list[float], limit: int) -> list[RetrievedChunk]:
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
            return results
        except Exception as exc:
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
