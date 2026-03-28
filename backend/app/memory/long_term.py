from __future__ import annotations

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid4, uuid5

from qdrant_client.http import models

from app.config.settings import Settings, get_settings
from app.llm.ollama_embeddings import OllamaEmbeddingsService
from app.memory.models import MemoryRecord
from app.rag.vectorstore.qdrant_store import QdrantStore
from app.storage import SQLiteStore


class LongTermMemoryStore:
    def __init__(
        self,
        sqlite_store: SQLiteStore,
        embeddings_service: OllamaEmbeddingsService | None = None,
        qdrant_store: QdrantStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._sqlite = sqlite_store
        self._embeddings = embeddings_service or OllamaEmbeddingsService(self._settings)
        self._qdrant = (qdrant_store or QdrantStore(settings=self._settings)).for_collection(
            self._settings.memory_qdrant_collection
        )

    def list_recent(self, user_scope: str, limit: int) -> list[MemoryRecord]:
        with self._sqlite.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, user_scope, type, content, salience,
                       confidence, created_at, is_pinned
                FROM memory_items
                WHERE user_scope = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_scope, limit),
            ).fetchall()
        return [
            MemoryRecord(
                id=row["id"],
                session_id=row["session_id"],
                user_scope=row["user_scope"],
                memory_type=row["type"],
                content=row["content"],
                salience=float(row["salience"]),
                confidence=float(row["confidence"]),
                created_at=row["created_at"],
                is_pinned=bool(row["is_pinned"]),
            )
            for row in rows
        ]

    async def store_memory(
        self,
        session_id: str | None,
        user_scope: str,
        memory_type: str,
        content: str,
        salience: float,
        confidence: float,
        source_turn_id: str | None,
    ) -> str:
        memory_item_id = str(uuid4())
        now = datetime.now(UTC).isoformat()
        with self._sqlite.connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_items (
                    id, session_id, user_scope, type, content, salience, confidence,
                    source_turn_id, created_at, expires_at, is_pinned
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_item_id,
                    session_id,
                    user_scope,
                    memory_type,
                    content,
                    salience,
                    confidence,
                    source_turn_id,
                    now,
                    None,
                    0,
                ),
            )

        vector = await self._embeddings.embed_query(content)
        self._qdrant.ensure_collection(vector_size=len(vector))
        vector_id = str(uuid5(NAMESPACE_URL, f"memory:{memory_item_id}"))
        payload = {
            "title": f"Memory:{memory_type}",
            "text": content,
            "url": f"memory://{memory_item_id}",
            "document_id": memory_item_id,
            "metadata": {
                "memory_item_id": memory_item_id,
                "session_id": session_id or "",
                "user_scope": user_scope,
                "salience": f"{salience:.4f}",
                "confidence": f"{confidence:.4f}",
                "created_at": now,
            },
        }
        self._qdrant._get_client().upsert(  # noqa: SLF001
            collection_name=self._qdrant.collection_name,
            points=[
                models.PointStruct(
                    id=vector_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

        with self._sqlite.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_embeddings (
                    memory_item_id, vector_id, embedding_model, indexed_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    memory_item_id,
                    vector_id,
                    self._settings.ollama_embedding_model,
                    now,
                ),
            )

        return memory_item_id

    async def retrieve(
        self,
        query: str,
        user_scope: str,
        limit: int,
    ) -> list[tuple[MemoryRecord, float]]:
        vector = await self._embeddings.embed_query(query)
        results = self._qdrant.search(query_vector=vector, limit=limit * 2)
        scored: list[tuple[MemoryRecord, float]] = []

        memory_ids: list[str] = []
        score_by_id: dict[str, float] = {}
        for item in results:
            memory_item_id = str(item.metadata.get("memory_item_id") or "").strip()
            if not memory_item_id:
                continue
            memory_ids.append(memory_item_id)
            score_by_id[memory_item_id] = float(item.score)

        if not memory_ids:
            return []

        placeholders = ",".join(["?" for _ in memory_ids])
        with self._sqlite.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT id, session_id, user_scope, type, content, salience,
                       confidence, created_at, is_pinned
                FROM memory_items
                WHERE id IN ({placeholders}) AND user_scope = ?
                """,
                tuple(memory_ids) + (user_scope,),
            ).fetchall()

        now = datetime.now(UTC)
        for row in rows:
            memory = MemoryRecord(
                id=row["id"],
                session_id=row["session_id"],
                user_scope=row["user_scope"],
                memory_type=row["type"],
                content=row["content"],
                salience=float(row["salience"]),
                confidence=float(row["confidence"]),
                created_at=row["created_at"],
                is_pinned=bool(row["is_pinned"]),
            )
            semantic = score_by_id.get(memory.id, 0.0)
            try:
                created_at = datetime.fromisoformat(memory.created_at)
                age_days = max((now - created_at).days, 0)
            except Exception:
                age_days = 0
            recency = 1.0 / (1.0 + (age_days / 30.0))
            hybrid = (semantic * 0.6) + (memory.salience * 0.25) + (recency * 0.15)
            scored.append((memory, hybrid))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def log_access(
        self,
        session_id: str,
        request_id: str | None,
        memory_item_id: str,
        score: float,
        selected: bool,
    ) -> None:
        with self._sqlite.connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_access_log (
                    id, session_id, request_id, memory_item_id, score, selected, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    session_id,
                    request_id,
                    memory_item_id,
                    score,
                    1 if selected else 0,
                    datetime.now(UTC).isoformat(),
                ),
            )
