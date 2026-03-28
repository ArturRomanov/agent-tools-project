from __future__ import annotations

from app.memory.models import SessionSummaryRecord
from app.storage import SQLiteStore


class SessionSummaryStore:
    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def latest(self, session_id: str) -> SessionSummaryRecord | None:
        with self._store.connection() as conn:
            row = conn.execute(
                """
                SELECT id, session_id, summary_text, covered_until_turn, created_at, quality_score
                FROM session_summaries
                WHERE session_id = ?
                ORDER BY covered_until_turn DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            return None
        return SessionSummaryRecord(
            id=row["id"],
            session_id=row["session_id"],
            summary_text=row["summary_text"],
            covered_until_turn=int(row["covered_until_turn"]),
            created_at=row["created_at"],
            quality_score=float(row["quality_score"]),
        )
