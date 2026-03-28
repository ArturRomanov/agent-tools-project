from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.memory.models import TurnRecord
from app.storage import SQLiteStore


class TurnStore:
    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    @staticmethod
    def estimate_tokens(text: str) -> int:
        cleaned = text.strip()
        return max(1, len(cleaned) // 4) if cleaned else 0

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_name: str | None = None,
        archived: bool = False,
    ) -> str:
        now = datetime.now(UTC).isoformat()
        turn_id = str(uuid4())
        with self._store.connection() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(sequence_no), 0) AS sequence_no
                FROM turns
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            next_seq = int(row["sequence_no"]) + 1
            conn.execute(
                """
                INSERT INTO turns (
                    id, session_id, role, content, tool_name,
                    token_estimate, created_at, sequence_no, archived
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    session_id,
                    role,
                    content,
                    tool_name,
                    self.estimate_tokens(content),
                    now,
                    next_seq,
                    1 if archived else 0,
                ),
            )
        return turn_id

    def recent_turns(self, session_id: str, limit: int) -> list[TurnRecord]:
        with self._store.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, token_estimate,
                       sequence_no, created_at, archived
                FROM turns
                WHERE session_id = ? AND archived = 0
                ORDER BY sequence_no DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        results = [
            TurnRecord(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                token_estimate=int(row["token_estimate"]),
                sequence_no=int(row["sequence_no"]),
                created_at=row["created_at"],
                archived=bool(row["archived"]),
            )
            for row in rows
        ]
        return list(reversed(results))

    def oldest_active_turns(self, session_id: str, limit: int) -> list[TurnRecord]:
        with self._store.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, token_estimate,
                       sequence_no, created_at, archived
                FROM turns
                WHERE session_id = ? AND archived = 0
                ORDER BY sequence_no ASC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        return [
            TurnRecord(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                token_estimate=int(row["token_estimate"]),
                sequence_no=int(row["sequence_no"]),
                created_at=row["created_at"],
                archived=bool(row["archived"]),
            )
            for row in rows
        ]

    def archive_turns(self, turn_ids: list[str]) -> None:
        if not turn_ids:
            return
        placeholders = ",".join(["?" for _ in turn_ids])
        with self._store.connection() as conn:
            conn.execute(
                f"UPDATE turns SET archived = 1 WHERE id IN ({placeholders})",
                tuple(turn_ids),
            )
