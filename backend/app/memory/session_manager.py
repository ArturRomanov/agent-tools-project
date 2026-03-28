from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.storage import SQLiteStore


class SessionManager:
    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def get_or_create(
        self,
        session_id: str | None,
        user_scope: str,
        memory_mode_default: str,
    ) -> str:
        resolved_session_id = session_id or str(uuid4())
        now = datetime.now(UTC).isoformat()
        with self._store.connection() as conn:
            existing = conn.execute(
                "SELECT id FROM sessions WHERE id = ?",
                (resolved_session_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO sessions (
                        id, created_at, updated_at, user_scope, memory_mode_default, status
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resolved_session_id,
                        now,
                        now,
                        user_scope,
                        memory_mode_default,
                        "active",
                    ),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (now, resolved_session_id),
                )
        return resolved_session_id
