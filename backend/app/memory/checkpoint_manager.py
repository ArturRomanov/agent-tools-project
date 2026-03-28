from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

from app.storage import SQLiteStore


class CheckpointManager:
    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def save_checkpoint(
        self,
        session_id: str,
        run_id: str,
        graph_node: str,
        state: dict[str, object],
    ) -> str:
        checkpoint_id = str(uuid4())
        with self._store.connection() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (id, session_id, run_id, graph_node, state_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    session_id,
                    run_id,
                    graph_node,
                    json.dumps(state),
                    datetime.now(UTC).isoformat(),
                ),
            )
        return checkpoint_id

    def load_checkpoint(self, checkpoint_id: str) -> dict[str, object] | None:
        with self._store.connection() as conn:
            row = conn.execute(
                "SELECT state_json FROM checkpoints WHERE id = ?",
                (checkpoint_id,),
            ).fetchone()

        if row is None:
            return None
        raw = row["state_json"]
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
        return None
