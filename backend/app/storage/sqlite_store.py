from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.config.settings import Settings, get_settings


class SQLiteStore:
    def __init__(self, settings: Settings | None = None, db_path: str | None = None) -> None:
        self._settings = settings or get_settings()
        self._db_path = Path(db_path or self._settings.memory_sqlite_path)
        self._initialized = False

    @property
    def path(self) -> Path:
        return self._db_path

    @contextmanager
    def connection(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        if self._initialized:
            return

        with self.connection() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    user_scope TEXT NOT NULL,
                    memory_mode_default TEXT NOT NULL,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_name TEXT,
                    token_estimate INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                );

                CREATE TABLE IF NOT EXISTS session_summaries (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    covered_until_turn INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                );

                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    user_scope TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    salience REAL NOT NULL,
                    confidence REAL NOT NULL,
                    source_turn_id TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    is_pinned INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions (id),
                    FOREIGN KEY (source_turn_id) REFERENCES turns (id)
                );

                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_item_id TEXT PRIMARY KEY,
                    vector_id TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    FOREIGN KEY (memory_item_id) REFERENCES memory_items (id)
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    graph_node TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                );

                CREATE TABLE IF NOT EXISTS memory_access_log (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    request_id TEXT,
                    memory_item_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    selected INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (id),
                    FOREIGN KEY (memory_item_id) REFERENCES memory_items (id)
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session_sequence
                    ON turns(session_id, sequence_no);
                CREATE INDEX IF NOT EXISTS idx_turns_session_created
                    ON turns(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_memory_items_user_scope_created
                    ON memory_items(user_scope, created_at);
                CREATE INDEX IF NOT EXISTS idx_memory_items_salience_created
                    ON memory_items(salience, created_at);
                CREATE INDEX IF NOT EXISTS idx_memory_items_expires_at
                    ON memory_items(expires_at);
                """
            )

        self._initialized = True
