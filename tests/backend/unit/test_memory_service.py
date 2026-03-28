import asyncio
import sqlite3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config.settings import Settings  # noqa: E402
from backend.app.memory.models import MemoryRecord  # noqa: E402
from backend.app.memory.service import MemoryService  # noqa: E402
from backend.app.memory.summarizer import SessionSummarizer  # noqa: E402
from backend.app.storage import SQLiteStore  # noqa: E402


class StubLongTermStore:
    def __init__(self) -> None:
        self.stored: list[str] = []

    async def retrieve(self, query: str, user_scope: str, limit: int):
        if "project" not in query.lower():
            return []
        return [
            (
                MemoryRecord(
                    id="mem-1",
                    session_id=None,
                    user_scope=user_scope,
                    memory_type="fact",
                    content="User project is agent-tools-project",
                    salience=0.8,
                    confidence=0.9,
                    created_at="2026-01-01T00:00:00+00:00",
                    is_pinned=False,
                ),
                0.92,
            )
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
        memory_id = f"stored-{len(self.stored) + 1}"
        self.stored.append(content)
        return memory_id

    def log_access(
        self,
        session_id: str,
        request_id: str | None,
        memory_item_id: str,
        score: float,
        selected: bool,
    ) -> None:
        return None


class StubSummarizer:
    def __init__(self, summary: str = "Facts: stub") -> None:
        self.summary = summary

    async def summarize_turns(self, turns):  # type: ignore[no-untyped-def]
        return self.summary


class FailingOllamaService:
    async def generate(self, request):  # type: ignore[no-untyped-def]
        raise RuntimeError("llm has closed")


def _build_service(
    tmp_path: Path, context_limit_tokens: int = 6000
) -> tuple[MemoryService, Path]:
    db_path = tmp_path / "memory.db"
    settings = Settings(
        memory_sqlite_path=str(db_path),
        memory_context_limit_tokens=context_limit_tokens,
        memory_keep_recent_turns=2,
        memory_recent_turn_window=10,
        memory_top_k=5,
    )
    sqlite_store = SQLiteStore(settings=settings)
    service = MemoryService(
        settings=settings,
        sqlite_store=sqlite_store,
        long_term_store=StubLongTermStore(),
        summarizer=StubSummarizer(),
    )
    return service, db_path


def test_memory_mode_off_does_not_persist(tmp_path: Path) -> None:
    service, db_path = _build_service(tmp_path)

    context = asyncio.run(
        service.prepare_context(
            query="hello", session_id=None, memory_mode="off", checkpoint_id=None
        )
    )
    persistence = asyncio.run(
        service.persist_after_run(
            session_id=context.session_id,
            user_query="hello",
            assistant_answer="world",
            memory_mode="off",
            user_scope="default",
            graph_state={"k": "v"},
        )
    )

    assert context.session_id
    assert persistence.checkpoint_id is None

    conn = sqlite3.connect(db_path)
    session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    turn_count = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    conn.close()
    assert session_count == 0
    assert turn_count == 0


def test_multi_turn_session_memory_context(tmp_path: Path) -> None:
    service, _ = _build_service(tmp_path)

    context1 = asyncio.run(
        service.prepare_context(
            query="first question",
            session_id=None,
            memory_mode="session",
            checkpoint_id=None,
        )
    )
    asyncio.run(
        service.persist_after_run(
            session_id=context1.session_id,
            user_query="first question",
            assistant_answer="first answer",
            memory_mode="session",
            user_scope="default",
            graph_state={"step": 1},
        )
    )

    context2 = asyncio.run(
        service.prepare_context(
            query="second question",
            session_id=context1.session_id,
            memory_mode="session",
            checkpoint_id=None,
        )
    )

    assert "first question" in context2.context_text
    assert "first answer" in context2.context_text


def test_long_term_memory_retrieval_carryover(tmp_path: Path) -> None:
    service, _ = _build_service(tmp_path)

    context = asyncio.run(
        service.prepare_context(
            query="What is my project?",
            session_id=None,
            memory_mode="long_term",
            checkpoint_id=None,
            user_scope="u1",
        )
    )

    assert "Relevant long-term memory" in context.context_text
    assert "agent-tools-project" in context.context_text


def test_checkpoint_save_and_load(tmp_path: Path) -> None:
    service, _ = _build_service(tmp_path)

    context = asyncio.run(
        service.prepare_context(
            query="hello", session_id=None, memory_mode="session", checkpoint_id=None
        )
    )
    persistence = asyncio.run(
        service.persist_after_run(
            session_id=context.session_id,
            user_query="hello",
            assistant_answer="world",
            memory_mode="session",
            user_scope="default",
            graph_state={"final_answer": "world", "source_count": 1},
        )
    )

    loaded = asyncio.run(
        service.prepare_context(
            query="resume",
            session_id=context.session_id,
            memory_mode="session",
            checkpoint_id=persistence.checkpoint_id,
        )
    )

    assert persistence.checkpoint_id is not None
    assert loaded.checkpoint_state == {"final_answer": "world", "source_count": 1}


def test_summarization_triggers_when_budget_exceeded(tmp_path: Path) -> None:
    service, db_path = _build_service(tmp_path, context_limit_tokens=1000)

    context = asyncio.run(
        service.prepare_context(
            query="start",
            session_id=None,
            memory_mode="session",
            checkpoint_id=None,
        )
    )
    for _ in range(20):
        result = asyncio.run(
            service.persist_after_run(
                session_id=context.session_id,
                user_query=(
                    " ".join(
                        [
                            "Please remember this very long sentence for budgeting behavior",
                            "and keep all details about architecture, testing, interfaces, and edge cases.",
                        ]
                        * 40
                    )
                ),
                assistant_answer=(
                    " ".join(
                        [
                            "Acknowledged and stored with details about architecture and tests.",
                            "Tracking edge cases, follow-up tasks, and open questions for future turns.",
                        ]
                        * 30
                    )
                ),
                memory_mode="session",
                user_scope="default",
                graph_state={"ok": True},
            )
        )

    assert result.summarized is True

    conn = sqlite3.connect(db_path)
    summary_count = conn.execute("SELECT COUNT(*) FROM session_summaries").fetchone()[0]
    archived_count = conn.execute(
        "SELECT COUNT(*) FROM turns WHERE archived = 1"
    ).fetchone()[0]
    conn.close()

    assert summary_count >= 1
    assert archived_count >= 1


def test_llm_summarizer_fallback_when_ollama_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    settings = Settings(
        memory_sqlite_path=str(db_path), memory_context_limit_tokens=1000
    )
    sqlite_store = SQLiteStore(settings=settings)
    summarizer = SessionSummarizer(llm_service=FailingOllamaService())  # type: ignore[arg-type]
    service = MemoryService(
        settings=settings,
        sqlite_store=sqlite_store,
        long_term_store=StubLongTermStore(),
        summarizer=summarizer,
    )

    context = asyncio.run(
        service.prepare_context(
            query="start",
            session_id=None,
            memory_mode="session",
            checkpoint_id=None,
        )
    )
    for _ in range(20):
        result = asyncio.run(
            service.persist_after_run(
                session_id=context.session_id,
                user_query=(
                    " ".join(
                        [
                            "Should have summary fallback due to LLM failure.",
                            "Include architecture, tests, interfaces, and edge cases.",
                        ]
                        * 40
                    )
                ),
                assistant_answer=(
                    " ".join(
                        [
                            "Acknowledged and stored with details about architecture and tests.",
                            "Tracking edge cases, follow-up tasks, and open questions for future turns.",
                        ]
                        * 30
                    )
                ),
                memory_mode="session",
                user_scope="default",
                graph_state={"ok": True},
            )
        )

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT summary_text FROM session_summaries ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    conn.close()

    assert result.summarized is True
    assert row is not None
    assert "Facts:" in row[0]
