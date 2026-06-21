import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.api.routes_chat import get_research_agent_service  # noqa: E402
from backend.app.config.settings import Settings  # noqa: E402
from backend.app.graph import ResearchAgentService  # noqa: E402
from backend.app.graph.research_graph import build_research_graph  # noqa: E402
from backend.app.llm.ollama_chat import ChatResponse as OllamaChatResponse  # noqa: E402
from backend.app.main import app  # noqa: E402
from backend.app.mcp.client import MCPClient  # noqa: E402
from backend.app.memory.service import MemoryService  # noqa: E402
from backend.app.schemas.tools import ToolResult, ToolSpec  # noqa: E402
from backend.app.storage import SQLiteStore  # noqa: E402


class StubOllamaService:
    async def generate(self, request, callbacks=None):
        return OllamaChatResponse(
            content='{"action":"final_answer","final_answer":"Planner answer"}',
            model="gpt-oss:120b",
        )

    async def stream(self, request, callbacks=None):
        yield type("Chunk", (), {"content": "Planner answer"})()


class StubMCPToolProxy:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "Search"
        self.input_hint = "query"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name, description=self.description, input_hint=self.input_hint
        )

    async def run(
        self, input_text: str, max_results: int = 5, timelimit: str | None = None
    ):
        return ToolResult(summary="no-op", sources=[])


class StubMCPToolRegistry:
    def __init__(self) -> None:
        self._tools = {
            "web_search": StubMCPToolProxy("web_search"),
            "rag_retrieve": StubMCPToolProxy("rag_retrieve"),
        }

    def get(self, name: str):
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def first_tool_name(self) -> str | None:
        return next(iter(self._tools.keys()), None)


class StubSummarizer:
    async def summarize_turns(self, turns):  # type: ignore[no-untyped-def]
        return "Facts: stub"


def _make_service(memory_service: MemoryService) -> ResearchAgentService:
    mock_client = MagicMock(spec=MCPClient)
    mock_client.name = "stub"
    mock_client.specs.return_value = [
        ToolSpec(name="web_search", description="Search", input_hint="query"),
        ToolSpec(name="rag_retrieve", description="RAG", input_hint="query"),
    ]
    service = ResearchAgentService(
        ollama_chat_service=StubOllamaService(),
        mcp_clients=[mock_client],
        memory_service=memory_service,
    )
    # Replace registry with our stub
    registry = StubMCPToolRegistry()
    service._tool_registry = registry
    service._graph = build_research_graph(
        ollama_chat_service=service._ollama_chat_service,
        tool_registry=registry,
        max_tool_calls=service._max_tool_calls,
    )
    return service


def test_chat_session_persists_across_requests(tmp_path: Path) -> None:
    db_path = tmp_path / "integration-memory.db"
    settings = Settings(memory_sqlite_path=str(db_path))
    memory_service = MemoryService(
        settings=settings,
        sqlite_store=SQLiteStore(settings=settings),
        summarizer=StubSummarizer(),
    )
    service = _make_service(memory_service)

    app.dependency_overrides[get_research_agent_service] = lambda: service
    client = TestClient(app)

    first = client.post(
        "/chat",
        json={"query": "First", "memory_mode": "session"},
    )
    assert first.status_code == 200
    session_id = first.json()["session_id"]

    second = client.post(
        "/chat",
        json={"query": "Second", "memory_mode": "session", "session_id": session_id},
    )
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id

    conn = sqlite3.connect(db_path)
    turn_count = conn.execute(
        "SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,)
    ).fetchone()[0]
    conn.close()

    assert turn_count == 4

    app.dependency_overrides.clear()


def test_chat_memory_off_creates_no_durable_records(tmp_path: Path) -> None:
    db_path = tmp_path / "integration-memory-off.db"
    settings = Settings(memory_sqlite_path=str(db_path))
    memory_service = MemoryService(
        settings=settings,
        sqlite_store=SQLiteStore(settings=settings),
        summarizer=StubSummarizer(),
    )
    service = _make_service(memory_service)

    app.dependency_overrides[get_research_agent_service] = lambda: service
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"query": "Hello", "memory_mode": "off"},
    )
    assert response.status_code == 200

    conn = sqlite3.connect(db_path)
    sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    turns = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    conn.close()

    assert sessions == 0
    assert turns == 0

    app.dependency_overrides.clear()
