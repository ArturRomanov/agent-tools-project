import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.api.routes_chat import get_research_agent_service  # noqa: E402
from backend.app.main import app  # noqa: E402
from backend.app.schemas.chat import ChatResponse, SourceItem, StreamEvent  # noqa: E402
from backend.app.tools.web_search import WebSearchError  # noqa: E402


class FakeResearchAgentService:
    def __init__(self, fail_run: bool = False, fail_stream: bool = False) -> None:
        self._fail_run = fail_run
        self._fail_stream = fail_stream

    async def run(
        self,
        query: str,
        max_results: int = 5,
        freshness: str = "auto",
        session_id: str | None = None,
        memory_mode: str = "off",
        checkpoint_id: str | None = None,
        request_id: str | None = None,
    ) -> ChatResponse:
        if self._fail_run:
            raise WebSearchError("tool failed")
        return ChatResponse(
            answer="Result answer",
            sources=[
                SourceItem(
                    title="Example",
                    url="https://example.com",
                    snippet="Example snippet",
                )
            ],
            session_id=session_id or "session-test",
            memory_metadata={"memory_mode": memory_mode},
        )

    async def stream(
        self,
        query: str,
        max_results: int = 5,
        freshness: str = "auto",
        session_id: str | None = None,
        memory_mode: str = "off",
        checkpoint_id: str | None = None,
        request_id: str | None = None,
    ):
        if self._fail_stream:
            raise WebSearchError("tool failed")
        yield StreamEvent(
            type="tool_selected", data={"tool": "web_search", "input": query}
        )
        yield StreamEvent(
            type="tool_result", data={"tool": "web_search", "summary": "ok"}
        )
        yield StreamEvent(
            type="sources",
            data={
                "sources": [
                    {
                        "title": "Example",
                        "url": "https://example.com",
                        "snippet": "Example snippet",
                    }
                ]
            },
        )
        yield StreamEvent(type="token", data={"text": "Result"})
        yield StreamEvent(type="done", data={"answer": "Result"})


def test_post_chat_happy_path() -> None:
    app.dependency_overrides[get_research_agent_service] = lambda: (
        FakeResearchAgentService()
    )
    client = TestClient(app)

    response = client.post("/chat", json={"query": "hello", "max_results": 5})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    body = response.json()
    assert body["answer"] == "Result answer"
    assert body["sources"][0]["url"] == "https://example.com"
    assert body["session_id"] == "session-test"

    app.dependency_overrides.clear()


def test_post_chat_validation_error() -> None:
    client = TestClient(app)
    response = client.post(
        "/chat",
        json={"query": "   ", "max_results": 5},
        headers={"X-Request-ID": "req-validation"},
    )
    assert response.status_code == 422
    assert response.headers["X-Request-ID"] == "req-validation"


def test_post_chat_provider_error_maps_to_502() -> None:
    app.dependency_overrides[get_research_agent_service] = lambda: (
        FakeResearchAgentService(fail_run=True)
    )
    client = TestClient(app)

    response = client.post("/chat", json={"query": "hello", "max_results": 5})

    assert response.status_code == 502
    app.dependency_overrides.clear()


def test_post_chat_stream_happy_path() -> None:
    app.dependency_overrides[get_research_agent_service] = lambda: (
        FakeResearchAgentService()
    )
    client = TestClient(app)

    with client.stream(
        "POST", "/chat/stream", json={"query": "hello", "max_results": 5}
    ) as response:
        assert response.status_code == 200
        assert response.headers["X-Request-ID"]
        payload = "".join(response.iter_text())

    events = [line for line in payload.splitlines() if line.startswith("event: ")]
    assert events == [
        "event: tool_selected",
        "event: tool_result",
        "event: sources",
        "event: token",
        "event: done",
    ]

    data_lines = [line for line in payload.splitlines() if line.startswith("data: ")]
    parsed = [json.loads(line.removeprefix("data: ")) for line in data_lines]
    assert parsed[-1]["answer"] == "Result"

    app.dependency_overrides.clear()


def test_post_chat_stream_validation_error() -> None:
    client = TestClient(app)
    response = client.post("/chat/stream", json={"query": "   ", "max_results": 5})
    assert response.status_code == 422


def test_post_chat_stream_emits_error_event() -> None:
    app.dependency_overrides[get_research_agent_service] = lambda: (
        FakeResearchAgentService(fail_stream=True)
    )
    client = TestClient(app)

    with client.stream(
        "POST", "/chat/stream", json={"query": "hello", "max_results": 5}
    ) as response:
        assert response.status_code == 200
        payload = "".join(response.iter_text())

    assert "event: error" in payload
    assert "tool failed" in payload

    app.dependency_overrides.clear()
