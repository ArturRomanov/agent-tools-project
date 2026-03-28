import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.graph import ResearchAgentService  # noqa: E402
from backend.app.llm.ollama_chat import ChatResponse as OllamaChatResponse  # noqa: E402
from backend.app.llm.ollama_chat import StreamChunk  # noqa: E402
from backend.app.schemas.chat import SourceItem, StreamEvent  # noqa: E402
from backend.app.tools.base import ToolResult, ToolSpec  # noqa: E402


class StubSearchTool:
    name = "web_search"
    description = "Search web"
    input_hint = "query"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name, description=self.description, input_hint=self.input_hint
        )

    async def run(
        self, input_text: str, max_results: int = 5, timelimit: str | None = None
    ) -> ToolResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("search failed")
        return ToolResult(
            summary=f"found {max_results}",
            sources=[
                SourceItem(title="Doc", url="https://example.com", snippet="Snippet")
            ],
        )


class StubRagTool:
    name = "rag_retrieve"
    description = "Retrieve from indexed docs"
    input_hint = "question"

    def __init__(self) -> None:
        self.calls = 0

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name, description=self.description, input_hint=self.input_hint
        )

    async def run(
        self, input_text: str, max_results: int = 5, timelimit: str | None = None
    ) -> ToolResult:
        self.calls += 1
        return ToolResult(
            summary=f"rag found {max_results}",
            sources=[
                SourceItem(
                    title="Indexed Doc",
                    url="rag://local/doc-1",
                    snippet="Indexed snippet",
                )
            ],
        )


class StubOllamaService:
    def __init__(
        self,
        generate_outputs: list[str] | None = None,
        stream_chunks: list[str] | None = None,
    ):
        self.generate_outputs = generate_outputs or []
        self.stream_chunks = stream_chunks or ["Syn", "thesis"]

    async def generate(self, request):
        if self.generate_outputs:
            content = self.generate_outputs.pop(0)
        else:
            content = '{"action":"final_answer","final_answer":"Synthesis answer"}'
        return OllamaChatResponse(content=content, model="gpt-oss:120b")

    async def stream(self, request):
        for chunk in self.stream_chunks:
            yield StreamChunk(content=chunk)


def test_run_direct_answer_path() -> None:
    service = ResearchAgentService(
        ollama_chat_service=StubOllamaService(
            generate_outputs=[
                '{"action":"final_answer","final_answer":"Direct answer"}',
                "Synthesis answer",
            ]
        ),
        web_search_tool=StubSearchTool(),
    )

    response = asyncio.run(service.run("Simple question"))

    assert response.answer == "Synthesis answer"
    assert response.sources == []


def test_run_tool_call_then_final_answer_path() -> None:
    search_tool = StubSearchTool()
    service = ResearchAgentService(
        ollama_chat_service=StubOllamaService(
            generate_outputs=[
                '{"action":"call_tool","tool_name":"web_search","tool_input":"Test news"}',
                '{"action":"final_answer","final_answer":"Tool-backed answer"}',
                "Tool-backed answer",
            ]
        ),
        web_search_tool=search_tool,
    )

    response = asyncio.run(service.run("Test news"))

    assert response.answer == "Tool-backed answer"
    assert len(response.sources) == 1
    assert search_tool.calls == 1


def test_run_loop_guard_limits_tool_calls_to_two() -> None:
    search_tool = StubSearchTool()
    service = ResearchAgentService(
        ollama_chat_service=StubOllamaService(
            generate_outputs=[
                '{"action":"call_tool","tool_name":"web_search","tool_input":"q1"}',
                '{"action":"call_tool","tool_name":"web_search","tool_input":"q2"}',
                '{"action":"call_tool","tool_name":"web_search","tool_input":"q3"}',
                "Synthesized after loop guard",
            ]
        ),
        web_search_tool=search_tool,
        max_tool_calls=2,
    )

    response = asyncio.run(service.run("Need deep search"))

    assert search_tool.calls == 2
    assert response.answer == "Synthesized after loop guard"


def test_run_invalid_planner_output_fallback() -> None:
    search_tool = StubSearchTool()
    service = ResearchAgentService(
        ollama_chat_service=StubOllamaService(
            generate_outputs=[
                "not-json",
                '{"action":"final_answer","final_answer":"Fallback recovered"}',
                "Fallback recovered",
            ]
        ),
        web_search_tool=search_tool,
    )

    response = asyncio.run(service.run("Recent Test new"))

    assert response.answer == "Fallback recovered"
    assert search_tool.calls == 1


def test_run_unknown_tool_fallback() -> None:
    search_tool = StubSearchTool()
    service = ResearchAgentService(
        ollama_chat_service=StubOllamaService(
            generate_outputs=[
                '{"action":"call_tool","tool_name":"unknown_tool","tool_input":"x"}',
                '{"action":"final_answer","final_answer":"Recovered from unknown tool"}',
                "Recovered from unknown tool",
            ]
        ),
        web_search_tool=search_tool,
    )

    response = asyncio.run(service.run("Question"))

    assert response.answer == "Recovered from unknown tool"
    assert search_tool.calls == 1


def test_stream_emits_tool_events_and_done() -> None:
    service = ResearchAgentService(
        ollama_chat_service=StubOllamaService(
            generate_outputs=[
                '{"action":"call_tool","tool_name":"web_search","tool_input":"Test"}',
                '{"action":"final_answer","final_answer":"Final from planner"}',
            ],
            stream_chunks=["Final ", "from ", "planner"],
        ),
        web_search_tool=StubSearchTool(),
    )

    async def collect() -> list[StreamEvent]:
        events = []
        async for event in service.stream("Test"):
            events.append(event)
        return events

    events = asyncio.run(collect())

    event_types = [event.type for event in events]
    assert event_types[:3] == ["tool_selected", "tool_result", "sources"]
    assert "token" in event_types
    assert event_types[-1] == "done"
    assert events[-1].data["answer"] == "Final from planner"

    streamed = "".join(
        event.data.get("text", "") for event in events if event.type == "token"
    )
    assert streamed == events[-1].data["answer"]


def test_run_rag_tool_call_then_final_answer_path() -> None:
    search_tool = StubSearchTool()
    rag_tool = StubRagTool()
    service = ResearchAgentService(
        ollama_chat_service=StubOllamaService(
            generate_outputs=[
                '{"action":"call_tool","tool_name":"rag_retrieve","tool_input":"Test"}',
                '{"action":"final_answer","final_answer":"RAG-backed answer"}',
                "RAG-backed answer",
            ]
        ),
        web_search_tool=search_tool,
        rag_tool=rag_tool,
    )

    response = asyncio.run(service.run("Test question"))

    assert response.answer == "RAG-backed answer"
    assert len(response.sources) == 1
    assert rag_tool.calls == 1
    assert search_tool.calls == 0
