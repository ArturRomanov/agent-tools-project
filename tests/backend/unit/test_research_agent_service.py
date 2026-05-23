import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.graph import ResearchAgentService  # noqa: E402
from backend.app.llm.ollama_chat import ChatResponse as OllamaChatResponse  # noqa: E402
from backend.app.llm.ollama_chat import StreamChunk  # noqa: E402
from backend.app.mcp.client import MCPClient  # noqa: E402
from backend.app.schemas.chat import SourceItem, StreamEvent  # noqa: E402
from backend.app.schemas.tools import ToolResult, ToolSpec  # noqa: E402


class StubMCPToolProxy:
    """A stub tool proxy that mimics what MCPToolRegistry + MCPToolProxy do."""

    def __init__(self, name: str, description: str, input_hint: str, fail: bool = False) -> None:
        self.name = name
        self.description = description
        self.input_hint = input_hint
        self.fail = fail
        self.calls = 0

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, input_hint=self.input_hint)

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


class StubMCPToolRegistry:
    """A stub registry to directly inject into ResearchAgentService."""

    def __init__(self, tools: list[StubMCPToolProxy]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def get(self, name: str):
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    def first_tool_name(self) -> str | None:
        if not self._tools:
            return None
        return next(iter(self._tools.keys()))


def _make_stub_mcp_client(name: str, tool_names: list[str]) -> MCPClient:
    """Create a mock MCPClient with given tool names."""
    client = MagicMock(spec=MCPClient)
    client.name = name
    client.specs.return_value = [
        ToolSpec(name=tn, description=f"Tool {tn}", input_hint="query")
        for tn in tool_names
    ]
    return client


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


def _make_service(
    ollama_service: StubOllamaService,
    search_tool: StubMCPToolProxy | None = None,
    rag_tool: StubMCPToolProxy | None = None,
    max_tool_calls: int = 2,
) -> ResearchAgentService:
    """Build a ResearchAgentService with a stub MCP registry injected."""
    tools = []
    if search_tool:
        tools.append(search_tool)
    if rag_tool:
        tools.append(rag_tool)
    if not tools:
        tools.append(StubMCPToolProxy("web_search", "Search web", "query"))

    # Use a real MCPClient mock to pass the mcp_clients check,
    # then monkey-patch the tool_registry with our stub.
    mock_client = _make_stub_mcp_client("stub", [t.name for t in tools])
    service = ResearchAgentService(
        ollama_chat_service=ollama_service,
        mcp_clients=[mock_client],
        max_tool_calls=max_tool_calls,
    )
    # Replace the MCPToolRegistry with our stub that uses proxies directly
    service._tool_registry = StubMCPToolRegistry(tools)
    # Rebuild the graph with the new registry
    from backend.app.graph.research_graph import build_research_graph
    service._graph = build_research_graph(
        ollama_chat_service=service._ollama_chat_service,
        tool_registry=service._tool_registry,
        max_tool_calls=max_tool_calls,
    )
    return service


def test_run_direct_answer_path() -> None:
    service = _make_service(
        StubOllamaService(
            generate_outputs=[
                '{"action":"final_answer","final_answer":"Direct answer"}',
                "Synthesis answer",
            ]
        ),
    )

    response = asyncio.run(service.run("Simple question"))

    assert response.answer == "Synthesis answer"
    assert response.sources == []


def test_run_tool_call_then_final_answer_path() -> None:
    search_tool = StubMCPToolProxy("web_search", "Search web", "query")
    service = _make_service(
        StubOllamaService(
            generate_outputs=[
                '{"action":"call_tool","tool_name":"web_search","tool_input":"Test news"}',
                '{"action":"final_answer","final_answer":"Tool-backed answer"}',
                "Tool-backed answer",
            ]
        ),
        search_tool=search_tool,
    )

    response = asyncio.run(service.run("Test news"))

    assert response.answer == "Tool-backed answer"
    assert len(response.sources) == 1
    assert search_tool.calls == 1


def test_run_loop_guard_limits_tool_calls_to_two() -> None:
    search_tool = StubMCPToolProxy("web_search", "Search web", "query")
    service = _make_service(
        StubOllamaService(
            generate_outputs=[
                '{"action":"call_tool","tool_name":"web_search","tool_input":"q1"}',
                '{"action":"call_tool","tool_name":"web_search","tool_input":"q2"}',
                '{"action":"call_tool","tool_name":"web_search","tool_input":"q3"}',
                "Synthesized after loop guard",
            ]
        ),
        search_tool=search_tool,
        max_tool_calls=2,
    )

    response = asyncio.run(service.run("Need deep search"))

    assert search_tool.calls == 2
    assert response.answer == "Synthesized after loop guard"


def test_run_invalid_planner_output_fallback() -> None:
    search_tool = StubMCPToolProxy("web_search", "Search web", "query")
    service = _make_service(
        StubOllamaService(
            generate_outputs=[
                "not-json",
                '{"action":"final_answer","final_answer":"Fallback recovered"}',
                "Fallback recovered",
            ]
        ),
        search_tool=search_tool,
    )

    response = asyncio.run(service.run("Recent Test new"))

    assert response.answer == "Fallback recovered"
    assert search_tool.calls == 1


def test_run_unknown_tool_fallback() -> None:
    search_tool = StubMCPToolProxy("web_search", "Search web", "query")
    service = _make_service(
        StubOllamaService(
            generate_outputs=[
                '{"action":"call_tool","tool_name":"unknown_tool","tool_input":"x"}',
                '{"action":"final_answer","final_answer":"Recovered from unknown tool"}',
                "Recovered from unknown tool",
            ]
        ),
        search_tool=search_tool,
    )

    response = asyncio.run(service.run("Question"))

    assert response.answer == "Recovered from unknown tool"
    assert search_tool.calls == 1


def test_stream_emits_tool_events_and_done() -> None:
    search_tool = StubMCPToolProxy("web_search", "Search web", "query")
    service = _make_service(
        StubOllamaService(
            generate_outputs=[
                '{"action":"call_tool","tool_name":"web_search","tool_input":"Test"}',
                '{"action":"final_answer","final_answer":"Final from planner"}',
            ],
            stream_chunks=["Final ", "from ", "planner"],
        ),
        search_tool=search_tool,
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
    search_tool = StubMCPToolProxy("web_search", "Search web", "query")
    rag_tool = StubMCPToolProxy("rag_retrieve", "Retrieve from indexed docs", "question")
    service = _make_service(
        StubOllamaService(
            generate_outputs=[
                '{"action":"call_tool","tool_name":"rag_retrieve","tool_input":"Test"}',
                '{"action":"final_answer","final_answer":"RAG-backed answer"}',
                "RAG-backed answer",
            ]
        ),
        search_tool=search_tool,
        rag_tool=rag_tool,
    )

    response = asyncio.run(service.run("Test question"))

    assert response.answer == "RAG-backed answer"
    assert len(response.sources) == 1
    assert rag_tool.calls == 1
    assert search_tool.calls == 0
