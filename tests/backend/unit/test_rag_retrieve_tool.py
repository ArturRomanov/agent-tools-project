import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.rag.retrieval import RagRetrievalError  # noqa: E402
from backend.app.schemas.chat import SourceItem  # noqa: E402
from backend.app.tools.rag_retrieve import RagRetrieveTool  # noqa: E402


class StubRetriever:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail

    async def retrieve(self, query: str, max_results: int):
        if self.should_fail:
            raise RagRetrievalError("failed")
        return [
            SourceItem(
                title="Doc 1",
                url="rag://local/doc-1",
                snippet="Chunk about test",
            ),
            SourceItem(
                title="Doc 1",
                url="rag://local/doc-1",
                snippet="Chunk about test",
            ),
        ][:max_results]


def test_rag_retrieve_tool_run_happy_path() -> None:
    tool = RagRetrieveTool(retriever=StubRetriever())  # type: ignore[arg-type]

    result = asyncio.run(tool.run("test", max_results=5))

    assert "Retrieved 2 RAG sources" in result.summary
    assert len(result.sources) == 1
    assert result.sources[0].url == "rag://local/doc-1"


def test_rag_retrieve_tool_run_propagates_error() -> None:
    tool = RagRetrieveTool(retriever=StubRetriever(should_fail=True))  # type: ignore[arg-type]

    try:
        asyncio.run(tool.run("test", max_results=5))
        assert False, "Expected RagRetrievalError"
    except RagRetrievalError:
        assert True
