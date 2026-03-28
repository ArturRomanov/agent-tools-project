import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.rag.retrieval import RagRetrievalError, RagRetriever  # noqa: E402
from backend.app.rag.vectorstore.qdrant_store import RetrievedChunk  # noqa: E402


class StubEmbeddingsService:
    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class StubStore:
    def search(self, query_vector: list[float], limit: int):
        return [
            RetrievedChunk(
                title="Doc A",
                snippet="chunk text",
                url="rag://local/doc-a",
                score=0.9,
                document_id="doc-a",
                metadata={},
            )
        ][:limit]


class FailingStore:
    def search(self, query_vector: list[float], limit: int):
        raise RuntimeError("qdrant closed")


def test_rag_retriever_happy_path() -> None:
    retriever = RagRetriever(
        embeddings_service=StubEmbeddingsService(),  # type: ignore[arg-type]
        store=StubStore(),  # type: ignore[arg-type]
    )

    result = asyncio.run(retriever.retrieve("policy", max_results=5))

    assert len(result) == 1
    assert result[0].url == "rag://local/doc-a"


def test_rag_retriever_error_translation() -> None:
    retriever = RagRetriever(
        embeddings_service=StubEmbeddingsService(),  # type: ignore[arg-type]
        store=FailingStore(),  # type: ignore[arg-type]
    )
    try:
        asyncio.run(retriever.retrieve("policy", max_results=5))
        assert False, "Expected RagRetrievalError"
    except RagRetrievalError:
        assert True
