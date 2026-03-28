import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.rag.vectorstore import qdrant_store as qdrant_store_module  # noqa: E402
from backend.app.config.settings import Settings  # noqa: E402
from backend.app.rag.vectorstore.qdrant_store import QdrantStore  # noqa: E402
from qdrant_client.http import models  # noqa: E402


class StubClient:
    def __init__(self) -> None:
        self.created_distance = None

    def collection_exists(self, collection_name: str) -> bool:
        return False

    def create_collection(self, collection_name: str, vectors_config) -> None:
        self.created_distance = vectors_config.distance


class QueryOnlyClient:
    def collection_exists(self, collection_name: str) -> bool:
        return True

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        limit: int,
        with_payload: bool,
    ):
        point = models.ScoredPoint(
            id="123e4567-e89b-12d3-a456-426614174000",
            version=1,
            score=0.91,
            payload={
                "title": "Doc",
                "text": "Snippet",
                "url": "https://example.com",
                "document_id": "doc-1",
                "metadata": {"k": "v"},
            },
            vector=None,
        )
        return models.QueryResponse(points=[point])


def test_for_collection_reuses_client() -> None:
    settings = Settings(rag_distance_metric="cosine")
    client = StubClient()
    base_store = QdrantStore(settings=settings, client=client, collection_name="base")

    derived_store = base_store.for_collection("other")

    assert derived_store.collection_name == "other"
    derived_store.ensure_collection(vector_size=8)
    assert client.created_distance == models.Distance.COSINE


def test_ensure_collection_metric_normalization() -> None:
    settings = Settings(rag_distance_metric=" COSINE ")
    client = StubClient()
    store = QdrantStore(settings=settings, client=client)

    store.ensure_collection(vector_size=8)

    assert client.created_distance == models.Distance.COSINE


def test_ensure_collection_unknown_metric_falls_back_to_cosine() -> None:
    settings = Settings(rag_distance_metric="unknown-metric")
    client = StubClient()
    store = QdrantStore(settings=settings, client=client)

    store.ensure_collection(vector_size=8)

    assert client.created_distance == models.Distance.COSINE


def test_local_client_cache_reuses_single_client_per_path(monkeypatch) -> None:
    created_paths = []

    class FakeQdrantClient:
        def __init__(self, path: str):
            self.path = path
            created_paths.append(path)

    qdrant_store_module._LOCAL_CLIENTS_BY_PATH.clear()
    monkeypatch.setattr(qdrant_store_module, "QdrantClient", FakeQdrantClient)
    settings = Settings(qdrant_path="./.data/qdrant-cache-test")

    store_a = QdrantStore(settings=settings)
    store_b = QdrantStore(settings=settings)
    client_a = store_a._get_client()  # noqa: SLF001
    client_b = store_b._get_client()  # noqa: SLF001

    assert client_a is client_b
    assert created_paths == ["./.data/qdrant-cache-test"]


def test_search_uses_query_points_when_search_absent() -> None:
    settings = Settings()
    store = QdrantStore(settings=settings, client=QueryOnlyClient())  # type: ignore[arg-type]

    results = store.search(query_vector=[0.1, 0.2], limit=3)

    assert len(results) == 1
    assert results[0].title == "Doc"
    assert results[0].snippet == "Snippet"
