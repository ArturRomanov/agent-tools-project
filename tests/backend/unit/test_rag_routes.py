import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.api import routes_rag  # noqa: E402
from backend.app.api.routes_rag import get_rag_ingest_service  # noqa: E402
from backend.app.main import app  # noqa: E402
from backend.app.schemas.rag import RagDocumentInput, RagIngestResponse  # noqa: E402


class FakeRagIngestService:
    def __init__(
        self, should_fail: bool = False, error_message: str = "provider failed"
    ) -> None:
        self._should_fail = should_fail
        self._error_message = error_message

    async def ingest(self, documents, collection_name=None) -> RagIngestResponse:
        if self._should_fail:

            class RagIngestError(RuntimeError):
                pass

            raise RagIngestError(self._error_message)
        return RagIngestResponse(
            collection_name=collection_name or "rag_documents",
            indexed_documents=len(documents),
            indexed_chunks=3,
        )


def _fake_doc() -> RagDocumentInput:
    return RagDocumentInput(
        id="doc-id",
        title="Extracted Title",
        text="Extracted text",
        metadata={"source_type": "pdf_upload"},
    )


def test_post_rag_documents_happy_path(monkeypatch) -> None:
    app.dependency_overrides[get_rag_ingest_service] = lambda: FakeRagIngestService()
    monkeypatch.setattr(
        routes_rag, "extract_pdf_document", lambda **kwargs: _fake_doc()
    )
    client = TestClient(app)

    response = client.post(
        "/rag/documents",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"collection_name": "rag_documents"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["collection_name"] == "rag_documents"
    assert body["indexed_documents"] == 1
    assert body["indexed_chunks"] == 3
    app.dependency_overrides.clear()


def test_post_rag_documents_happy_path_only_file(monkeypatch) -> None:
    app.dependency_overrides[get_rag_ingest_service] = lambda: FakeRagIngestService()
    monkeypatch.setattr(
        routes_rag, "extract_pdf_document", lambda **kwargs: _fake_doc()
    )
    client = TestClient(app)

    response = client.post(
        "/rag/documents",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["collection_name"] == "rag_documents"
    assert body["indexed_documents"] == 1
    app.dependency_overrides.clear()


def test_post_rag_documents_invalid_file_type() -> None:
    client = TestClient(app)
    response = client.post(
        "/rag/documents",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 422


def test_post_rag_documents_empty_file() -> None:
    client = TestClient(app)
    response = client.post(
        "/rag/documents",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 422


def test_post_rag_documents_metadata_plain_text_coerced(monkeypatch) -> None:
    app.dependency_overrides[get_rag_ingest_service] = lambda: FakeRagIngestService()
    observed_metadata = {}

    def _fake_extract(**kwargs):
        observed_metadata["value"] = kwargs.get("metadata")
        return _fake_doc()

    monkeypatch.setattr(routes_rag, "extract_pdf_document", _fake_extract)
    client = TestClient(app)

    response = client.post(
        "/rag/documents",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"metadata_json": "not-json"},
    )
    assert response.status_code == 200
    assert observed_metadata["value"] == {"note": "not-json"}
    app.dependency_overrides.clear()


def test_post_rag_documents_collection_override_with_plain_text_metadata(
    monkeypatch,
) -> None:
    app.dependency_overrides[get_rag_ingest_service] = lambda: FakeRagIngestService()
    observed_metadata = {}

    def _fake_extract(**kwargs):
        observed_metadata["value"] = kwargs.get("metadata")
        return _fake_doc()

    monkeypatch.setattr(routes_rag, "extract_pdf_document", _fake_extract)
    client = TestClient(app)

    response = client.post(
        "/rag/documents",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"collection_name": "Test", "url": "test.test", "metadata_json": "Test"},
    )
    assert response.status_code == 200
    assert response.json()["collection_name"] == "Test"
    assert observed_metadata["value"] == {"note": "Test"}
    app.dependency_overrides.clear()


def test_post_rag_documents_metadata_non_object_json_wrapped(monkeypatch) -> None:
    app.dependency_overrides[get_rag_ingest_service] = lambda: FakeRagIngestService()
    observed_metadata = {}

    def _fake_extract(**kwargs):
        observed_metadata["value"] = kwargs.get("metadata")
        return _fake_doc()

    monkeypatch.setattr(routes_rag, "extract_pdf_document", _fake_extract)
    client = TestClient(app)

    response = client.post(
        "/rag/documents",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"metadata_json": '"abc"'},
    )
    assert response.status_code == 200
    assert observed_metadata["value"] == {"value": "abc"}
    app.dependency_overrides.clear()


def test_post_rag_documents_metadata_object_passthrough(monkeypatch) -> None:
    app.dependency_overrides[get_rag_ingest_service] = lambda: FakeRagIngestService()
    observed_metadata = {}

    def _fake_extract(**kwargs):
        observed_metadata["value"] = kwargs.get("metadata")
        return _fake_doc()

    monkeypatch.setattr(routes_rag, "extract_pdf_document", _fake_extract)
    client = TestClient(app)

    response = client.post(
        "/rag/documents",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"metadata_json": '{"team":"platform"}'},
    )
    assert response.status_code == 200
    assert observed_metadata["value"] == {"team": "platform"}
    app.dependency_overrides.clear()


def test_post_rag_documents_provider_error_maps_to_502(monkeypatch) -> None:
    app.dependency_overrides[get_rag_ingest_service] = lambda: FakeRagIngestService(
        should_fail=True
    )
    monkeypatch.setattr(
        routes_rag, "extract_pdf_document", lambda **kwargs: _fake_doc()
    )
    client = TestClient(app)

    response = client.post(
        "/rag/documents",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert response.status_code == 502
    app.dependency_overrides.clear()
