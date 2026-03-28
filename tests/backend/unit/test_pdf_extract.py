import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.rag.ingest import pdf_extract  # noqa: E402
from backend.app.rag.ingest.pdf_extract import PdfExtractionError, extract_pdf_document  # noqa: E402


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeMeta:
    def __init__(self, title: str | None = None, author: str | None = None) -> None:
        self.title = title
        self.author = author
        self.subject = None
        self.creator = None
        self.producer = None


class _FakeReader:
    def __init__(
        self, pages: list[_FakePage], metadata: _FakeMeta | None = None
    ) -> None:
        self.pages = pages
        self.metadata = metadata


def test_extract_pdf_document_happy_path_with_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        pdf_extract,
        "PdfReader",
        lambda _: _FakeReader(
            pages=[_FakePage("Page one"), _FakePage("Page two")],
            metadata=_FakeMeta(title="Test Title", author="Test"),
        ),
    )

    result = extract_pdf_document(
        file_bytes=b"fake-pdf-bytes",
        filename="test.pdf",
        url="https://example.com/test",
        metadata={"team": "platform"},
    )

    assert result.title == "Test Title"
    assert "Page one" in result.text
    assert result.metadata is not None
    assert result.metadata["source_type"] == "pdf_upload"
    assert result.metadata["filename"] == "test.pdf"
    assert result.metadata["page_count"] == 2
    assert result.metadata["author"] == "Test"
    assert result.metadata["team"] == "platform"


def test_extract_pdf_document_fallback_title_from_filename(monkeypatch) -> None:
    monkeypatch.setattr(
        pdf_extract,
        "PdfReader",
        lambda _: _FakeReader(
            pages=[_FakePage("Only page")],
            metadata=_FakeMeta(title=None),
        ),
    )

    result = extract_pdf_document(
        file_bytes=b"fake-pdf-bytes",
        filename="internal_guide.pdf",
    )
    assert result.title == "internal_guide"


def test_extract_pdf_document_empty_text_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        pdf_extract,
        "PdfReader",
        lambda _: _FakeReader(
            pages=[_FakePage(""), _FakePage("")],
            metadata=_FakeMeta(title="No text"),
        ),
    )
    try:
        extract_pdf_document(file_bytes=b"fake-pdf-bytes", filename="no-text.pdf")
        assert False, "Expected PdfExtractionError"
    except PdfExtractionError:
        assert True
