from __future__ import annotations

import hashlib
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PdfExtractionError(RuntimeError):
    pass


def extract_pdf_document(
    file_bytes: bytes,
    filename: str,
    url: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract text and metadata from a PDF.

    Returns a dict with keys: id, title, text, url, metadata.
    """
    if not file_bytes:
        raise PdfExtractionError("Uploaded file is empty")

    try:
        reader = PdfReader(BytesIO(file_bytes))
        raw_text = "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if not raw_text:
            raise PdfExtractionError("PDF does not contain extractable text")

        info = reader.metadata or {}
        doc_title = (
            getattr(info, "title", None)
            or (info.get("/Title") if hasattr(info, "get") else None)
            or Path(filename).stem
        )
        title = str(doc_title).strip() or Path(filename).stem
        document_id = hashlib.sha256(file_bytes).hexdigest()

        combined_metadata: dict[str, Any] = {
            "source_type": "pdf_upload",
            "filename": filename,
            "page_count": len(reader.pages),
        }
        optional_fields = {
            "author": getattr(info, "author", None),
            "subject": getattr(info, "subject", None),
            "creator": getattr(info, "creator", None),
            "producer": getattr(info, "producer", None),
        }
        for key, value in optional_fields.items():
            if value is None:
                continue
            cleaned = str(value).strip()
            if cleaned:
                combined_metadata[key] = cleaned
        if metadata:
            combined_metadata.update(metadata)

        return {
            "id": document_id,
            "title": title,
            "text": raw_text,
            "url": url,
            "metadata": combined_metadata,
        }
    except PdfExtractionError:
        raise
    except Exception as exc:
        raise PdfExtractionError("Failed to parse PDF content") from exc
