from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.observability.logging_utils import log_event
from app.rag.ingest import PdfExtractionError, extract_pdf_document
from app.rag.ingest.service import RagIngestService
from app.schemas.rag import RagIngestResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@lru_cache
def get_rag_ingest_service() -> RagIngestService:
    return RagIngestService()


@router.post("/rag/documents", response_model=RagIngestResponse)
async def ingest_documents(
    file: UploadFile = File(...),
    collection_name: str | None = Form(default=None),
    url: str | None = Form(default=None),
    metadata_json: str | None = Form(default=None),
    service: RagIngestService = Depends(get_rag_ingest_service),
) -> RagIngestResponse:
    content_type = (file.content_type or "").lower()
    filename = file.filename or "document.pdf"
    if content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF uploads are supported")
    if collection_name is not None and not collection_name.strip():
        raise HTTPException(status_code=422, detail="collection_name must not be blank")
    normalized_collection_name = collection_name.strip() if collection_name else None

    user_metadata: dict[str, Any] | None = None
    metadata_raw = (metadata_json or "").strip()
    if metadata_raw:
        try:
            parsed = json.loads(metadata_raw)
            if isinstance(parsed, dict):
                user_metadata = parsed
            else:
                user_metadata = {"value": parsed}
                log_event(
                    logger,
                    "rag.ingest.metadata.coerced",
                    mode="json_value_wrap",
                    route="/rag/documents",
                )
        except json.JSONDecodeError:
            user_metadata = {"note": metadata_raw}
            log_event(
                logger,
                "rag.ingest.metadata.coerced",
                mode="plain_text",
                route="/rag/documents",
            )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    try:
        document = extract_pdf_document(
            file_bytes=file_bytes,
            filename=filename,
            url=url,
            metadata=user_metadata,
        )
        return await service.ingest(
            documents=[document],
            collection_name=normalized_collection_name,
        )
    except Exception as exc:
        if isinstance(exc, PdfExtractionError):
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if type(exc).__name__ != "RagIngestError":
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        message = str(exc)
        if "must not be blank" in message.lower() or "no chunks produced" in message.lower():
            raise HTTPException(status_code=422, detail=message) from exc
        raise HTTPException(status_code=502, detail=message) from exc
