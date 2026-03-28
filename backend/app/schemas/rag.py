from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class RagDocumentInput(BaseModel):
    id: str = Field(..., description="Unique document id")
    title: str = Field(..., description="Document title")
    text: str = Field(..., description="Document full text")
    url: str | None = Field(default=None, description="Optional canonical URL")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata map")

    @field_validator("id", "title", "text")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class RagIngestResponse(BaseModel):
    collection_name: str
    indexed_documents: int
    indexed_chunks: int
