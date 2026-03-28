from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    query: str = Field(..., description="User input prompt")
    max_results: int = Field(default=5, ge=1, le=10)
    freshness: Literal["auto", "any", "day", "week", "month"] = Field(default="auto")
    session_id: str | None = Field(default=None)
    memory_mode: Literal["off", "session", "long_term"] = Field(default="off")
    checkpoint_id: str | None = Field(default=None)

    @field_validator("query")
    @classmethod
    def _query_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class SourceItem(BaseModel):
    title: str
    url: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    session_id: str
    memory_metadata: dict[str, Any] = Field(default_factory=dict)


StreamEventType = Literal[
    "token",
    "sources",
    "tool_selected",
    "tool_result",
    "memory_loaded",
    "memory_summarized",
    "checkpoint_saved",
    "done",
    "error",
]


class StreamEvent(BaseModel):
    type: StreamEventType
    data: dict[str, Any]
