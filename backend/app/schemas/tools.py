from __future__ import annotations

from dataclasses import dataclass

from app.schemas.chat import SourceItem


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_hint: str


@dataclass(frozen=True)
class ToolResult:
    summary: str
    sources: list[SourceItem]
