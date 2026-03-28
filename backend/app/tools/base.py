from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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


class AgentTool(Protocol):
    name: str
    description: str
    input_hint: str

    async def run(
        self,
        input_text: str,
        max_results: int = 5,
        timelimit: str | None = None,
    ) -> ToolResult: ...

    def spec(self) -> ToolSpec: ...
