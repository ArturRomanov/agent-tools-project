from __future__ import annotations

from dataclasses import dataclass

from app.tools.base import AgentTool, ToolSpec


@dataclass
class ToolRegistry:
    _tools: dict[str, AgentTool]

    @classmethod
    def from_tools(cls, tools: list[AgentTool]) -> "ToolRegistry":
        return cls(_tools={tool.name: tool for tool in tools})

    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    def first_tool_name(self) -> str | None:
        if not self._tools:
            return None
        return next(iter(self._tools.keys()))
