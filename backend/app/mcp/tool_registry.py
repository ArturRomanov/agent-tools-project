from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from mcp.types import TextContent

from app.mcp.client import MCPClient
from app.schemas.chat import SourceItem
from app.schemas.tools import ToolResult, ToolSpec

logger = logging.getLogger(__name__)


@dataclass
class MCPToolProxy:
    """Satisfies the AgentTool protocol by delegating run() to the owning MCPClient."""

    name: str
    description: str
    input_hint: str
    _client: MCPClient

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, input_hint=self.input_hint)

    async def run(
        self,
        input_text: str,
        max_results: int = 5,
        timelimit: str | None = None,
    ) -> ToolResult:
        arguments: dict = {"query": input_text, "max_results": max_results}
        if timelimit is not None:
            arguments["timelimit"] = timelimit

        call_result = await self._client.call_tool(self.name, arguments)

        # Parse the JSON response from the MCP tool
        text_parts = []
        for content in call_result.content:
            if isinstance(content, TextContent):
                text_parts.append(content.text)
            elif hasattr(content, "text"):
                text_parts.append(content.text)

        raw_text = "\n".join(text_parts)
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            return ToolResult(summary=raw_text, sources=[])

        summary = parsed.get("summary", "")
        raw_sources = parsed.get("sources", [])
        sources = [
            SourceItem(
                title=s.get("title", ""),
                url=s.get("url", ""),
                snippet=s.get("snippet", ""),
            )
            for s in raw_sources
            if isinstance(s, dict)
        ]
        return ToolResult(summary=summary, sources=sources)


class MCPToolRegistry:
    """Aggregates tools from multiple MCPClient instances.

    Provides the same interface as ToolRegistry: specs(), get(), first_tool_name().
    """

    def __init__(self, clients: list[MCPClient]) -> None:
        self._proxies: dict[str, MCPToolProxy] = {}
        for client in clients:
            for spec in client.specs():
                if spec.name in self._proxies:
                    logger.warning(
                        "mcp.tool_registry.name_collision",
                        extra={
                            "event": "mcp.tool_registry.name_collision",
                            "tool": spec.name,
                            "server": client.name,
                        },
                    )
                self._proxies[spec.name] = MCPToolProxy(
                    name=spec.name,
                    description=spec.description,
                    input_hint=spec.input_hint,
                    _client=client,
                )

    def get(self, name: str) -> MCPToolProxy | None:
        return self._proxies.get(name)

    def specs(self) -> list[ToolSpec]:
        return [proxy.spec() for proxy in self._proxies.values()]

    def first_tool_name(self) -> str | None:
        if not self._proxies:
            return None
        return next(iter(self._proxies.keys()))
