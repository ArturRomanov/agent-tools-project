from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, TextContent

from app.schemas.tools import ToolResult, ToolSpec
from app.schemas.chat import SourceItem

logger = logging.getLogger(__name__)


class MCPClientError(RuntimeError):
    pass


@dataclass
class MCPClient:
    """1:1 MCP client — manages a single ClientSession to a single MCP server."""

    name: str
    server_url: str
    _session: ClientSession | None = field(default=None, repr=False)
    _tool_specs: list[ToolSpec] = field(default_factory=list, repr=False)
    _tool_names: set[str] = field(default_factory=set, repr=False)

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[MCPClient]:
        """Connect to the MCP server via Streamable HTTP, initialize, discover tools."""
        logger.info(
            "mcp.client.connecting",
            extra={"event": "mcp.client.connecting", "server": self.name, "url": self.server_url},
        )
        async with streamablehttp_client(self.server_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                self._session = session

                tools_result = await session.list_tools()
                self._tool_specs = []
                self._tool_names = set()
                for tool in tools_result.tools:
                    spec = ToolSpec(
                        name=tool.name,
                        description=tool.description or "",
                        input_hint=_extract_input_hint(tool.inputSchema),
                    )
                    self._tool_specs.append(spec)
                    self._tool_names.add(tool.name)

                logger.info(
                    "mcp.client.connected",
                    extra={
                        "event": "mcp.client.connected",
                        "server": self.name,
                        "tools": sorted(self._tool_names),
                    },
                )
                try:
                    yield self
                finally:
                    self._session = None
                    self._tool_specs = []
                    self._tool_names = set()

    def specs(self) -> list[ToolSpec]:
        return list(self._tool_specs)

    def has_tool(self, name: str) -> bool:
        return name in self._tool_names

    def tool_names(self) -> list[str]:
        return sorted(self._tool_names)

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        """Call a tool on this server. Raises MCPClientError if not connected or tool not found."""
        if self._session is None:
            raise MCPClientError(f"MCPClient '{self.name}' is not connected")
        if name not in self._tool_names:
            raise MCPClientError(f"Tool '{name}' not found on server '{self.name}'")

        logger.info(
            "mcp.client.call_tool",
            extra={
                "event": "mcp.client.call_tool",
                "server": self.name,
                "tool": name,
            },
        )
        result = await self._session.call_tool(name, arguments)
        return result


def _extract_input_hint(schema: dict | None) -> str:
    """Extract a human-readable input hint from the JSON schema."""
    if not schema:
        return ""
    props = schema.get("properties", {})
    required = schema.get("required", [])
    if not props:
        return ""
    hints = []
    for prop_name in required:
        prop = props.get(prop_name, {})
        desc = prop.get("description", prop_name)
        hints.append(desc)
    if hints:
        return "; ".join(hints)
    first_prop = next(iter(props.values()), {})
    return first_prop.get("description", "")
