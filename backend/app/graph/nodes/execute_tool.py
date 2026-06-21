from __future__ import annotations

import logging
import time

from app.config.settings import get_settings
from app.graph.state import AgentState
from app.observability.logging_utils import log_event, sanitize_text, summarize_sources
from app.schemas.chat import SourceItem

logger = logging.getLogger(__name__)
settings = get_settings()


async def execute_selected_tool(
    state: AgentState, tool_registry: object
) -> AgentState:
    started_at = time.perf_counter()
    tool_name = state.get("selected_tool")
    if not tool_name:
        log_event(logger, "agent.tool.execute.error", error_type="NoToolSelected")
        return {
            "should_continue": False,
            "latest_tool_result_summary": "No tool selected.",
        }

    tool = tool_registry.get(tool_name)
    if tool is None:
        log_event(
            logger,
            "agent.tool.execute.error",
            tool=tool_name,
            error_type="UnknownToolSelected",
        )
        return {
            "should_continue": False,
            "latest_tool_result_summary": f"Unknown tool selected: {tool_name}.",
        }

    tool_input = (state.get("tool_input") or state["user_query"]).strip() or state["user_query"]
    max_results = state.get("max_results", 5)
    requested_freshness = state.get("freshness", "auto")

    timelimit = None
    if requested_freshness == "day":
        timelimit = "d"
    elif requested_freshness == "week":
        timelimit = "w"
    elif requested_freshness == "month":
        timelimit = "m"

    log_event(
        logger,
        "agent.tool.execute.start",
        tool=tool_name,
        max_results=max_results,
        freshness_bucket=requested_freshness,
        timelimit=timelimit or "none",
        tool_input_excerpt=sanitize_text(
            tool_input,
            settings.log_payload_mode,
            settings.log_payload_max_chars,
        ),
    )

    result = await tool.run(tool_input, max_results=max_results, timelimit=timelimit)

    source_meta = summarize_sources(result.sources)
    log_event(
        logger,
        "agent.tool.execute.end",
        tool=tool_name,
        duration_ms=int((time.perf_counter() - started_at) * 1000),
        summary_excerpt=sanitize_text(
            result.summary,
            settings.log_payload_mode,
            settings.log_payload_max_chars,
        ),
        **source_meta,
    )

    return {
        "sources": result.sources,
        "candidate_sources": [],
        "rewritten_query": tool_input,
        "freshness_bucket": requested_freshness,
        "search_policy": "mcp",
        "latest_tool_result_summary": result.summary,
        "tool_calls_count": state.get("tool_calls_count", 0) + 1,
        "selected_tool": tool_name,
        "tool_input": tool_input,
    }
