from __future__ import annotations

import logging
import time

from app.config.settings import get_settings
from app.graph.state import AgentState
from app.graph.tool_registry import ToolRegistry
from app.observability.logging_utils import log_event, sanitize_text, summarize_sources
from app.retrieval import (
    detect_freshness_bucket,
    freshness_to_timelimit,
    rank_results,
    rewrite_query,
)
from app.schemas.chat import SourceItem

logger = logging.getLogger(__name__)
settings = get_settings()


async def execute_selected_tool(state: AgentState, tool_registry: ToolRegistry) -> AgentState:
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

    rewritten_query = tool_input
    freshness_bucket = "any"
    timelimit = None
    retrieval_mode = "fallback_run"
    candidate_sources: list[SourceItem] = []
    result = None

    try:
        rewritten_query = rewrite_query(tool_input)
        freshness_bucket = detect_freshness_bucket(rewritten_query, requested_freshness)
        timelimit = freshness_to_timelimit(freshness_bucket)

        if hasattr(tool, "search"):
            retrieval_mode = "search_rerank"
            candidate_limit = min(max_results * 3, 20)
            raw_results = await tool.search(  # type: ignore[attr-defined]
                rewritten_query,
                max_results=candidate_limit,
                timelimit=timelimit,
            )
            ranked_results = rank_results(
                rewritten_query,
                raw_results,
                freshness_bucket=freshness_bucket,
                max_results=max_results,
            )
            final_results = [item.result for item in ranked_results]
            candidate_sources = [
                SourceItem(title=item.title, url=item.url, snippet=item.snippet)
                for item in raw_results
            ]
            sources = [
                SourceItem(title=item.title, url=item.url, snippet=item.snippet)
                for item in final_results
            ]
            top_score = ranked_results[0].score if ranked_results else 0.0
            summary = (
                f"Retrieved {len(sources)} ranked web sources "
                f"(candidates={len(raw_results)}, top_score={top_score:.2f})."
            )
            result = type("ToolResultLike", (), {"sources": sources, "summary": summary})()
    except Exception as exc:
        log_event(
            logger,
            "agent.tool.execute.retrieval_fallback",
            tool=tool_name,
            error_type=type(exc).__name__,
        )

    log_event(
        logger,
        "agent.tool.execute.start",
        tool=tool_name,
        max_results=max_results,
        search_policy=retrieval_mode,
        freshness_bucket=freshness_bucket,
        timelimit=timelimit or "none",
        rewritten_query_excerpt=sanitize_text(
            rewritten_query,
            settings.log_payload_mode,
            settings.log_payload_max_chars,
        ),
        tool_input_excerpt=sanitize_text(
            tool_input,
            settings.log_payload_mode,
            settings.log_payload_max_chars,
        ),
    )

    if result is None:
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
        candidate_count=len(candidate_sources),
        **source_meta,
    )

    return {
        "sources": result.sources,
        "candidate_sources": candidate_sources,
        "rewritten_query": rewritten_query,
        "freshness_bucket": freshness_bucket,
        "search_policy": retrieval_mode,
        "latest_tool_result_summary": result.summary,
        "tool_calls_count": state.get("tool_calls_count", 0) + 1,
        "selected_tool": tool_name,
        "tool_input": tool_input,
    }
