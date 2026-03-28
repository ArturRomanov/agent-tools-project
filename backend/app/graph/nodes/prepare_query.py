from __future__ import annotations

from app.graph.state import AgentState


class AgentValidationError(ValueError):
    pass


async def prepare_query(state: AgentState) -> AgentState:
    user_query = (state.get("user_query") or "").strip()
    if not user_query:
        raise AgentValidationError("Query must not be blank")

    max_results = state.get("max_results", 5)
    if not isinstance(max_results, int) or max_results < 1:
        raise AgentValidationError("max_results must be >= 1")
    freshness = state.get("freshness", "auto")
    if freshness not in {"auto", "any", "day", "week", "month"}:
        raise AgentValidationError("freshness must be one of auto|any|day|week|month")

    return {
        "stream_mode": state.get("stream_mode"),
        "session_id": state.get("session_id"),
        "memory_mode": state.get("memory_mode", "off"),
        "memory_context": state.get("memory_context"),
        "memory_item_ids": state.get("memory_item_ids", []),
        "checkpoint_state": state.get("checkpoint_state"),
        "user_query": user_query,
        "max_results": max_results,
        "freshness": freshness,
        "sources": state.get("sources", []),
        "tool_calls_count": state.get("tool_calls_count", 0),
        "should_continue": True,
    }
