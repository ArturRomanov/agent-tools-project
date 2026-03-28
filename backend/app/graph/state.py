from __future__ import annotations

from typing import Literal, TypedDict

from app.schemas.chat import SourceItem


class AgentState(TypedDict, total=False):
    session_id: str
    memory_mode: Literal["off", "session", "long_term"]
    memory_context: str | None
    memory_item_ids: list[str]
    checkpoint_state: dict[str, object] | None
    stream_mode: bool
    user_query: str
    max_results: int
    freshness: Literal["auto", "any", "day", "week", "month"]
    sources: list[SourceItem]
    candidate_sources: list[SourceItem]
    final_answer: str
    rewritten_query: str | None
    freshness_bucket: Literal["any", "day", "week", "month"] | None
    search_policy: str | None
    selected_tool: str | None
    tool_input: str | None
    tool_calls_count: int
    should_continue: bool
    latest_tool_result_summary: str | None
    planner_action: Literal["call_tool", "final_answer"] | None
