from __future__ import annotations

import json
import logging
import time

from app.config.settings import get_settings
from app.graph.planner_schema import PlannerDecision
from app.graph.state import AgentState
from app.graph.tool_registry import ToolRegistry
from app.llm.ollama_chat import ChatMessage, ChatRequest, OllamaChatService
from app.observability.logging_utils import log_event, sanitize_text

PLANNER_PROMPT = (
    "You are an agent planner. Decide whether to call a tool or answer directly. "
    "Respond only with JSON matching the schema: "
    '{"action":"call_tool|final_answer","tool_name":string|null,'
    '"tool_input":string|null,"final_answer":string|null,"reasoning_hint":string|null}. '
    "If user asks for recent/latest/today information, strongly prefer calling tools. "
    "Use rag_retrieve for questions likely answerable from indexed/private/local documents. "
    "Use web_search for public or current internet information. "
    "If tool call is needed, set action=call_tool and choose one available tool. "
    "If not needed, set action=final_answer and provide final_answer."
)
logger = logging.getLogger(__name__)
settings = get_settings()


def _extract_json(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        parts = [part.strip() for part in candidate.split("```") if part.strip()]
        for part in parts:
            if part.startswith("json"):
                return part.removeprefix("json").strip()
            if part.startswith("{"):
                return part
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        return candidate[start : end + 1]
    return candidate


async def plan_next_step(
    state: AgentState,
    ollama_chat_service: OllamaChatService,
    tool_registry: ToolRegistry,
    max_tool_calls: int,
) -> AgentState:
    started_at = time.perf_counter()
    tool_specs = tool_registry.specs()
    tool_lines = (
        "\n".join(
            f"- {spec.name}: {spec.description}. Input hint: {spec.input_hint}"
            for spec in tool_specs
        )
        or "- none"
    )

    sources = state.get("sources", [])
    source_summary = (
        "\n".join(
            f"[{idx}] {source.title} {source.url}" for idx, source in enumerate(sources, start=1)
        )
        or "No sources yet"
    )

    prompt = (
        f"User query: {state['user_query']}\n"
        f"Memory context: {state.get('memory_context') or 'none'}\n"
        f"Requested freshness mode: {state.get('freshness', 'auto')}\n"
        f"Tool calls already used: {state.get('tool_calls_count', 0)} / {max_tool_calls}\n"
        f"Last tool summary: {state.get('latest_tool_result_summary') or 'none'}\n"
        f"Available tools:\n{tool_lines}\n"
        f"Current sources:\n{source_summary}\n"
        "Decide next action now."
    )

    request = ChatRequest(
        messages=[
            ChatMessage(role="system", content=PLANNER_PROMPT),
            ChatMessage(role="user", content=prompt),
        ]
    )
    log_event(
        logger,
        "agent.planner.start",
        node="planner",
        tool_count=len(tool_specs),
        tool_calls_count=state.get("tool_calls_count", 0),
        prompt_excerpt=sanitize_text(
            prompt,
            settings.log_payload_mode,
            settings.log_payload_max_chars,
        ),
    )

    response = await ollama_chat_service.generate(request)
    response_excerpt = sanitize_text(
        response.content,
        settings.log_payload_mode,
        settings.log_payload_max_chars,
    )

    try:
        decision = PlannerDecision.model_validate(json.loads(_extract_json(response.content)))
    except Exception:
        log_event(
            logger,
            "agent.planner.error",
            node="planner",
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            error_type="InvalidPlannerJson",
            response_excerpt=response_excerpt,
        )
        if state.get("tool_calls_count", 0) < max_tool_calls:
            fallback_tool = tool_registry.first_tool_name()
            if fallback_tool is not None:
                log_event(
                    logger,
                    "agent.planner.fallback",
                    node="planner",
                    tool=fallback_tool,
                    status="invalid_json",
                )
                return {
                    "selected_tool": fallback_tool,
                    "tool_input": state["user_query"],
                    "should_continue": True,
                    "planner_action": "call_tool",
                }
        return {
            "should_continue": False,
            "planner_action": "final_answer",
            "final_answer": None,
            "latest_tool_result_summary": (
                "Planner output was invalid JSON; falling back to synthesis."
            ),
        }

    if decision.action == "call_tool":
        if state.get("tool_calls_count", 0) >= max_tool_calls:
            log_event(
                logger,
                "agent.planner.fallback",
                node="planner",
                status="tool_limit_reached",
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            return {
                "should_continue": False,
                "planner_action": "final_answer",
                "final_answer": None,
                "latest_tool_result_summary": "Reached tool call limit; falling back to synthesis.",
            }

        tool_name = (decision.tool_name or "").strip()
        if not tool_name or tool_registry.get(tool_name) is None:
            if (
                state.get("tool_calls_count", 0) < max_tool_calls
                and tool_registry.first_tool_name()
            ):
                fallback_tool = tool_registry.first_tool_name()
                log_event(
                    logger,
                    "agent.planner.fallback",
                    node="planner",
                    tool=fallback_tool,
                    status="unknown_tool",
                    duration_ms=int((time.perf_counter() - started_at) * 1000),
                )
                return {
                    "selected_tool": fallback_tool,
                    "tool_input": state["user_query"],
                    "should_continue": True,
                    "planner_action": "call_tool",
                }
            return {
                "should_continue": False,
                "planner_action": "final_answer",
                "final_answer": None,
                "latest_tool_result_summary": (
                    "Planner selected unknown tool; falling back to synthesis."
                ),
            }

        log_event(
            logger,
            "agent.planner.decision",
            node="planner",
            status=decision.action,
            tool=tool_name,
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            response_excerpt=response_excerpt,
        )

        return {
            "selected_tool": tool_name,
            "tool_input": (
                (decision.tool_input or state["user_query"]).strip() or state["user_query"]
            ),
            "should_continue": True,
            "planner_action": "call_tool",
        }
    log_event(
        logger,
        "agent.planner.decision",
        node="planner",
        status=decision.action,
        tool=decision.tool_name,
        duration_ms=int((time.perf_counter() - started_at) * 1000),
        response_excerpt=response_excerpt,
    )

    if state.get("stream_mode"):
        return {
            "should_continue": False,
            "planner_action": "final_answer",
            "final_answer": None,
            "latest_tool_result_summary": "Planner final answer suppressed for streaming.",
        }

    return {
        "should_continue": False,
        "planner_action": "final_answer",
        "final_answer": (decision.final_answer or "").strip() or None,
    }
