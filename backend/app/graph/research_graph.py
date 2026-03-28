from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    AgentValidationError,
    build_chat_request,
    execute_selected_tool,
    finalize_answer,
    plan_next_step,
    prepare_query,
)
from app.graph.state import AgentState
from app.graph.tool_registry import ToolRegistry
from app.llm.ollama_chat import OllamaChatError, OllamaChatService
from app.memory import MemoryService
from app.observability.logging_utils import log_event, summarize_sources
from app.rag.retrieval import RagRetrievalError
from app.schemas.chat import ChatResponse, SourceItem, StreamEvent
from app.tools.base import AgentTool
from app.tools.rag_retrieve import RagRetrieveTool
from app.tools.web_search import DuckDuckGoWebSearchTool, WebSearchError


class AgentExecutionError(RuntimeError):
    pass


logger = logging.getLogger(__name__)


def build_research_graph(
    ollama_chat_service: OllamaChatService,
    tool_registry: ToolRegistry,
    max_tool_calls: int,
):
    graph = StateGraph(AgentState)

    async def _prepare_node(state: AgentState) -> AgentState:
        return await prepare_query(state)

    async def _planner_node(state: AgentState) -> AgentState:
        return await plan_next_step(
            state,
            ollama_chat_service=ollama_chat_service,
            tool_registry=tool_registry,
            max_tool_calls=max_tool_calls,
        )

    async def _execute_tool_node(state: AgentState) -> AgentState:
        return await execute_selected_tool(state, tool_registry=tool_registry)

    async def _finalize_node(state: AgentState) -> AgentState:
        return await finalize_answer(state)

    def _route_after_planner(state: AgentState) -> str:
        if state.get("should_continue") and state.get("selected_tool"):
            log_event(
                logger,
                "agent.graph.route_decision",
                node="planner",
                status="execute_tool",
                tool=state.get("selected_tool"),
                tool_calls_count=state.get("tool_calls_count", 0),
            )
            return "execute_tool"
        log_event(
            logger,
            "agent.graph.route_decision",
            node="planner",
            status="finalize",
            tool_calls_count=state.get("tool_calls_count", 0),
        )
        return "finalize"

    graph.add_node("prepare_query", _prepare_node)
    graph.add_node("planner", _planner_node)
    graph.add_node("execute_tool", _execute_tool_node)
    graph.add_node("finalize", _finalize_node)

    graph.add_edge(START, "prepare_query")
    graph.add_edge("prepare_query", "planner")
    graph.add_conditional_edges("planner", _route_after_planner)
    graph.add_edge("execute_tool", "planner")
    graph.add_edge("finalize", END)

    return graph.compile()


class ResearchAgentService:
    def __init__(
        self,
        ollama_chat_service: OllamaChatService | None = None,
        web_search_tool: AgentTool | None = None,
        rag_tool: AgentTool | None = None,
        max_tool_calls: int = 2,
        memory_service: MemoryService | None = None,
    ) -> None:
        self._ollama_chat_service = ollama_chat_service or OllamaChatService()
        self._web_search_tool = web_search_tool or DuckDuckGoWebSearchTool()
        self._rag_tool = rag_tool or RagRetrieveTool()
        self._memory_service = memory_service or MemoryService()
        self._tool_registry = ToolRegistry.from_tools([self._web_search_tool, self._rag_tool])
        self._max_tool_calls = max_tool_calls
        self._graph = build_research_graph(
            ollama_chat_service=self._ollama_chat_service,
            tool_registry=self._tool_registry,
            max_tool_calls=self._max_tool_calls,
        )

    @staticmethod
    def _normalize_sources(raw_sources: list[object]) -> list[SourceItem]:
        normalized: list[SourceItem] = []
        for source in raw_sources:
            if isinstance(source, SourceItem):
                normalized.append(source)
                continue
            if hasattr(source, "model_dump"):
                normalized.append(SourceItem.model_validate(source.model_dump()))
                continue
            normalized.append(SourceItem.model_validate(source))
        return normalized

    async def run(
        self,
        query: str,
        max_results: int = 5,
        freshness: str = "auto",
        session_id: str | None = None,
        memory_mode: str = "off",
        checkpoint_id: str | None = None,
        user_scope: str = "default",
        request_id: str | None = None,
    ) -> ChatResponse:
        started_at = time.perf_counter()
        context_pack = await self._memory_service.prepare_context(
            query=query,
            session_id=session_id,
            memory_mode=memory_mode,
            checkpoint_id=checkpoint_id,
            user_scope=user_scope,
            request_id=request_id,
        )
        memory_context = context_pack.context_text
        if context_pack.checkpoint_state:
            memory_context = (
                f"{memory_context}\n\nCheckpoint state:\n{context_pack.checkpoint_state}"
            ).strip()

        initial_state: AgentState = {
            "session_id": context_pack.session_id,
            "memory_mode": context_pack.memory_mode,
            "memory_context": memory_context,
            "memory_item_ids": context_pack.used_memory_item_ids,
            "checkpoint_state": context_pack.checkpoint_state,
            "user_query": query,
            "max_results": max_results,
            "freshness": freshness,
            "sources": [],
            "final_answer": "",
            "tool_calls_count": 0,
            "should_continue": True,
        }

        try:
            log_event(logger, "agent.run.start", max_results=max_results)
            output: AgentState = await self._graph.ainvoke(initial_state)
        except (AgentValidationError, WebSearchError, RagRetrievalError, OllamaChatError):
            log_event(
                logger,
                "agent.run.error",
                duration_ms=int((time.perf_counter() - started_at) * 1000),
            )
            raise
        except Exception as exc:
            log_event(
                logger,
                "agent.run.error",
                duration_ms=int((time.perf_counter() - started_at) * 1000),
                error_type=type(exc).__name__,
            )
            raise AgentExecutionError("Failed to execute research agent") from exc

        sources = self._normalize_sources(output.get("sources", []))
        checkpoint_saved_id = None
        summarized = False
        stored_memory_ids: list[str] = []

        request = build_chat_request(
            output["user_query"],
            sources,
            memory_context=output.get("memory_context"),
        )
        synthesis_response = await self._ollama_chat_service.generate(request)
        final_answer = synthesis_response.content.strip()
        try:
            persistence = await self._memory_service.persist_after_run(
                session_id=context_pack.session_id,
                user_query=query,
                assistant_answer=final_answer,
                memory_mode=context_pack.memory_mode,
                user_scope=context_pack.user_scope,
                graph_state={
                    "final_answer": final_answer,
                    "source_count": len(sources),
                },
            )
            checkpoint_saved_id = persistence.checkpoint_id
            summarized = persistence.summarized
            stored_memory_ids = persistence.stored_memory_item_ids
        except Exception:
            checkpoint_saved_id = None
            summarized = False
            stored_memory_ids = []
        source_meta = summarize_sources(sources)
        log_event(
            logger,
            "agent.run.end",
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            status="final_from_synthesis",
            **source_meta,
        )
        return ChatResponse(
            answer=final_answer,
            sources=sources,
            session_id=context_pack.session_id,
            memory_metadata={
                "memory_mode": context_pack.memory_mode,
                "used_memory_item_ids": context_pack.used_memory_item_ids,
                "summary_used": context_pack.summary_used,
                "summary_updated": summarized,
                "stored_memory_item_ids": stored_memory_ids,
                "checkpoint_id": checkpoint_saved_id,
            },
        )

    async def stream(
        self,
        query: str,
        max_results: int = 5,
        freshness: str = "auto",
        session_id: str | None = None,
        memory_mode: str = "off",
        checkpoint_id: str | None = None,
        user_scope: str = "default",
        request_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        started_at = time.perf_counter()
        context_pack = await self._memory_service.prepare_context(
            query=query,
            session_id=session_id,
            memory_mode=memory_mode,
            checkpoint_id=checkpoint_id,
            user_scope=user_scope,
            request_id=request_id,
        )
        memory_context = context_pack.context_text
        if context_pack.checkpoint_state:
            memory_context = (
                f"{memory_context}\n\nCheckpoint state:\n{context_pack.checkpoint_state}"
            ).strip()
        initial_state: AgentState = {
            "session_id": context_pack.session_id,
            "memory_mode": context_pack.memory_mode,
            "memory_context": memory_context,
            "memory_item_ids": context_pack.used_memory_item_ids,
            "checkpoint_state": context_pack.checkpoint_state,
            "user_query": query,
            "max_results": max_results,
            "freshness": freshness,
            "sources": [],
            "final_answer": "",
            "tool_calls_count": 0,
            "should_continue": True,
        }

        try:
            log_event(logger, "agent.stream.start", max_results=max_results)
            sources = []
            final_answer = ""
            sources_emitted = False
            checkpoint_saved_id = None

            if context_pack.context_text or context_pack.used_memory_item_ids:
                yield StreamEvent(
                    type="memory_loaded",
                    data={
                        "session_id": context_pack.session_id,
                        "memory_mode": context_pack.memory_mode,
                        "memory_item_ids": context_pack.used_memory_item_ids,
                        "summary_used": context_pack.summary_used,
                    },
                )

            async for update in self._graph.astream(initial_state, stream_mode="updates"):
                if not isinstance(update, dict):
                    continue

                for node_name, node_update in update.items():
                    if not isinstance(node_update, dict):
                        continue
                    log_event(
                        logger,
                        "agent.graph.node_update",
                        node=node_name,
                        tool=node_update.get("selected_tool"),
                        tool_calls_count=node_update.get("tool_calls_count"),
                    )

                    if node_name == "planner" and node_update.get("selected_tool"):
                        yield StreamEvent(
                            type="tool_selected",
                            data={
                                "tool": node_update["selected_tool"],
                                "input": node_update.get("tool_input") or query,
                            },
                        )

                    if node_name == "execute_tool":
                        sources = self._normalize_sources(node_update.get("sources", sources))
                        yield StreamEvent(
                            type="tool_result",
                            data={
                                "tool": node_update.get("selected_tool") or "web_search",
                                "summary": node_update.get("latest_tool_result_summary")
                                or "Tool execution completed.",
                            },
                        )
                        yield StreamEvent(
                            type="sources",
                            data={"sources": [source.model_dump() for source in sources]},
                        )
                        sources_emitted = True

                    if node_name == "finalize" and node_update.get("final_answer"):
                        final_answer = node_update["final_answer"]

            if not sources_emitted:
                source_payload = [source.model_dump() for source in sources]
                yield StreamEvent(type="sources", data={"sources": source_payload})

            request = build_chat_request(query, sources, memory_context=memory_context)
            answer_parts: list[str] = []
            async for chunk in self._ollama_chat_service.stream(request):
                answer_parts.append(chunk.content)
                yield StreamEvent(type="token", data={"text": chunk.content})

            final_answer = "".join(answer_parts)
            persistence = await self._memory_service.persist_after_run(
                session_id=context_pack.session_id,
                user_query=query,
                assistant_answer=final_answer,
                memory_mode=context_pack.memory_mode,
                user_scope=context_pack.user_scope,
                graph_state={
                    "final_answer": final_answer,
                    "source_count": len(sources),
                },
            )
            checkpoint_saved_id = persistence.checkpoint_id
            if persistence.summarized:
                yield StreamEvent(
                    type="memory_summarized",
                    data={"session_id": context_pack.session_id},
                )
            if checkpoint_saved_id:
                yield StreamEvent(
                    type="checkpoint_saved",
                    data={
                        "session_id": context_pack.session_id,
                        "checkpoint_id": checkpoint_saved_id,
                    },
                )
            yield StreamEvent(type="done", data={"answer": final_answer})
            source_meta = summarize_sources(sources)
            log_event(
                logger,
                "agent.stream.end",
                duration_ms=int((time.perf_counter() - started_at) * 1000),
                status="final_from_synthesis",
                chunk_count=len(answer_parts),
                **source_meta,
            )
            return

        except (AgentValidationError, WebSearchError, RagRetrievalError, OllamaChatError) as exc:
            log_event(
                logger,
                "agent.stream.error",
                duration_ms=int((time.perf_counter() - started_at) * 1000),
                error_type=type(exc).__name__,
            )
            yield StreamEvent(type="error", data={"message": str(exc)})
        except Exception:
            log_event(
                logger,
                "agent.stream.error",
                duration_ms=int((time.perf_counter() - started_at) * 1000),
                error_type="UnexpectedError",
            )
            yield StreamEvent(type="error", data={"message": "Failed to execute research agent"})
