from __future__ import annotations

import json
import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.graph import ResearchAgentService
from app.observability.context import get_request_id
from app.observability.logging_utils import log_event
from app.schemas.chat import ChatRequest, ChatResponse, StreamEvent

router = APIRouter()
logger = logging.getLogger(__name__)


@lru_cache
def get_research_agent_service() -> ResearchAgentService:
    return ResearchAgentService()


def _is_validation_error(exc: Exception) -> bool:
    return type(exc).__name__ == "AgentValidationError"


def _is_provider_error(exc: Exception) -> bool:
    return type(exc).__name__ in {"WebSearchError", "RagRetrievalError", "OllamaChatError"}


def _is_execution_error(exc: Exception) -> bool:
    return type(exc).__name__ == "AgentExecutionError"


def _serialize_sse_event(event: StreamEvent) -> str:
    return f"event: {event.type}\ndata: {json.dumps(event.data)}\n\n"


@router.post("/chat", response_model=ChatResponse)
async def run_chat(
    payload: ChatRequest,
    service: ResearchAgentService = Depends(get_research_agent_service),
) -> ChatResponse:
    log_event(
        logger,
        "api.chat.run.start",
        route="/chat",
        max_results=payload.max_results,
        freshness=payload.freshness,
    )
    try:
        response = await service.run(
            payload.query,
            max_results=payload.max_results,
            freshness=payload.freshness,
            session_id=payload.session_id,
            memory_mode=payload.memory_mode,
            checkpoint_id=payload.checkpoint_id,
            request_id=get_request_id(),
        )
        log_event(
            logger,
            "api.chat.run.end",
            route="/chat",
            status=200,
            source_count=len(response.sources),
        )
        return response
    except Exception as exc:
        if _is_validation_error(exc):
            log_event(
                logger,
                "api.chat.run.error",
                route="/chat",
                status=422,
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if _is_provider_error(exc):
            log_event(
                logger,
                "api.chat.run.error",
                route="/chat",
                status=502,
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if _is_execution_error(exc):
            log_event(
                logger,
                "api.chat.run.error",
                route="/chat",
                status=500,
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        log_event(
            logger,
            "api.chat.run.error",
            route="/chat",
            status=500,
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat/stream")
async def run_chat_stream(
    payload: ChatRequest,
    service: ResearchAgentService = Depends(get_research_agent_service),
) -> StreamingResponse:
    log_event(
        logger,
        "api.chat.stream.start",
        route="/chat/stream",
        max_results=payload.max_results,
        freshness=payload.freshness,
    )

    async def event_generator():
        event_counts: dict[str, int] = {}
        try:
            async for event in service.stream(
                payload.query,
                max_results=payload.max_results,
                freshness=payload.freshness,
                session_id=payload.session_id,
                memory_mode=payload.memory_mode,
                checkpoint_id=payload.checkpoint_id,
                request_id=get_request_id(),
            ):
                event_counts[event.type] = event_counts.get(event.type, 0) + 1
                log_event(
                    logger,
                    "api.chat.stream.event",
                    route="/chat/stream",
                    event_type=event.type,
                )
                yield _serialize_sse_event(event)
            log_event(
                logger,
                "api.chat.stream.end",
                route="/chat/stream",
                status=200,
                event_counts=event_counts,
            )
        except Exception as exc:
            log_event(
                logger,
                "api.chat.stream.error",
                route="/chat/stream",
                error_type=type(exc).__name__,
            )
            error_event = StreamEvent(type="error", data={"message": str(exc)})
            yield _serialize_sse_event(error_event)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
