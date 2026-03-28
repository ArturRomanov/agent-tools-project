from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator

from app.config.settings import Settings, get_settings
from app.observability.logging_utils import log_event, sanitize_text


class OllamaChatError(RuntimeError):
    pass


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(..., description="Chat message role")
    content: str = Field(..., description="Chat message content")

    @field_validator("content")
    @classmethod
    def _content_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("content must not be blank")
        return cleaned


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float | None = None

    @field_validator("messages")
    @classmethod
    def _messages_must_not_be_empty(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if not value:
            raise ValueError("messages must not be empty")
        return value


class ChatResponse(BaseModel):
    content: str
    model: str
    response_id: str | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class StreamChunk:
    content: str
    response_id: str | None = None
    metadata: dict[str, object] | None = None


logger = logging.getLogger(__name__)


class OllamaChatService:
    def __init__(
        self,
        settings: Settings | None = None,
        client: ChatOllama | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or self._build_client(
            model=self._settings.ollama_chat_model,
            temperature=self._settings.ollama_temperature,
        )

    async def generate(self, request: ChatRequest) -> ChatResponse:
        started_at = time.perf_counter()
        client = self._resolve_client(request)
        prompt_excerpt = sanitize_text(
            self._messages_summary(request.messages),
            self._settings.log_payload_mode,
            self._settings.log_payload_max_chars,
        )
        log_event(
            logger,
            "llm.generate.start",
            model=request.model or self._settings.ollama_chat_model,
            prompt_excerpt=prompt_excerpt,
        )
        try:
            response = await client.ainvoke(self._to_langchain_messages(request.messages))
        except Exception as exc:
            log_event(
                logger,
                "llm.generate.error",
                duration_ms=int((time.perf_counter() - started_at) * 1000),
                model=request.model or self._settings.ollama_chat_model,
                error_type=type(exc).__name__,
            )
            raise OllamaChatError("Failed to generate response from Ollama") from exc

        content = self._extract_content(response)
        log_event(
            logger,
            "llm.generate.end",
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            model=request.model or self._settings.ollama_chat_model,
            response_excerpt=sanitize_text(
                content,
                self._settings.log_payload_mode,
                self._settings.log_payload_max_chars,
            ),
        )
        return ChatResponse(
            content=content,
            model=(request.model or self._settings.ollama_chat_model),
            response_id=getattr(response, "id", None),
            metadata=getattr(response, "response_metadata", None),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        started_at = time.perf_counter()
        client = self._resolve_client(request)
        chunk_count = 0
        log_event(
            logger,
            "llm.stream.start",
            model=request.model or self._settings.ollama_chat_model,
            prompt_excerpt=sanitize_text(
                self._messages_summary(request.messages),
                self._settings.log_payload_mode,
                self._settings.log_payload_max_chars,
            ),
        )
        try:
            async for chunk in client.astream(self._to_langchain_messages(request.messages)):
                content = self._extract_content(chunk)
                if not content:
                    continue
                chunk_count += 1
                log_event(
                    logger,
                    "llm.stream.chunk",
                    model=request.model or self._settings.ollama_chat_model,
                    chunk_count=chunk_count,
                )
                yield StreamChunk(
                    content=content,
                    response_id=getattr(chunk, "id", None),
                    metadata=getattr(chunk, "response_metadata", None),
                )
            log_event(
                logger,
                "llm.stream.end",
                duration_ms=int((time.perf_counter() - started_at) * 1000),
                model=request.model or self._settings.ollama_chat_model,
                chunk_count=chunk_count,
            )
        except Exception as exc:
            log_event(
                logger,
                "llm.stream.error",
                duration_ms=int((time.perf_counter() - started_at) * 1000),
                model=request.model or self._settings.ollama_chat_model,
                error_type=type(exc).__name__,
            )
            raise OllamaChatError("Failed to stream response from Ollama") from exc

    def _resolve_client(self, request: ChatRequest) -> ChatOllama:
        target_model = request.model or self._settings.ollama_chat_model
        target_temperature = self._settings.ollama_temperature
        if request.temperature is not None:
            target_temperature = request.temperature
        if request.model is None and request.temperature is None:
            return self._client
        return self._build_client(model=target_model, temperature=target_temperature)

    def _build_client(self, model: str, temperature: float) -> ChatOllama:
        client_kwargs: dict[str, float] = {}
        if self._settings.ollama_timeout_seconds is not None:
            client_kwargs["timeout"] = self._settings.ollama_timeout_seconds

        kwargs: dict[str, object] = {
            "base_url": self._settings.ollama_base_url,
            "model": model,
            "temperature": temperature,
        }
        if client_kwargs:
            kwargs["client_kwargs"] = client_kwargs
        return ChatOllama(**kwargs)

    @staticmethod
    def _to_langchain_messages(messages: Iterable[ChatMessage]) -> list[BaseMessage]:
        converted: list[BaseMessage] = []
        for message in messages:
            if message.role == "system":
                converted.append(SystemMessage(content=message.content))
            elif message.role == "assistant":
                converted.append(AIMessage(content=message.content))
            else:
                converted.append(HumanMessage(content=message.content))
        return converted

    @staticmethod
    def _extract_content(message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "".join(parts)
        return str(content)

    @staticmethod
    def _messages_summary(messages: Iterable[ChatMessage]) -> str:
        lines = []
        for message in messages:
            lines.append(f"{message.role}: {message.content}")
        return " | ".join(lines)
