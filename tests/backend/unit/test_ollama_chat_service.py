import asyncio
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config.settings import Settings  # noqa: E402
from backend.app.llm.ollama_chat import (  # noqa: E402
    ChatMessage,
    ChatRequest,
    OllamaChatError,
    OllamaChatService,
)


class FakeChatOllama:
    created: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.messages_seen: list[list[object]] = []
        FakeChatOllama.created.append(kwargs)

    async def ainvoke(self, messages: list[object]) -> AIMessage:
        self.messages_seen.append(messages)
        return AIMessage(
            content="hello from ollama",
            id="resp-1",
            response_metadata={"finish_reason": "stop"},
        )

    async def astream(self, messages: list[object]):
        self.messages_seen.append(messages)
        yield AIMessage(content="hel", id="chunk-1")
        yield AIMessage(content="lo", id="chunk-2")


def test_creates_client_from_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeChatOllama.created.clear()
    monkeypatch.setattr("backend.app.llm.ollama_chat.ChatOllama", FakeChatOllama)

    service = OllamaChatService(
        settings=Settings(
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="gpt-oss:120b",
            ollama_temperature=0.0,
        )
    )

    assert isinstance(service, OllamaChatService)
    assert FakeChatOllama.created[0]["base_url"] == "http://localhost:11434"
    assert FakeChatOllama.created[0]["model"] == "gpt-oss:120b"
    assert FakeChatOllama.created[0]["temperature"] == 0.0


def test_generate_maps_messages_and_returns_response() -> None:
    client = FakeChatOllama()
    service = OllamaChatService(settings=Settings(), client=client)
    request = ChatRequest(
        messages=[
            ChatMessage(role="system", content="You are concise."),
            ChatMessage(role="user", content="Say hi"),
        ]
    )

    response = asyncio.run(service.generate(request))

    assert response.content == "hello from ollama"
    assert response.model == "gpt-oss:120b"
    assert response.response_id == "resp-1"
    assert response.metadata == {"finish_reason": "stop"}
    assert len(client.messages_seen) == 1
    assert len(client.messages_seen[0]) == 2
    assert client.messages_seen[0][0].type == "system"
    assert client.messages_seen[0][1].type == "human"


def test_stream_yields_ordered_chunks() -> None:
    client = FakeChatOllama()
    service = OllamaChatService(settings=Settings(), client=client)
    request = ChatRequest(messages=[ChatMessage(role="user", content="Stream please")])

    async def collect():
        results = []
        async for chunk in service.stream(request):
            results.append(chunk)
        return results

    chunks = asyncio.run(collect())

    assert [chunk.content for chunk in chunks] == ["hel", "lo"]
    assert [chunk.response_id for chunk in chunks] == ["chunk-1", "chunk-2"]


def test_per_request_overrides_create_override_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeChatOllama.created.clear()
    monkeypatch.setattr("backend.app.llm.ollama_chat.ChatOllama", FakeChatOllama)

    service = OllamaChatService(settings=Settings())
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="Use another model")],
        model="qwen3:4b",
        temperature=0.7,
    )

    response = asyncio.run(service.generate(request))

    assert response.model == "qwen3:4b"
    assert FakeChatOllama.created[-1]["model"] == "qwen3:4b"
    assert FakeChatOllama.created[-1]["temperature"] == 0.7


def test_provider_error_is_wrapped_in_domain_error() -> None:
    class FailingClient:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            raise RuntimeError("provider down")

    service = OllamaChatService(settings=Settings(), client=FailingClient())  # type: ignore[arg-type]
    request = ChatRequest(messages=[ChatMessage(role="user", content="hello")])

    with pytest.raises(OllamaChatError) as exc_info:
        asyncio.run(service.generate(request))

    assert "Failed to generate response from Ollama" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
