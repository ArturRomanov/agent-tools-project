from __future__ import annotations

from app.llm.ollama_chat import ChatMessage, ChatRequest
from app.schemas.chat import SourceItem

SYSTEM_PROMPT = (
    "You are a careful research assistant. Use provided sources when available. "
    "If sources are insufficient, say so clearly."
)


def build_chat_request(
    user_query: str,
    sources: list[SourceItem],
    memory_context: str | None = None,
) -> ChatRequest:
    if sources:
        source_lines = "\n".join(
            f"[{idx}] {source.title} - {source.url}\n{source.snippet}"
            for idx, source in enumerate(sources, start=1)
        )
    else:
        source_lines = "No search results found. Explain that sources were unavailable."

    user_prompt = (
        f"User question:\n{user_query}\n\n"
        f"Relevant memory context:\n{memory_context or 'No persisted memory context.'}\n\n"
        "Search results:\n"
        f"{source_lines}\n\n"
        "Answer the question and cite source URLs inline when possible."
    )

    return ChatRequest(
        messages=[
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]
    )
