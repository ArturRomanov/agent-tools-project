from __future__ import annotations

from app.llm.ollama_chat import ChatMessage, ChatRequest, OllamaChatService
from app.memory.models import TurnRecord

SUMMARY_PROMPT = (
    "You summarize chat history for long-term memory compression. "
    "Return a concise, structured summary with sections: Facts, Decisions, Open Questions. "
    "Keep it under 12 lines. Omit empty sections."
)


class SessionSummarizer:
    def __init__(self, llm_service: OllamaChatService | None = None) -> None:
        self._llm_service = llm_service or OllamaChatService()

    async def summarize_turns(self, turns: list[TurnRecord]) -> str:
        if not turns:
            return "No prior context."

        history = "\n".join(
            f"{turn.role.upper()}: {turn.content.strip()}" for turn in turns if turn.content.strip()
        )
        if not history:
            return "No significant context."

        request = ChatRequest(
            messages=[
                ChatMessage(role="system", content=SUMMARY_PROMPT),
                ChatMessage(role="user", content=history),
            ]
        )

        try:
            response = await self._llm_service.generate(request)
            summary = response.content.strip()
            return summary or "No significant context."
        except Exception:
            return self._fallback_summary(turns)

    @staticmethod
    def _fallback_summary(turns: list[TurnRecord]) -> str:
        facts: list[str] = []
        decisions: list[str] = []
        open_items: list[str] = []

        for turn in turns:
            content = turn.content.strip().replace("\n", " ")
            lowered = content.lower()
            if "?" in content:
                open_items.append(content[:180])
            elif any(token in lowered for token in ["decide", "will", "should", "plan"]):
                decisions.append(content[:180])
            else:
                facts.append(content[:180])

        sections: list[str] = []
        if facts:
            sections.append("Facts: " + " | ".join(facts[:5]))
        if decisions:
            sections.append("Decisions: " + " | ".join(decisions[:5]))
        if open_items:
            sections.append("Open Questions: " + " | ".join(open_items[:5]))
        if not sections:
            return "No significant context."
        return "\n".join(sections)
