from __future__ import annotations

import logging
import time

from app.config.settings import get_settings
from app.observability.logging_utils import log_event, sanitize_text
from app.rag.retrieval.retriever import RagRetrievalError, RagRetriever
from app.schemas.chat import SourceItem
from app.tools.base import ToolResult, ToolSpec

logger = logging.getLogger(__name__)
settings = get_settings()


class RagRetrieveTool:
    name = "rag_retrieve"
    description = "Retrieve relevant passages from indexed internal documents."
    input_hint = "Natural-language question for indexed documents"

    def __init__(self, retriever: RagRetriever | None = None) -> None:
        self._retriever = retriever

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_hint=self.input_hint,
        )

    async def run(
        self,
        input_text: str,
        max_results: int = 5,
        timelimit: str | None = None,
    ) -> ToolResult:
        del timelimit
        started_at = time.perf_counter()
        query = input_text.strip()
        if not query:
            raise RagRetrievalError("RAG query must not be blank")
        log_event(
            logger,
            "agent.tool.execute.start",
            tool=self.name,
            max_results=max_results,
            tool_input_excerpt=sanitize_text(
                query,
                settings.log_payload_mode,
                settings.log_payload_max_chars,
            ),
        )
        sources = await self._get_retriever().retrieve(query, max_results=max_results)
        summary = f"Retrieved {len(sources)} RAG sources for query: {query}"
        log_event(
            logger,
            "agent.tool.execute.end",
            tool=self.name,
            source_count=len(sources),
            duration_ms=int((time.perf_counter() - started_at) * 1000),
        )
        return ToolResult(summary=summary, sources=self._dedup_sources(sources)[:max_results])

    @staticmethod
    def _dedup_sources(sources: list[SourceItem]) -> list[SourceItem]:
        seen: set[tuple[str, str]] = set()
        deduped: list[SourceItem] = []
        for source in sources:
            key = (source.url, source.snippet)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(source)
        return deduped

    def _get_retriever(self) -> RagRetriever:
        if self._retriever is None:
            self._retriever = RagRetriever()
        return self._retriever
