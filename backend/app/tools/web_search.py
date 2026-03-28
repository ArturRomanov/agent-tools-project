from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Protocol

from duckduckgo_search import DDGS

from app.config.settings import get_settings
from app.observability.logging_utils import log_event, sanitize_text
from app.schemas.chat import SourceItem
from app.tools.base import ToolResult, ToolSpec


class WebSearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class WebSearchTool(Protocol):
    async def search(
        self,
        query: str,
        max_results: int = 5,
        timelimit: str | None = None,
    ) -> list[SearchResult]: ...


class DuckDuckGoWebSearchTool:
    name = "web_search"
    description = "Search the web for recent and relevant sources."
    input_hint = "A concise search query"
    _logger = logging.getLogger(__name__)
    _settings = get_settings()

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
        results = await self.search(input_text, max_results=max_results, timelimit=timelimit)
        sources = [
            SourceItem(title=result.title, url=result.url, snippet=result.snippet)
            for result in results
        ]
        summary = f"Retrieved {len(sources)} web sources for query: {input_text}"
        return ToolResult(summary=summary, sources=sources)

    async def search(
        self,
        query: str,
        max_results: int = 5,
        timelimit: str | None = None,
    ) -> list[SearchResult]:
        started_at = time.perf_counter()
        cleaned_query = query.strip()
        if not cleaned_query:
            raise WebSearchError("Search query must not be blank")

        log_event(
            self._logger,
            "tool.web_search.request",
            tool=self.name,
            max_results=max_results,
            timelimit=timelimit or "none",
            query_excerpt=sanitize_text(
                cleaned_query,
                self._settings.log_payload_mode,
                self._settings.log_payload_max_chars,
            ),
        )
        try:
            results = await asyncio.to_thread(
                self._search_sync,
                cleaned_query,
                max_results,
                timelimit,
            )
            log_event(
                self._logger,
                "tool.web_search.response",
                tool=self.name,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
                result_count=len(results),
            )
            return results
        except WebSearchError:
            raise
        except Exception as exc:
            log_event(
                self._logger,
                "tool.web_search.error",
                tool=self.name,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
                error_type=type(exc).__name__,
            )
            raise WebSearchError("Failed to execute web search") from exc

    @staticmethod
    def _search_sync(query: str, max_results: int, timelimit: str | None) -> list[SearchResult]:
        try:
            raw_results = list(DDGS().text(query, max_results=max_results, timelimit=timelimit))
        except Exception as exc:
            raise WebSearchError("Failed to execute web search") from exc

        seen_urls: set[str] = set()
        results: list[SearchResult] = []
        for item in raw_results:
            title = str(item.get("title") or "").strip()
            url = str(item.get("href") or item.get("url") or item.get("link") or "").strip()
            snippet = str(item.get("body") or item.get("snippet") or "").strip()

            if not title or not url or url in seen_urls:
                continue

            seen_urls.add(url)
            results.append(SearchResult(title=title, url=url, snippet=snippet))

            if len(results) >= max_results:
                break

        return results
