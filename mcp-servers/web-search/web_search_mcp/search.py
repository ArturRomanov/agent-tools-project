from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


class WebSearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


async def web_search(
    query: str,
    max_results: int = 5,
    timelimit: str | None = None,
) -> list[SearchResult]:
    cleaned_query = query.strip()
    if not cleaned_query:
        raise WebSearchError("Search query must not be blank")

    try:
        results = await asyncio.to_thread(
            _search_sync,
            cleaned_query,
            max_results,
            timelimit,
        )
        return results
    except WebSearchError:
        raise
    except Exception as exc:
        raise WebSearchError("Failed to execute web search") from exc


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
