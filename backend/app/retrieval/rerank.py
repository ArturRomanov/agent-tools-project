from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from app.tools.web_search import SearchResult

_TOKEN_RE = re.compile(r"\b\w+\b")
_RECENCY_HINT_RE = re.compile(
    r"\b(today|yesterday|hour|hours ago|just in|this week|breaking)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RankedResult:
    result: SearchResult
    score: float


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def _host(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower()


def rank_results(
    query: str,
    results: list[SearchResult],
    freshness_bucket: str,
    max_results: int,
) -> list[RankedResult]:
    query_tokens = _tokens(query)
    ranked: list[RankedResult] = []
    seen_urls: set[str] = set()

    for result in results:
        if result.url in seen_urls:
            continue
        seen_urls.add(result.url)

        haystack = f"{result.title} {result.snippet}"
        text_tokens = _tokens(haystack)

        overlap = len(query_tokens & text_tokens)
        phrase_bonus = 3.0 if query.lower() in haystack.lower() else 0.0
        recency_bonus = 0.0
        if freshness_bucket != "any" and _RECENCY_HINT_RE.search(haystack):
            recency_bonus = 2.0

        score = overlap * 2.0 + phrase_bonus + recency_bonus
        ranked.append(RankedResult(result=result, score=score))

    ranked.sort(key=lambda item: item.score, reverse=True)

    # Lightweight host diversity adjustment.
    host_counts: dict[str, int] = {}
    diversified: list[RankedResult] = []
    for item in ranked:
        host = _host(item.result.url)
        factor = host_counts.get(host, 0) * 0.25
        host_counts[host] = host_counts.get(host, 0) + 1
        diversified.append(RankedResult(result=item.result, score=item.score - factor))

    diversified.sort(key=lambda item: item.score, reverse=True)
    return diversified[:max_results]
