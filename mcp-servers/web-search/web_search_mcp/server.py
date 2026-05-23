from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from web_search_mcp.retrieval import (
    detect_freshness_bucket,
    freshness_to_timelimit,
    rank_results,
    rewrite_query,
)
from web_search_mcp.search import SearchResult, web_search as do_web_search

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="web-search",
    instructions="Web search tool for finding recent and relevant public information.",
    host="0.0.0.0",
    port=8001,
)


@mcp.tool()
async def web_search(
    query: str,
    max_results: int = 5,
    timelimit: str | None = None,
) -> str:
    """Search the web for recent and relevant sources using DuckDuckGo.

    Args:
        query: A concise search query.
        max_results: Maximum number of results to return (1-20).
        timelimit: Time filter — "d" for past day, "w" for past week, "m" for past month.

    Returns:
        JSON string with summary and list of sources [{title, url, snippet}].
    """
    rewritten = rewrite_query(query)
    freshness_bucket = detect_freshness_bucket(rewritten)
    effective_timelimit = timelimit or freshness_to_timelimit(freshness_bucket)

    candidate_limit = min(max_results * 3, 20)
    raw_results: list[SearchResult] = await do_web_search(
        rewritten,
        max_results=candidate_limit,
        timelimit=effective_timelimit,
    )

    ranked = rank_results(
        rewritten,
        raw_results,
        freshness_bucket=freshness_bucket,
        max_results=max_results,
    )

    sources = [
        {"title": item.result.title, "url": item.result.url, "snippet": item.result.snippet}
        for item in ranked
    ]
    top_score = f"{ranked[0].score:.2f}" if ranked else "0.00"
    summary = (
        f"Retrieved {len(sources)} ranked web sources "
        f"(candidates={len(raw_results)}, top_score={top_score})."
    )

    return json.dumps({"summary": summary, "sources": sources})
