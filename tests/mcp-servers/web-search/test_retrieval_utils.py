"""Tests for web-search MCP server retrieval utilities (migrated from test_retrieval_quality.py)."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
WEB_SEARCH_DIR = ROOT_DIR / "mcp-servers" / "web-search"
if str(WEB_SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_SEARCH_DIR))

from web_search_mcp.retrieval.query_rewrite import rewrite_query  # noqa: E402
from web_search_mcp.retrieval.recency import (  # noqa: E402
    detect_freshness_bucket,
    freshness_to_timelimit,
)
from web_search_mcp.retrieval.rerank import rank_results  # noqa: E402
from web_search_mcp.search import SearchResult  # noqa: E402


def test_rewrite_query_normalizes_recent_new_wording() -> None:
    rewritten = rewrite_query("Recent Test new")
    assert rewritten.lower() == "recent test news"


def test_detect_freshness_bucket_auto_modes() -> None:
    assert detect_freshness_bucket("latest test news", "auto") == "week"
    assert detect_freshness_bucket("today test", "auto") == "day"
    assert detect_freshness_bucket("what is test", "auto") == "any"


def test_freshness_to_timelimit_mapping() -> None:
    assert freshness_to_timelimit("day") == "d"
    assert freshness_to_timelimit("week") == "w"
    assert freshness_to_timelimit("month") == "m"
    assert freshness_to_timelimit("any") is None


def test_rerank_prefers_overlap_and_dedupes() -> None:
    results = [
        SearchResult(
            title="Test news today",
            url="https://news.example.com/a",
            snippet="Test update from today",
        ),
        SearchResult(
            title="Completely unrelated",
            url="https://other.example.com/x",
            snippet="Nothing about the target topic",
        ),
        SearchResult(
            title="Test news duplicate",
            url="https://news.example.com/a",
            snippet="Duplicate url should be removed",
        ),
    ]

    ranked = rank_results("Test news", results, freshness_bucket="week", max_results=5)

    assert len(ranked) == 2
    assert ranked[0].result.url == "https://news.example.com/a"
    assert ranked[0].score >= ranked[1].score
