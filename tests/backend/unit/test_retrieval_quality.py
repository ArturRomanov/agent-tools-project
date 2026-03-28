import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.retrieval.query_rewrite import rewrite_query  # noqa: E402
from backend.app.retrieval.recency import (  # noqa: E402
    detect_freshness_bucket,
    freshness_to_timelimit,
)
from backend.app.retrieval.rerank import rank_results  # noqa: E402
from backend.app.tools.web_search import DuckDuckGoWebSearchTool, SearchResult  # noqa: E402


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


def test_web_search_passes_timelimit_to_ddgs(monkeypatch) -> None:
    class FakeDDGS:
        seen_timelimit: str | None = None

        def text(self, query: str, max_results: int, timelimit: str | None = None):
            FakeDDGS.seen_timelimit = timelimit
            return [
                {
                    "title": "Test",
                    "href": "https://example.com/test",
                    "body": "Test body",
                }
            ]

    monkeypatch.setattr("backend.app.tools.web_search.DDGS", FakeDDGS)

    tool = DuckDuckGoWebSearchTool()
    results = tool._search_sync("Test", max_results=5, timelimit="w")

    assert FakeDDGS.seen_timelimit == "w"
    assert len(results) == 1
