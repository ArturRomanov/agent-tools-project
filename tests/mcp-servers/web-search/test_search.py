"""Tests for web-search MCP server search module (migrated from test_retrieval_quality.py)."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
WEB_SEARCH_DIR = ROOT_DIR / "mcp-servers" / "web-search"
if str(WEB_SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_SEARCH_DIR))

from web_search_mcp import search as search_module  # noqa: E402
from web_search_mcp.search import _search_sync  # noqa: E402


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

    monkeypatch.setattr(search_module, "DDGS", FakeDDGS)

    results = _search_sync("Test", max_results=5, timelimit="w")

    assert FakeDDGS.seen_timelimit == "w"
    assert len(results) == 1
