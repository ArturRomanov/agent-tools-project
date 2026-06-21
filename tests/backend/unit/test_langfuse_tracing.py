from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config.settings import Settings  # noqa: E402
from backend.app.observability.langfuse_tracing import (  # noqa: E402
    create_langfuse_handler,
    init_langfuse,
    langfuse_attributes,
    shutdown_langfuse,
)


@pytest.fixture(autouse=True)
def _reset_langfuse_state():
    """Reset module-level state between tests."""
    import backend.app.observability.langfuse_tracing as mod

    mod._langfuse_client = None
    mod._langfuse_available = False
    yield
    mod._langfuse_client = None
    mod._langfuse_available = False


def _make_settings(**overrides) -> Settings:
    defaults = {
        "langfuse_enabled": False,
        "langfuse_secret_key": "",
        "langfuse_public_key": "",
        "langfuse_host": "http://localhost:3100",
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestInitLangfuse:
    def test_returns_false_when_disabled(self):
        settings = _make_settings(langfuse_enabled=False)
        assert init_langfuse(settings) is False

    def test_returns_false_when_enabled_but_keys_missing(self):
        settings = _make_settings(langfuse_enabled=True)
        assert init_langfuse(settings) is False

    def test_returns_false_when_secret_key_missing(self):
        settings = _make_settings(
            langfuse_enabled=True,
            langfuse_public_key="test",
            langfuse_secret_key="",
        )
        assert init_langfuse(settings) is False

    def test_returns_false_when_public_key_missing(self):
        settings = _make_settings(
            langfuse_enabled=True,
            langfuse_public_key="",
            langfuse_secret_key="test",
        )
        assert init_langfuse(settings) is False

    @patch("backend.app.observability.langfuse_tracing.Langfuse", create=True)
    def test_returns_true_on_successful_init(self, mock_langfuse_cls):
        import backend.app.observability.langfuse_tracing as mod

        mock_client = MagicMock()
        mock_langfuse_cls.return_value = mock_client

        with patch.dict("sys.modules", {"langfuse": MagicMock(Langfuse=mock_langfuse_cls)}):
            settings = _make_settings(
                langfuse_enabled=True,
                langfuse_secret_key="test",
                langfuse_public_key="test",
                langfuse_host="http://langfuse:3000",
            )
            result = init_langfuse(settings)

        assert result is True
        assert mod._langfuse_available is True
        assert mod._langfuse_client is not None

    def test_returns_false_on_import_error(self):
        import backend.app.observability.langfuse_tracing as mod

        settings = _make_settings(
            langfuse_enabled=True,
            langfuse_secret_key="test",
            langfuse_public_key="test",
        )

        with patch.dict("sys.modules", {"langfuse": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'langfuse'"),
            ):
                result = init_langfuse(settings)

        assert result is False
        assert mod._langfuse_available is False


class TestCreateLangfuseHandler:
    def test_returns_none_when_not_available(self):
        handler = create_langfuse_handler(trace_id="test")
        assert handler is None

    def test_returns_none_when_client_is_none(self):
        import backend.app.observability.langfuse_tracing as mod

        mod._langfuse_available = True
        mod._langfuse_client = None
        handler = create_langfuse_handler(trace_id="test")
        assert handler is None

    def test_returns_handler_when_available(self):
        import backend.app.observability.langfuse_tracing as mod

        mock_client = MagicMock()
        mod._langfuse_client = mock_client
        mod._langfuse_available = True

        mock_handler = MagicMock()
        with patch(
            "backend.app.observability.langfuse_tracing.CallbackHandler",
            create=True,
            return_value=mock_handler,
        ) as mock_handler_cls:
            with patch.dict(
                "sys.modules",
                {"langfuse.langchain": MagicMock(CallbackHandler=mock_handler_cls)},
            ):
                handler = create_langfuse_handler(trace_id="req-abc")

        assert handler is mock_handler


class TestLangfuseAttributes:
    def test_returns_nullcontext_when_not_available(self):
        ctx = langfuse_attributes(session_id="sess-1", tags=["chat"])
        # Should be usable as a context manager without error
        with ctx:
            pass

    def test_returns_propagate_attributes_when_available(self):
        import backend.app.observability.langfuse_tracing as mod

        mod._langfuse_available = True

        mock_ctx = MagicMock()
        mock_propagate = MagicMock(return_value=mock_ctx)
        with patch.dict(
            "sys.modules",
            {"langfuse": MagicMock(propagate_attributes=mock_propagate)},
        ):
            result = langfuse_attributes(
                session_id="sess-1", tags=["chat", "run"]
            )

        assert result is mock_ctx
        mock_propagate.assert_called_once_with(
            session_id="sess-1", tags=["chat", "run"]
        )


class TestShutdownLangfuse:
    def test_safe_when_not_initialized(self):
        # Should not raise
        shutdown_langfuse()

    def test_flushes_and_shuts_down_client(self):
        import backend.app.observability.langfuse_tracing as mod

        mock_client = MagicMock()
        mod._langfuse_client = mock_client
        mod._langfuse_available = True

        shutdown_langfuse()

        mock_client.flush.assert_called_once()
        mock_client.shutdown.assert_called_once()
        assert mod._langfuse_client is None
        assert mod._langfuse_available is False

    def test_handles_flush_exception_gracefully(self):
        import backend.app.observability.langfuse_tracing as mod

        mock_client = MagicMock()
        mock_client.flush.side_effect = RuntimeError("connection lost")
        mod._langfuse_client = mock_client
        mod._langfuse_available = True

        # Should not raise
        shutdown_langfuse()

        assert mod._langfuse_client is None
        assert mod._langfuse_available is False
