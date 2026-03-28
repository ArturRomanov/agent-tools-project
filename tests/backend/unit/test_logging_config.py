import json
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config.logging import configure_logging  # noqa: E402
from backend.app.config.settings import Settings  # noqa: E402


def test_json_formatter_outputs_required_keys(capsys) -> None:
    configure_logging(
        Settings(log_format="json", log_level="INFO", log_payload_mode="sanitized")
    )
    logger = logging.getLogger("backend.test.json")

    logger.info("hello", extra={"event": "test.event", "status": 200})
    output = capsys.readouterr().out.strip()

    payload = json.loads(output)
    assert payload["event"] == "test.event"
    assert payload["status"] == 200
    assert payload["logger"] == "backend.test.json"
    assert "timestamp" in payload


def test_plain_formatter_outputs_human_readable_line(capsys) -> None:
    configure_logging(Settings(log_format="plain", log_level="INFO"))
    logger = logging.getLogger("backend.test.plain")

    logger.info("plain line", extra={"event": "plain.event"})
    output = capsys.readouterr().out.strip()

    assert "plain line" in output
    assert output.startswith("20")
