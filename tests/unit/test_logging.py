"""Unit tests for logging modules."""

import logging
from unittest.mock import MagicMock

from citrasense.logging._citrasense_logger import (
    ColoredFormatter,
    ExcludeHttpRequestFilter,
    ExcludeWebLogsFilter,
)
from citrasense.logging.web_log_handler import WebLogHandler

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_exclude_http_request_filter():
    f = ExcludeHttpRequestFilter()
    record = logging.LogRecord("test", logging.INFO, "", 0, "HTTP Request: GET /api", (), None)
    assert f.filter(record) is False

    record2 = logging.LogRecord("test", logging.INFO, "", 0, "Normal log", (), None)
    assert f.filter(record2) is True


def test_exclude_web_logs_filter():
    f = ExcludeWebLogsFilter()

    uvicorn_record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, "access", (), None)
    assert f.filter(uvicorn_record) is False

    ws_record = logging.LogRecord("test", logging.INFO, "", 0, "WebSocket connected", (), None)
    assert f.filter(ws_record) is False

    http_record = logging.LogRecord("test", logging.INFO, "", 0, "HTTP Request: POST", (), None)
    assert f.filter(http_record) is False

    normal = logging.LogRecord("test", logging.INFO, "", 0, "Task started", (), None)
    assert f.filter(normal) is True


# ---------------------------------------------------------------------------
# ColoredFormatter
# ---------------------------------------------------------------------------


def test_colored_formatter_adds_colors():
    fmt = ColoredFormatter(fmt="%(levelname)s %(message)s")
    record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
    result = fmt.format(record)
    assert "hello" in result
    assert record.levelname == "INFO"


def test_colored_formatter_all_levels():
    fmt = ColoredFormatter(fmt="%(levelname)s")
    for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
        record = logging.LogRecord("test", level, "", 0, "msg", (), None)
        result = fmt.format(record)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# WebLogHandler
# ---------------------------------------------------------------------------


def test_web_log_handler_buffers_logs():
    h = WebLogHandler(max_logs=5)
    h.setFormatter(logging.Formatter("%(message)s"))
    for i in range(7):
        record = logging.LogRecord("test", logging.INFO, "", 0, f"msg-{i}", (), None)
        h.emit(record)
    assert len(h.log_buffer) == 5


def test_web_log_handler_filters_uvicorn():
    h = WebLogHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, "access log", (), None)
    h.emit(record)
    assert len(h.log_buffer) == 0


def test_web_log_handler_filters_websocket():
    h = WebLogHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("test", logging.INFO, "", 0, "WebSocket client", (), None)
    h.emit(record)
    assert len(h.log_buffer) == 0


def test_web_log_handler_keeps_errors():
    h = WebLogHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("uvicorn", logging.ERROR, "", 0, "WebSocket error", (), None)
    h.emit(record)
    assert len(h.log_buffer) == 1


def test_web_log_handler_get_recent_logs():
    h = WebLogHandler(max_logs=100)
    h.setFormatter(logging.Formatter("%(message)s"))
    for i in range(10):
        record = logging.LogRecord("test", logging.INFO, "", 0, f"msg-{i}", (), None)
        h.emit(record)
    recent = h.get_recent_logs(limit=3)
    assert len(recent) == 3
    assert recent[-1]["message"] == "msg-9"


def test_web_log_handler_get_all_logs():
    h = WebLogHandler(max_logs=100)
    h.setFormatter(logging.Formatter("%(message)s"))
    for i in range(5):
        record = logging.LogRecord("test", logging.INFO, "", 0, f"msg-{i}", (), None)
        h.emit(record)
    assert len(h.get_recent_logs()) == 5


def test_web_log_handler_format_time():
    h = WebLogHandler()
    record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
    ts = h.format_time(record)
    assert "T" in ts


def test_web_log_handler_set_web_app():
    h = WebLogHandler()
    app = MagicMock()
    h.set_web_app(app, loop=MagicMock())
    assert h.web_app is app
