"""Per-sensor LoggerAdapter helper.

Wraps a logger so every record carries an ``extra={'sensor_id': ...}``
attribute that :class:`~citrasense.logging.WebLogHandler` exposes to the
web log panel for per-sensor filtering.

Use :func:`get_sensor_logger` at the top of any per-sensor scope (runtime,
adapter, manager) so downstream ``logger.info(...)`` calls automatically
tag their records.
"""

from __future__ import annotations

import logging
from typing import Any


class SensorLoggerAdapter(logging.LoggerAdapter):
    """Inject ``sensor_id`` into every record's ``extra`` dict.

    The adapter also prepends ``[sensor_id]`` to the formatted message so
    log files (which don't know about the ``extra`` attribute) still show
    which sensor emitted the message.
    """

    def __init__(self, logger: logging.Logger, sensor_id: str) -> None:
        super().__init__(logger, {"sensor_id": sensor_id})
        self._sensor_id = sensor_id

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:  # type: ignore[override]
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("sensor_id", self._sensor_id)
        return f"[{self._sensor_id}] {msg}", kwargs

    def getChild(self, suffix: str) -> SensorLoggerAdapter:
        """Return an adapter wrapping a child logger with the same ``sensor_id``.

        ``logging.LoggerAdapter`` doesn't delegate ``getChild`` by default, and
        many of our components (e.g. ``PipelineRegistry``) do
        ``logger.getChild(...)`` on whatever logger gets passed in.  Returning
        a plain logger there would drop the sensor tag, so we re-wrap.
        """
        child_logger = self.logger.getChild(suffix)
        return SensorLoggerAdapter(child_logger, self._sensor_id)


def get_sensor_logger(parent: logging.Logger | logging.LoggerAdapter, sensor_id: str) -> SensorLoggerAdapter:
    """Return a :class:`SensorLoggerAdapter` bound to ``sensor_id``.

    Accepts either a plain logger or another adapter; in the adapter case
    the underlying logger is unwrapped so we don't end up with nested
    prefixes.
    """
    base = parent.logger if isinstance(parent, logging.LoggerAdapter) else parent
    return SensorLoggerAdapter(base, sensor_id)
