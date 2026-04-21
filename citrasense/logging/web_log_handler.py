"""Log handler that streams logs to web clients via WebSocket."""

import asyncio
import logging
from collections import deque


class WebLogHandler(logging.Handler):
    """Custom log handler that buffers logs and makes them available to web clients."""

    def __init__(self, max_logs: int = 1000):
        super().__init__()
        self.log_buffer = deque(maxlen=max_logs)
        self.web_app = None
        self.loop = None

    def set_web_app(self, web_app, loop=None):
        """Set the web app instance for broadcasting logs."""
        self.web_app = web_app
        self.loop = loop

    def emit(self, record):
        """Emit a log record."""
        try:
            # Filter out routine web-related logs from the web UI
            # but keep ERROR and CRITICAL level messages
            if record.levelno < logging.ERROR:
                if (
                    record.name.startswith("uvicorn")
                    or "WebSocket" in record.getMessage()
                    or "HTTP Request:" in record.getMessage()
                ):
                    return

            # Get the original levelname without color codes
            # Record.levelname might have ANSI codes from ColoredFormatter
            level = record.levelname
            # Strip ANSI codes from level if present
            import re

            level = re.sub(r"\x1b\[\d+m", "", level)

            _prefix = "citrasense."
            short_name = record.name[len(_prefix) :] if record.name.startswith(_prefix) else record.name

            log_entry = {
                "timestamp": self.format_time(record),
                "level": level,
                "message": record.getMessage(),
                "module": short_name,
            }
            self.log_buffer.append(log_entry)

            # Broadcast to web clients if available
            if self.web_app and hasattr(self.web_app, "broadcast_log") and self.loop:
                # Schedule the broadcast in the web server's event loop
                try:
                    if self.loop.is_running():
                        asyncio.run_coroutine_threadsafe(self.web_app.broadcast_log(log_entry), self.loop)
                except Exception:
                    # Silently fail if we can't broadcast
                    pass

        except Exception:
            self.handleError(record)

    def format_time(self, record):
        """Format the timestamp."""
        from datetime import datetime

        return datetime.fromtimestamp(record.created).isoformat()

    def get_recent_logs(self, limit: int | None = None):
        """Get recent log entries."""
        if limit:
            return list(self.log_buffer)[-limit:]
        return list(self.log_buffer)
