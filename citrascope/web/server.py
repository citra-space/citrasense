"""Web server management for CitraScope."""

import asyncio
import logging
import threading
import time

import uvicorn

from citrascope.constants import DEFAULT_WEB_PORT
from citrascope.logging import CITRASCOPE_LOGGER, WebLogHandler
from citrascope.web.app import CitraScopeWebApp


class _HighFrequencyEndpointFilter(logging.Filter):
    """Suppress access-log entries for endpoints that fire many times per second."""

    _SUPPRESSED = ("/api/camera/preview",)

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(ep in msg for ep in self._SUPPRESSED)


class CitraScopeWebServer:
    """Manages the web server and its configuration."""

    def __init__(self, daemon, host: str = "0.0.0.0", port: int = DEFAULT_WEB_PORT):
        self.daemon = daemon
        self.host = host
        self.port = port
        self.web_app = None
        self.web_log_handler = None
        self._broadcast_task = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Set up web log handler
        self._setup_log_handler()

    def _setup_log_handler(self):
        """Set up the web log handler."""
        self.web_log_handler = WebLogHandler(max_logs=1000)
        # Use a simpler format for web display
        formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
        self.web_log_handler.setFormatter(formatter)
        # Set handler level to DEBUG so it captures everything
        self.web_log_handler.setLevel(logging.DEBUG)
        CITRASCOPE_LOGGER.addHandler(self.web_log_handler)
        CITRASCOPE_LOGGER.info("Web log handler attached to CITRASCOPE_LOGGER")

    def ensure_log_handler(self):
        """Ensure the web log handler is still attached to the logger."""
        if self.web_log_handler and self.web_log_handler not in CITRASCOPE_LOGGER.handlers:
            CITRASCOPE_LOGGER.addHandler(self.web_log_handler)
            CITRASCOPE_LOGGER.info("Re-attached web log handler to CITRASCOPE_LOGGER")

    def configure_uvicorn_logging(self):
        """Configure Uvicorn to use CITRASCOPE_LOGGER."""
        uvicorn_logger = logging.getLogger("uvicorn")
        uvicorn_logger.handlers = CITRASCOPE_LOGGER.handlers
        uvicorn_logger.setLevel(CITRASCOPE_LOGGER.level)
        uvicorn_logger.propagate = False

        uvicorn_access = logging.getLogger("uvicorn.access")
        uvicorn_access.handlers = CITRASCOPE_LOGGER.handlers
        uvicorn_access.setLevel(CITRASCOPE_LOGGER.level)
        uvicorn_access.propagate = False
        uvicorn_access.addFilter(_HighFrequencyEndpointFilter())

        uvicorn_error = logging.getLogger("uvicorn.error")
        uvicorn_error.handlers = CITRASCOPE_LOGGER.handlers
        uvicorn_error.setLevel(CITRASCOPE_LOGGER.level)
        uvicorn_error.propagate = False

    def start(self):
        """Start web server in a separate thread with its own event loop."""

        def run_async_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.run())
            except KeyboardInterrupt:
                pass
            finally:
                loop.close()

        thread = threading.Thread(target=run_async_server, daemon=True)
        thread.start()
        CITRASCOPE_LOGGER.info("Web server thread started")
        # Give the web server a moment to start up
        time.sleep(1)

    async def run(self):
        """Run the web server."""
        try:
            self.web_app = CitraScopeWebApp(daemon=self.daemon, web_log_handler=self.web_log_handler)

            # Store the event loop so thread-safe bridges (send_toast, log handler) can schedule work
            self._loop = asyncio.get_event_loop()

            if self.web_log_handler:
                self.web_log_handler.set_web_app(self.web_app, self._loop)

            CITRASCOPE_LOGGER.info(f"Starting web server on http://{self.host}:{self.port}")

            # Configure Uvicorn logging
            self.configure_uvicorn_logging()

            config = uvicorn.Config(
                self.web_app.app,
                host=self.host,
                port=self.port,
                log_config=None,
                access_log=True,
                ws_ping_interval=None,
            )
            server = uvicorn.Server(config)

            # Start status broadcast loop
            self._broadcast_task = asyncio.create_task(self._status_broadcast_loop())

            await server.serve()
        except OSError as e:
            if e.errno == 48:  # Address already in use
                CITRASCOPE_LOGGER.error(
                    f"Port {self.port} is already in use. Please stop any other services using this port "
                    f"or use --web-port to specify a different port."
                )
            else:
                CITRASCOPE_LOGGER.error(f"Web server OS error: {e}", exc_info=True)
        except Exception as e:
            CITRASCOPE_LOGGER.error(f"Web server error: {e}", exc_info=True)

    def send_toast(self, message: str, toast_type: str = "info", toast_id: str | None = None) -> None:
        """Thread-safe bridge: schedule a toast broadcast on the web event loop."""
        if self.web_app and self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.web_app.broadcast_toast(message, toast_type, toast_id),
                self._loop,
            )

    async def _status_broadcast_loop(self):
        """Periodically broadcast status and tasks to web clients."""
        check_counter = 0
        while True:
            try:
                await asyncio.sleep(1)  # Update every 1 second
                if self.web_app:
                    await self.web_app.broadcast_status()
                    await self.web_app.broadcast_tasks()
                    await self.web_app.broadcast_preview()

                check_counter += 1
                if check_counter >= 10:
                    check_counter = 0
                    self.ensure_log_handler()
                    if self.web_app:
                        await self.web_app.connection_manager.prune_stale()
            except Exception as e:
                CITRASCOPE_LOGGER.error(f"Status broadcast error: {e}")
