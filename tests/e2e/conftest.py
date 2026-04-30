"""E2E test infrastructure: live CitraSense server with a fresh config."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Generator

import pytest
from playwright.sync_api import Page

from citrasense.citrasense_daemon import CitraSenseDaemon
from citrasense.settings.citrasense_settings import CitraSenseSettings


def _find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def citrasense_server(tmp_path_factory: pytest.TempPathFactory) -> Generator[str, None, None]:
    """Start a CitraSense daemon with an empty config in a temp directory.

    Uses --base-dir semantics: all state (config, data, logs, cache) is
    rooted under a single temporary directory. No monkeypatching needed.
    Yields the base URL (e.g. http://127.0.0.1:12345).
    """
    base_dir = tmp_path_factory.mktemp("citrasense_e2e")
    port = _find_free_port()

    settings = CitraSenseSettings.load(web_port=port, base_dir=base_dir)

    daemon = CitraSenseDaemon(settings)
    base_url = f"http://127.0.0.1:{port}"

    # Run the daemon's web server + init in a background thread.
    # We replicate the relevant parts of daemon.run() without signal handling.
    def _run_daemon():
        assert daemon.web_server is not None
        daemon.web_server.start()
        daemon._initialize_components()
        while not daemon._stop_requested:
            time.sleep(0.5)

    thread = threading.Thread(target=_run_daemon, name="e2e-daemon", daemon=True)
    thread.start()

    # Wait for the web server to be ready
    import requests

    for _ in range(40):
        try:
            r = requests.get(f"{base_url}/api/config/status", timeout=1)
            if r.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(0.25)
    else:
        pytest.fail("CitraSense web server did not start within 10 seconds")

    yield base_url

    daemon._stop_requested = True
    thread.join(timeout=5)


@pytest.fixture
def app_page(page: Page, citrasense_server: str) -> Page:
    """Navigate a Playwright page to the CitraSense dashboard."""
    page.goto(citrasense_server)
    # Wait for Alpine.js to initialize the store
    page.wait_for_function("() => window.Alpine && Alpine.store('citrasense')")
    return page
