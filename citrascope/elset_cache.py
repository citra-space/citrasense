"""Elset hot list: in-memory cache of latest TLEs from Citra API, file-backed."""

import json
import threading
import time
from pathlib import Path
from typing import Any

import platformdirs

from citrascope.settings.citrascope_settings import APP_AUTHOR, APP_NAME


def _normalize_api_response(raw_list: list[Any]) -> list[dict]:
    """Map API response to processor-ready list of { satellite_id, name, tle }.

    API items have satelliteId, satelliteName, tle (list of 2 strings).
    If any field is missing, derive from TLE line 2 or use a placeholder.
    """
    result = []
    for item in raw_list or []:
        if not isinstance(item, dict):
            continue
        tle = item.get("tle")
        if not tle or not isinstance(tle, list) or len(tle) < 2:
            continue
        line1, line2 = str(tle[0]).strip(), str(tle[1]).strip()
        satellite_id = item.get("satelliteId")
        if not satellite_id and len(line2) >= 12:
            # NORAD catalog number in TLE line 2, columns 3-7 (1-indexed)
            satellite_id = line2[2:7].strip() or "unknown"
        name = item.get("satelliteName") or satellite_id or "unknown"
        result.append(
            {
                "satellite_id": str(satellite_id) if satellite_id else "unknown",
                "name": str(name),
                "tle": [line1, line2],
            }
        )
    return result


def _default_cache_path() -> Path:
    """Default cache path under platform user data dir (same convention as images/processing)."""
    return Path(platformdirs.user_data_dir(APP_NAME, appauthor=APP_AUTHOR)) / "processing" / "elset_cache.json"


class ElsetCache:
    """In-memory cache of latest elsets (processor-ready: satellite_id, name, tle). File-backed."""

    def __init__(self, cache_path: Path | str | None = None):
        """Initialize cache.

        Args:
            cache_path: Path to JSON file for persistence. If None, uses default under
                platformdirs user_data_dir (processing/elset_cache.json). Pass a path to override (e.g. in tests).
        """
        if cache_path is not None:
            self._cache_path = Path(cache_path)
        else:
            self._cache_path = _default_cache_path()
        self._list: list[dict] = []
        self._lock = threading.Lock()
        self._last_refresh_epoch: float = 0.0

    def get_elsets(self) -> list[dict]:
        """Return current list of processor-ready elsets (thread-safe)."""
        with self._lock:
            return list(self._list)

    def load_from_file(self, path: Path | None = None) -> None:
        """Load cache from JSON file (processor-ready format). No normalization.

        If path is None, uses the path passed to __init__. If file does not exist, in-memory list is unchanged.
        """
        p = path if path is not None else self._cache_path
        if not p or not Path(p).exists():
            return
        try:
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, list):
                with self._lock:
                    self._list = data
                    self._last_refresh_epoch = Path(p).stat().st_mtime
        except (json.JSONDecodeError, OSError):
            pass

    def refresh(self, api_client: Any, logger: Any = None, days: int = 14) -> bool:
        """Fetch latest elsets from API, normalize once, update memory and write to file.

        Returns True if refresh succeeded, False otherwise.
        """
        raw = api_client.get_elsets_latest(days=days)
        if raw is None:
            if logger:
                logger.warning("ElsetCache: get_elsets_latest returned None")
            return False
        normalized = _normalize_api_response(raw)
        with self._lock:
            self._list = normalized
        if self._cache_path:
            try:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._cache_path, "w") as f:
                    json.dump(self._list, f, indent=0, separators=(",", ":"))
            except OSError as e:
                if logger:
                    logger.warning("ElsetCache: failed to write cache file: %s", e)
        self._last_refresh_epoch = time.time()
        if logger:
            logger.info("ElsetCache: refreshed %d elsets", len(self._list))
        return True

    def refresh_if_stale(self, api_client: Any, logger: Any = None, interval_hours: float = 6) -> bool:
        """If last refresh was more than interval_hours ago, call refresh(). Returns True if refreshed."""
        with self._lock:
            elapsed = time.time() - self._last_refresh_epoch
        if elapsed >= interval_hours * 3600:
            return self.refresh(api_client, logger=logger)
        return False
