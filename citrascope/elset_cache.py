"""Elset hot list: in-memory cache of latest TLEs from Citra API, file-backed."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import platformdirs

from citrascope.settings.citrascope_settings import APP_AUTHOR, APP_NAME

_LOW_COUNT_THRESHOLD = 25_000

_logger = logging.getLogger(__name__)


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
        self._source: str = ""

    @classmethod
    def from_snapshot(cls, elsets: list[dict]) -> ElsetCache:
        """Create an in-memory-only cache pre-populated with the given elset list.

        Used by the reprocessing tool to reconstruct the elset state that was
        captured in ``elset_cache_snapshot.json`` at original processing time.
        The returned cache has no file backing and will not write to disk.
        """
        import tempfile

        cache = cls.__new__(cls)
        cache._cache_path = None  # type: ignore[assignment]
        cache._list = []
        cache._lock = threading.Lock()
        cache._last_refresh_epoch = 0.0
        cache._source = "snapshot"

        if elsets:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(elsets, f)
                tmp_path = Path(f.name)
            try:
                cache.load_from_file(expected_source="", path=tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)

        return cache

    def _clear(self) -> None:
        """Reset in-memory state under the lock."""
        with self._lock:
            self._list = []
            self._source = ""
            self._last_refresh_epoch = 0.0

    def get_elsets(self) -> list[dict]:
        """Return current list of processor-ready elsets (thread-safe)."""
        with self._lock:
            return list(self._list)

    def get_health(self) -> dict[str, Any]:
        """Thread-safe snapshot of cache health for status broadcasts."""
        with self._lock:
            return {
                "elset_count": len(self._list),
                "last_refresh": self._last_refresh_epoch,
                "source": self._source,
            }

    def load_from_file(self, expected_source: str = "", path: Path | None = None) -> None:
        """Load cache from JSON file if the source tag matches.

        Supports both the new tagged format (dict with "source" + "elsets" keys) and
        the legacy bare-list format.  If expected_source is non-empty and doesn't match
        the stored source, the cached data is discarded (a refresh will be needed).
        """
        p = path if path is not None else self._cache_path
        if not p or not Path(p).exists():
            return
        try:
            with open(p) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        if isinstance(data, dict) and "elsets" in data:
            stored_source = data.get("source", "")
            elsets = data["elsets"]
            if not isinstance(elsets, list):
                _logger.warning("ElsetCache: corrupt cache file (elsets is not a list) — discarding")
                self._clear()
                return
            if expected_source and stored_source != expected_source:
                _logger.warning(
                    "ElsetCache: source mismatch (file=%s, expected=%s) — discarding cached data",
                    stored_source,
                    expected_source,
                )
                self._clear()
                return
        elif isinstance(data, list):
            if expected_source:
                _logger.warning("ElsetCache: legacy cache format (no source tag) — discarding cached data")
                self._clear()
                return
            elsets = data
            stored_source = ""
        else:
            return

        with self._lock:
            self._list = elsets
            self._source = stored_source
            self._last_refresh_epoch = Path(p).stat().st_mtime

    def refresh(self, api_client: Any, logger: Any = None, days: int = 14) -> bool:
        """Fetch latest elsets from API, normalize once, update memory and write to file.

        Returns True if refresh succeeded, False otherwise.
        """
        source_key: str = getattr(api_client, "cache_source_key", type(api_client).__name__)
        try:
            raw = api_client.get_elsets_latest(days=days)
        except Exception as e:
            if logger:
                logger.warning("ElsetCache: get_elsets_latest raised %s: %s", type(e).__name__, e)
            return False
        if raw is None:
            if logger:
                logger.warning("ElsetCache: get_elsets_latest returned None")
            return False
        normalized = _normalize_api_response(raw)
        now = time.time()
        with self._lock:
            self._list = normalized
            self._source = source_key
            self._last_refresh_epoch = now
        if self._cache_path:
            try:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                wrapper = {
                    "source": source_key,
                    "refreshed_at": datetime.now(timezone.utc).isoformat(),
                    "elsets": normalized,
                }
                with open(self._cache_path, "w") as f:
                    json.dump(wrapper, f, indent=0, separators=(",", ":"))
            except OSError as e:
                if logger:
                    logger.warning("ElsetCache: failed to write cache file: %s", e)
        count = len(normalized)
        if logger:
            logger.info("ElsetCache: refreshed %d elsets (source=%s)", count, source_key)
            if count < _LOW_COUNT_THRESHOLD:
                logger.warning(
                    "ElsetCache: only %d elsets loaded (expected >= %d) — satellite matching may miss targets",
                    count,
                    _LOW_COUNT_THRESHOLD,
                )
        return True

    def refresh_if_stale(self, api_client: Any, logger: Any = None, interval_hours: float = 6) -> bool:
        """If last refresh was more than interval_hours ago, call refresh(). Returns True if refreshed."""
        with self._lock:
            elapsed = time.time() - self._last_refresh_epoch
        if elapsed >= interval_hours * 3600:
            return self.refresh(api_client, logger=logger)
        return False
