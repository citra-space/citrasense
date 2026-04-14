"""Abstract base class for all hardware device types."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar, TypeVar

from citrascope.hardware.abstract_astro_hardware_adapter import SettingSchemaEntry

_T = TypeVar("_T")


class AbstractHardwareDevice(ABC):
    """Base class for all hardware devices (cameras, mounts, filter wheels, focusers).

    Provides common interface elements shared by all device types.
    """

    logger: logging.Logger

    _hardware_probe_cache: ClassVar[dict[str, tuple[object, float]]] = {}

    def __init__(self, logger: logging.Logger, **kwargs):
        """Initialize the hardware device.

        Args:
            logger: Logger instance for this device
            **kwargs: Device-specific configuration parameters
        """
        self.logger = logger.getChild(type(self).__name__)

    @classmethod
    @abstractmethod
    def get_friendly_name(cls) -> str:
        """Return human-readable name for this device.

        Returns:
            Friendly display name (e.g., 'ZWO ASI294MC Pro', 'Celestron CGX')
        """
        pass

    @classmethod
    @abstractmethod
    def get_dependencies(cls) -> dict[str, str | list[str]]:
        """Return required Python packages and installation info.

        Returns:
            Dict with keys:
                - packages: list of required package names
                - install_extra: pyproject.toml extra name for pip install
        """
        pass

    @classmethod
    @abstractmethod
    def get_settings_schema(cls) -> list[SettingSchemaEntry]:
        """Return schema describing configurable settings for this device.

        Returns:
            List of setting schema entries (without device-type prefix)
        """
        pass

    @classmethod
    def _cached_hardware_probe(
        cls,
        probe_fn: Callable[[], _T],
        *,
        fallback: _T,
        cache_key: str = "default",
        cache_ttl: float = float("inf"),
        timeout: float = 10.0,
    ) -> _T:
        """Run a hardware probe in a subprocess with result caching.

        Subclasses that enumerate native hardware (cameras, focusers, etc.)
        should call this instead of invoking native SDKs directly.  The probe
        runs in a separate process with its own GIL and is killed if it
        exceeds *timeout*, preventing a hung USB device from freezing the
        web server.

        Parameters
        ----------
        probe_fn:
            **Module-level** picklable callable (no arguments, picklable
            return value).  Must be defined at module scope so the ``spawn``
            start method on macOS can pickle it.
        fallback:
            Returned when the probe times out, raises, or cannot start.
        cache_key:
            Distinguishes multiple independent probes on the same device
            class (e.g. ``"cameras"`` vs ``"read_modes"``).  Defaults to
            ``"default"``.
        cache_ttl:
            Seconds to cache a probe result (success or fallback).
            Defaults to infinity — use ``_clear_probe_cache`` or the
            ``POST /api/hardware/scan`` endpoint to force a refresh.
        timeout:
            Maximum seconds before the probe subprocess is killed.
        """
        from citrascope.hardware.probe_runner import run_hardware_probe

        full_key = f"{cls.__module__}.{cls.__qualname__}:{cache_key}"

        entry = cls._hardware_probe_cache.get(full_key)
        if entry is not None:
            cached_result, cached_at = entry
            if time.time() - cached_at < cache_ttl:
                return cached_result  # type: ignore[return-value]

        result = run_hardware_probe(
            probe_fn,
            timeout=timeout,
            fallback=fallback,
            description=f"{cls.__qualname__} {cache_key} probe",
        )

        cls._hardware_probe_cache[full_key] = (result, time.time())
        return result

    @classmethod
    def _clear_probe_cache(cls, cache_key: str = "default") -> None:
        """Evict a cached probe result, forcing a fresh probe on next call.

        Useful for "Scan Hardware" buttons or after a reconnect.
        """
        full_key = f"{cls.__module__}.{cls.__qualname__}:{cache_key}"
        cls._hardware_probe_cache.pop(full_key, None)

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the hardware device.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from the hardware device."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if device is connected and responsive.

        Returns:
            True if connected, False otherwise
        """
        pass
