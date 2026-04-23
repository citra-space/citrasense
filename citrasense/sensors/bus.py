"""In-process event bus seam for sensors.

The :class:`SensorBus` protocol is the only way streaming sensors hand events
off to the rest of the daemon. Today the single shipped implementation is
:class:`InProcessBus` — a plain in-process pub/sub that synchronously fans out
to any handler whose subject pattern matches. Tomorrow an identical-shaped
broker-backed implementation (e.g. NATS) can slot in without rewriting sensor
code.

Subject naming convention (documented, not enforced)
----------------------------------------------------

Streaming sensors should publish under the following hierarchical subjects, so
subscribers can use glob patterns like ``sensors.*.events.acquisition``::

    sensors.{sensor_id}.events.status            # connection, health, state transitions
    sensors.{sensor_id}.events.acquisition       # AcquisitionEvent from streaming sensors
    sensors.{sensor_id}.events.frame_captured    # on-demand sensors announcing frames

Validators that enforce this convention can land later; phase 1 just writes it
down.

Scope notes (intentional non-goals for phase 1)
-----------------------------------------------

* No broker dependency (no ``nats-py``, ``paho-mqtt``, etc.).
* No wire schemas / serialization commitments — events are in-process
  pydantic ``BaseModel`` instances.
* No schema versioning.
* No harmonization with :mod:`citrasense.preview_bus` (tracked as a follow-up).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from fnmatch import fnmatchcase
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class Subscription(Protocol):
    """Opaque handle returned by :meth:`SensorBus.subscribe`.

    Callers hold it only long enough to eventually invoke
    :meth:`unsubscribe`. Identity/equality is not defined — don't key dicts
    on it.
    """

    def unsubscribe(self) -> None:
        """Stop delivering events to the associated handler. Idempotent."""


@runtime_checkable
class SensorBus(Protocol):
    """In-process pub/sub seam for sensor events.

    Deliberately a two-method protocol so an out-of-process broker can
    implement the same interface later. Handlers receive
    ``(subject, event)``; delivery order within a subject is publish order.
    """

    def publish(self, subject: str, event: BaseModel) -> None:
        """Synchronously fan out ``event`` to every matching subscriber."""
        ...

    def subscribe(
        self,
        subject_pattern: str,
        handler: Callable[[str, BaseModel], None],
    ) -> Subscription:
        """Register ``handler`` for subjects matching ``subject_pattern``.

        ``subject_pattern`` uses :func:`fnmatch.fnmatchcase` semantics
        (``*`` matches any single segment-less substring, ``?`` a single
        character). A literal pattern equal to the subject always matches.
        """
        ...


class _InProcessSubscription:
    """Concrete :class:`Subscription` backed by a reference to the owning bus."""

    __slots__ = ("_active", "_bus", "_handler", "_pattern")

    def __init__(
        self,
        bus: InProcessBus,
        pattern: str,
        handler: Callable[[str, BaseModel], None],
    ) -> None:
        self._bus = bus
        self._pattern = pattern
        self._handler = handler
        self._active = True

    def unsubscribe(self) -> None:
        if not self._active:
            return
        self._active = False
        self._bus._remove(self._pattern, self._handler)


class InProcessBus:
    """Phase-1 in-process pub/sub. Synchronous dispatch on publish.

    Thread-safe for concurrent ``publish`` / ``subscribe`` / unsubscribe.
    ``publish`` snapshots the handlers matching the current subject while
    holding the lock, then dispatches outside the lock (avoids reentrancy
    deadlocks if a handler publishes during its own execution).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # {pattern: [handlers]} — ordered by registration.
        self._subs: dict[str, list[Callable[[str, BaseModel], None]]] = {}

    def publish(self, subject: str, event: BaseModel) -> None:
        with self._lock:
            # Snapshot the handlers that match so dispatch is lock-free.
            matched: list[Callable[[str, BaseModel], None]] = []
            for pattern, handlers in self._subs.items():
                if pattern == subject or fnmatchcase(subject, pattern):
                    matched.extend(handlers)

        for handler in matched:
            handler(subject, event)

    def subscribe(
        self,
        subject_pattern: str,
        handler: Callable[[str, BaseModel], None],
    ) -> Subscription:
        with self._lock:
            self._subs.setdefault(subject_pattern, []).append(handler)
        return _InProcessSubscription(self, subject_pattern, handler)

    def _remove(
        self,
        pattern: str,
        handler: Callable[[str, BaseModel], None],
    ) -> None:
        with self._lock:
            handlers = self._subs.get(pattern)
            if not handlers:
                return
            try:
                handlers.remove(handler)
            except ValueError:
                return
            if not handlers:
                del self._subs[pattern]
