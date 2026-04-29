"""Transport-agnostic source of radar observations and status.

:class:`PassiveRadarSensor` owns exactly one :class:`DetectionSource`
implementation. The v1 implementation is
:class:`~citrasense.sensors.radar.nats_detection_source.NatsDetectionSource`
which subscribes to ``pr_sensor``'s NATS subjects. A hypothetical future
HTTP/SSE source that fronts ``pr_control`` would land in this directory as
a sibling and be selected by a settings flag — nothing else in the
codebase would need to change.

The protocol is intentionally narrow: five callback hooks for subscription
topics, lifecycle (``start`` / ``stop`` / ``is_running``), a single
request-reply entry point (``send_command``), and a staleness helper so
the sensor can surface "offline" without reaching into the source's
private state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

#: Callback type for a single incoming NATS message payload (already
#: JSON-decoded into a ``dict``).  Malformed messages are logged and
#: swallowed by the implementation, not delivered.
MessageHandler = Callable[[dict[str, Any]], None]


@runtime_checkable
class DetectionSource(Protocol):
    """Narrow contract every radar transport adapter must implement."""

    def start(
        self,
        *,
        on_observation: MessageHandler | None = None,
        on_detection: MessageHandler | None = None,
        on_status: MessageHandler | None = None,
        on_health: MessageHandler | None = None,
        on_stations: MessageHandler | None = None,
        on_error: MessageHandler | None = None,
        on_announce: MessageHandler | None = None,
        on_depart: MessageHandler | None = None,
    ) -> None:
        """Connect to the transport and subscribe to every callback
        whose handler is non-``None``.

        Idempotent: calling ``start`` again is a no-op if the source is
        already running.  Callers may swap handlers between
        ``stop`` / ``start`` cycles but not while running — mutate
        handlers via :meth:`set_handlers` instead.
        """
        ...

    def stop(self) -> None:
        """Unsubscribe, close any connections, and join worker threads.

        Idempotent.  After ``stop()`` the source may be reused via a
        fresh :meth:`start` call.
        """
        ...

    def is_running(self) -> bool:
        """Return ``True`` iff the transport is connected and dispatching."""
        ...

    def is_connected(self) -> bool:
        """Return ``True`` iff the underlying transport reports a
        healthy connection right now.

        Different from :meth:`is_running` in that the source may be
        "running" (attempting reconnects) while the transport is down.
        """
        ...

    def set_handlers(
        self,
        *,
        on_observation: MessageHandler | None = None,
        on_detection: MessageHandler | None = None,
        on_status: MessageHandler | None = None,
        on_health: MessageHandler | None = None,
        on_stations: MessageHandler | None = None,
        on_error: MessageHandler | None = None,
        on_announce: MessageHandler | None = None,
        on_depart: MessageHandler | None = None,
    ) -> None:
        """Replace one or more message handlers at runtime.

        Handlers set to ``None`` are cleared.  Any handler whose
        argument is left at the sentinel default is untouched.
        """
        ...

    def send_command(
        self,
        suffix: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Issue a request-reply on the control subject for the configured sensor.

        For the NATS implementation this maps to
        ``radar.control.{sensor_id}.{suffix}``.  Returns the decoded
        reply payload or raises :class:`TimeoutError` /
        :class:`ConnectionError` on transport failures.  Concrete
        implementations may raise their own exceptions; callers should
        treat any exception as "command failed".
        """
        ...

    def is_stream_stale(self, max_age_s: float) -> bool:
        """Return ``True`` iff the most-recent status message is older
        than *max_age_s* seconds (or no status has ever arrived).

        Used by :class:`PassiveRadarSensor` to surface an "offline" state
        when ``pr_sensor``'s 5-second heartbeat stops arriving.
        """
        ...
