"""Run hardware probe functions in subprocesses to isolate GIL contention.

Native SDK calls via ctypes (especially ``dlopen`` during library loading) hold
the CPython GIL.  If the hardware is unresponsive the call blocks indefinitely
and every Python thread — including the web server's event loop — freezes.

By running probes in a separate ``multiprocessing.Process``, each probe gets
its own GIL.  A hung probe is killed after a timeout without affecting the
parent process.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import multiprocessing.connection
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger("citrascope.ProbeRunner")

T = TypeVar("T")

PROBE_TIMEOUT_SECONDS = 5.0

_spawn_ctx = multiprocessing.get_context("spawn")


@dataclasses.dataclass
class _ProbeError:
    """Sentinel sent by the subprocess when the probe raises."""

    message: str


def _subprocess_target(probe_fn: Callable[[], object], send_conn: multiprocessing.connection.Connection) -> None:
    """Entry point for the probe subprocess.

    Must be module-level (not a closure) so it can be pickled by the
    ``spawn`` start method.
    """
    try:
        result = probe_fn()
        send_conn.send(result)
    except Exception as exc:
        send_conn.send(_ProbeError(str(exc)))
    finally:
        send_conn.close()


def run_hardware_probe(
    probe_fn: Callable[[], T],
    *,
    timeout: float = PROBE_TIMEOUT_SECONDS,
    fallback: T,
    description: str = "hardware probe",
) -> T:
    """Run *probe_fn* in a subprocess and return its result.

    Parameters
    ----------
    probe_fn:
        A **module-level** callable (must be picklable for the ``spawn``
        start method on macOS).  It receives no arguments and returns a
        picklable result.
    timeout:
        Maximum seconds to wait for the probe.  On expiry the subprocess
        is killed.
    fallback:
        Value returned when the probe times out or raises.
    description:
        Human-readable label for log messages.

    Returns
    -------
    The probe result on success, or *fallback* on timeout / error.
    """
    recv_conn, send_conn = _spawn_ctx.Pipe(duplex=False)

    proc = _spawn_ctx.Process(target=_subprocess_target, args=(probe_fn, send_conn), daemon=True)
    proc.start()
    send_conn.close()

    try:
        if recv_conn.poll(timeout):
            result = recv_conn.recv()
            proc.join(timeout=2)
            if isinstance(result, _ProbeError):
                logger.warning("%s raised: %s — using fallback", description, result.message)
                return fallback
            return result  # type: ignore[return-value]

        logger.warning(
            "%s timed out after %.0fs — hardware may need a power cycle",
            description,
            timeout,
        )
    except Exception:
        logger.warning("%s failed unexpectedly — using fallback", description, exc_info=True)
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
        recv_conn.close()

    return fallback
