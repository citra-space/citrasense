"""Transport layer for ZWO AM mount communication.

Provides an abstract transport interface and two concrete implementations:
  - **SerialTransport** — USB serial via ``pyserial``  (9600 8N1)
  - **TcpTransport** — WiFi serial-over-TCP via stdlib ``socket``

Both carry the same ``:#``-terminated command/response protocol, so the
``ZwoAmMount`` device class is transport-agnostic.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("citrascope.ZwoAmTransport")

DEFAULT_BAUD_RATE = 9600
DEFAULT_TIMEOUT_S = 2.0
DEFAULT_RETRY_COUNT = 3

# Brief pause after fire-and-forget commands to let firmware process them.
_FIRE_AND_FORGET_DELAY_S = 0.05


class ZwoAmTransport(ABC):
    """Abstract byte transport for ZWO mounts."""

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S, retry_count: int = DEFAULT_RETRY_COUNT) -> None:
        self.timeout_s = timeout_s
        self.retry_count = max(1, retry_count)
        self._lock = threading.RLock()

    @abstractmethod
    def open(self) -> None:
        """Open the underlying connection.  Raises on failure."""

    @abstractmethod
    def close(self) -> None:
        """Close the underlying connection."""

    @abstractmethod
    def is_open(self) -> bool: ...

    @abstractmethod
    def _write(self, data: bytes) -> None: ...

    @abstractmethod
    def _read_until_hash(self) -> str:
        """Read bytes until ``#`` terminator, returning the full response including ``#``."""

    @abstractmethod
    def _clear_input(self) -> None:
        """Discard any buffered incoming bytes."""

    # ---- high-level helpers (shared by both transports) ----

    def send_command(self, command: str) -> str:
        """Send *command* and return the ``#``-terminated response string."""
        with self._lock:
            self._clear_input()
            self._write(command.encode("ascii"))
            response = self._read_until_hash()
            logger.debug("TX %s  RX %s", command, response)
            return response

    def send_command_with_retry(self, command: str) -> str:
        with self._lock:
            last_error: Exception | None = None
            for attempt in range(self.retry_count):
                try:
                    self._clear_input()
                    self._write(command.encode("ascii"))
                    response = self._read_until_hash()
                    logger.debug("TX %s  RX %s", command, response)
                    return response
                except Exception as exc:
                    logger.warning("Command %r failed (attempt %d/%d): %s", command, attempt + 1, self.retry_count, exc)
                    last_error = exc
                    time.sleep(0.1)
            raise last_error  # type: ignore[misc]

    def send_command_no_response(self, command: str) -> None:
        """Send a fire-and-forget command (no response expected)."""
        with self._lock:
            self._write(command.encode("ascii"))
            logger.debug("TX %s  (no response)", command)
            time.sleep(_FIRE_AND_FORGET_DELAY_S)

    def _read_bool_response(self) -> str:
        """Read a ``1``/``0`` boolean response, tolerating missing ``#``.

        Must be called while ``_lock`` is already held and after the
        command bytes have been written.  Returns the raw buffer string
        (``"1"`` or ``"0"``).
        """
        buf = ""
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            ch = self._try_read_one()
            if ch is None:
                if buf in ("1", "0"):
                    time.sleep(0.05)
                    extra = self._try_read_one()
                    if extra and extra != "#":
                        buf += extra
                    break
                time.sleep(0.01)
                continue
            if ch == "#":
                break
            buf += ch
            if buf in ("1", "0"):
                time.sleep(0.1)
        return buf

    def send_command_bool(self, command: str) -> bool:
        """Send a command that returns ``1`` / ``0`` (possibly without ``#``).

        ZWO firmware is inconsistent — some boolean replies omit the ``#``.
        This method tolerates both forms with a short post-read wait.
        """
        with self._lock:
            self._clear_input()
            self._write(command.encode("ascii"))
            buf = self._read_bool_response()
            logger.debug("TX %s  RX(bool) %s", command, buf)
            return buf == "1"

    def send_command_bool_with_retry(self, command: str) -> bool:
        with self._lock:
            last_error: Exception | None = None
            for attempt in range(self.retry_count):
                try:
                    self._clear_input()
                    self._write(command.encode("ascii"))
                    buf = self._read_bool_response()
                    logger.debug("TX %s  RX(bool) %s", command, buf)
                    return buf == "1"
                except Exception as exc:
                    logger.warning("Bool cmd %r failed (%d/%d): %s", command, attempt + 1, self.retry_count, exc)
                    last_error = exc
                    time.sleep(0.1)
            raise last_error  # type: ignore[misc]

    def send_goto_command(self, command: str) -> str:
        """Send a GoTo-style command whose success response is a bare ``0`` (no ``#``).

        LX200 `:MS#` returns ``0`` on success (no terminator) or an error
        code like ``e6#`` / ``6#`` on failure.  Normal ``_read_until_hash``
        would hang on the success case.
        """
        with self._lock:
            self._clear_input()
            self._write(command.encode("ascii"))

            buf = ""
            deadline = time.monotonic() + self.timeout_s
            while time.monotonic() < deadline:
                ch = self._try_read_one()
                if ch is None:
                    if buf == "0":
                        logger.debug("TX %s  RX(goto) %s  [success]", command, buf)
                        return buf
                    time.sleep(0.01)
                    continue
                if ch == "#":
                    break
                buf += ch
                if buf == "0":
                    time.sleep(0.15)
                    extra = self._try_read_one()
                    if extra is None:
                        logger.debug("TX %s  RX(goto) %s  [success]", command, buf)
                        return buf
                    if extra == "#":
                        break
                    buf += extra

            logger.debug("TX %s  RX(goto) %s", command, buf)
            return buf

    @abstractmethod
    def _try_read_one(self) -> str | None:
        """Non-blocking read of a single character, or ``None`` if nothing available."""


# ---------------------------------------------------------------------------
# Serial (USB) transport
# ---------------------------------------------------------------------------


class SerialTransport(ZwoAmTransport):
    """USB serial transport using ``pyserial``."""

    def __init__(
        self,
        port: str,
        baud_rate: int = DEFAULT_BAUD_RATE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        retry_count: int = DEFAULT_RETRY_COUNT,
    ) -> None:
        super().__init__(timeout_s=timeout_s, retry_count=retry_count)
        self.port_path = port
        self.baud_rate = baud_rate
        self._serial: Any = None

    def open(self) -> None:
        import serial as _serial  # type: ignore[reportMissingImports]

        with self._lock:
            self._serial = _serial.Serial(
                port=self.port_path,
                baudrate=self.baud_rate,
                bytesize=_serial.EIGHTBITS,
                parity=_serial.PARITY_NONE,
                stopbits=_serial.STOPBITS_ONE,
                timeout=self.timeout_s,
                write_timeout=self.timeout_s,
            )
            time.sleep(0.1)

    def close(self) -> None:
        with self._lock:
            if self._serial and self._serial.is_open:
                self._serial.close()
            self._serial = None

    def __del__(self) -> None:
        self.close()

    def is_open(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def _write(self, data: bytes) -> None:
        assert self._serial is not None
        self._serial.write(data)
        self._serial.flush()

    def _read_until_hash(self) -> str:
        assert self._serial is not None
        buf = ""
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            raw = self._serial.read(1)
            if raw:
                ch = raw.decode("ascii", errors="replace")
                buf += ch
                if ch == "#":
                    return buf
            else:
                time.sleep(0.01)
        raise TimeoutError(f"Timed out waiting for response (got so far: {buf!r})")

    def _clear_input(self) -> None:
        if self._serial:
            self._serial.reset_input_buffer()

    def _try_read_one(self) -> str | None:
        assert self._serial is not None
        if self._serial.in_waiting:
            raw = self._serial.read(1)
            if raw:
                return raw.decode("ascii", errors="replace")
        return None


# ---------------------------------------------------------------------------
# TCP (WiFi) transport
# ---------------------------------------------------------------------------


class TcpTransport(ZwoAmTransport):
    """WiFi TCP transport — same serial protocol over a network socket."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        retry_count: int = DEFAULT_RETRY_COUNT,
    ) -> None:
        super().__init__(timeout_s=timeout_s, retry_count=retry_count)
        self.host = host
        self.tcp_port = port
        self._sock: socket.socket | None = None

    def open(self) -> None:
        with self._lock:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(self.timeout_s)
            self._sock.connect((self.host, self.tcp_port))
            time.sleep(0.1)

    def close(self) -> None:
        with self._lock:
            if self._sock:
                try:
                    self._sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self._sock.close()
            self._sock = None

    def __del__(self) -> None:
        self.close()

    def is_open(self) -> bool:
        return self._sock is not None

    def _write(self, data: bytes) -> None:
        assert self._sock is not None
        self._sock.sendall(data)

    def _read_until_hash(self) -> str:
        assert self._sock is not None
        buf = ""
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            try:
                raw = self._sock.recv(1)
            except TimeoutError:
                continue
            if raw:
                ch = raw.decode("ascii", errors="replace")
                buf += ch
                if ch == "#":
                    return buf
        raise TimeoutError(f"Timed out waiting for TCP response (got so far: {buf!r})")

    def _clear_input(self) -> None:
        if self._sock is None:
            return
        self._sock.setblocking(False)
        try:
            while True:
                data = self._sock.recv(256)
                if not data:
                    break
        except BlockingIOError:
            pass
        finally:
            self._sock.settimeout(self.timeout_s)

    def _try_read_one(self) -> str | None:
        assert self._sock is not None
        self._sock.setblocking(False)
        try:
            raw = self._sock.recv(1)
            if raw:
                return raw.decode("ascii", errors="replace")
        except BlockingIOError:
            pass
        finally:
            self._sock.settimeout(self.timeout_s)
        return None
