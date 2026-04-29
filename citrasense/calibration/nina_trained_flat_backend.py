"""FlatCaptureBackend that drives NINA's trained-flat wizard over REST.

Delegates capture to NINA's ``GET /flats/trained-flat`` endpoint —
operator must have already trained the Flat Wizard (per-filter
exposure + panel brightness).  Waits on the
``IMAGE-SAVE`` WebSocket events for completion, with
``/flats/status`` as a belt-and-suspenders fallback, then downloads
each FLAT frame via ``/image/{idx}?raw_fits=true`` and writes it to
the library's temp directory as a FITS file.  MasterBuilder takes
over from there and produces the stacked, normalized master exactly
as it would for direct hardware.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from citrasense.calibration import FilterSlot
from citrasense.calibration.calibration_library import CalibrationLibrary

if TYPE_CHECKING:
    from citrasense.hardware.nina.nina_adapter import NinaAdvancedHttpAdapter
    from citrasense.hardware.nina.nina_event_listener import NinaEventListener

logger = logging.getLogger("citrasense.NinaTrainedFlatBackend")

ProgressCallback = Callable[[int, int, str, str], None]


class NinaTrainedFlatBackend:
    """Drive NINA's trained-flat capture for a single filter and return FITS paths.

    NINA's ``/flats/trained-flat`` blocks on the trained exposure table,
    so we do not auto-expose here.  Waiting is a mix of REST polling
    (``/flats/status``) and WebSocket ``IMAGE-SAVE`` notifications —
    the latter is more responsive but the former guarantees we
    eventually notice completion even if the WS drops mid-job.
    """

    STATUS_POLL_INTERVAL = 5.0
    WS_WAIT_INTERVAL = 1.0
    DEADLINE_PER_FRAME_S = 45.0
    DEADLINE_BASE_S = 60.0

    def __init__(
        self,
        adapter: NinaAdvancedHttpAdapter,
        event_listener: NinaEventListener | None = None,
    ) -> None:
        self._adapter = adapter
        # Late-bind the listener so it tracks reconnects in the adapter.
        self._event_listener = event_listener
        self._local_cancel = threading.Event()

    @property
    def supported_frame_types(self) -> set[str]:
        return {"flat"}

    def cancel(self) -> None:
        self._local_cancel.set()
        try:
            self._adapter.stop_flats()
        except Exception as e:
            logger.warning("Failed to stop NINA flats on cancel: %s", e)

    def capture_flat_frames(
        self,
        *,
        filter_slot: FilterSlot | None,
        count: int,
        gain: int,
        binning: int,
        initial_exposure: float,  # ignored: NINA uses its trained table
        library: CalibrationLibrary,
        cancel_event: threading.Event | None,
        on_progress: ProgressCallback | None,
    ) -> list[Path]:
        if filter_slot is None:
            raise ValueError("NINA trained flats require a filter slot")

        self._local_cancel.clear()
        listener = self._event_listener or self._adapter.event_listener
        if listener is None:
            raise RuntimeError("NINA event listener not connected — cannot run trained flats")

        start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        self._report(
            on_progress,
            0,
            count,
            "flat",
            f"Starting NINA trained-flat wizard for {filter_slot.name or filter_slot.position}",
        )

        matched_count = [0]
        images_ready = threading.Event()

        def _on_image_saved(stats: dict[str, Any]) -> None:
            image_type = str(stats.get("ImageType", "")).upper()
            if image_type and image_type != "FLAT":
                return
            matched_count[0] += 1
            filename = stats.get("Filename", "")
            self._report(
                on_progress,
                matched_count[0],
                count,
                "flat",
                f"Captured {filename or 'frame'} ({matched_count[0]}/{count})",
            )
            if matched_count[0] >= count:
                images_ready.set()

        prev_callback = listener.on_image_save
        listener.on_image_save = _on_image_saved

        try:
            self._adapter.run_trained_flat(
                filter_id=filter_slot.position,
                count=count,
                gain=gain,
                binning=binning,
            )

            deadline = time.monotonic() + self.DEADLINE_BASE_S + self.DEADLINE_PER_FRAME_S * count
            last_status_poll = 0.0
            while matched_count[0] < count:
                if self._is_cancelled(cancel_event):
                    self._adapter.stop_flats()
                    logger.info("NINA trained flats cancelled by caller")
                    break

                if time.monotonic() > deadline:
                    logger.error("NINA trained flats timed out (%d/%d frames received)", matched_count[0], count)
                    self._adapter.stop_flats()
                    break

                # Shared wakeup: either the IMAGE-SAVE callback sets images_ready
                # (when we have hit count) or we poll /flats/status for fallback
                # completion (e.g. WS disconnected mid-job).
                images_ready.wait(timeout=self.WS_WAIT_INTERVAL)

                now = time.monotonic()
                if now - last_status_poll >= self.STATUS_POLL_INTERVAL:
                    last_status_poll = now
                    status = self._adapter.poll_flat_status()
                    state = str(status.get("State", "")).lower()
                    total = int(status.get("TotalImageCount", 0) or 0)
                    if state in ("finished", "idle") and total >= count:
                        logger.info("NINA flats reported %s with %d images — finalising", state, total)
                        break
        finally:
            listener.on_image_save = prev_callback

        frames = self._adapter.list_recent_flat_images(start_iso)
        if not frames:
            logger.error("No FLAT frames found in NINA history since %s", start_iso)
            return []

        if len(frames) > count:
            frames = frames[-count:]

        paths: list[Path] = []
        for i, row in enumerate(frames):
            idx = int(row.get("_index", -1))
            if idx < 0:
                continue
            try:
                image_bytes = self._download_image_bytes(idx)
            except Exception as e:
                logger.error("Failed to download NINA flat frame index %d: %s", idx, e)
                continue

            tmp_path = library.tmp_dir / f"flat_nina_{i:04d}_{int(time.time())}.fits"
            with open(tmp_path, "wb") as f:
                f.write(image_bytes)
            paths.append(tmp_path)

        self._report(
            on_progress,
            len(paths),
            count,
            "flat",
            f"Downloaded {len(paths)} frames from NINA for {filter_slot.name or filter_slot.position}",
        )
        return paths

    def _is_cancelled(self, cancel_event: threading.Event | None) -> bool:
        if self._local_cancel.is_set():
            return True
        return bool(cancel_event and cancel_event.is_set())

    def _download_image_bytes(self, index: int) -> bytes:
        resp = requests.get(
            f"{self._adapter.nina_api_path}/image/{index}",
            params={"raw_fits": "true"},
            timeout=self._adapter.COMMAND_TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")
        payload = resp.json()
        if not payload.get("Success"):
            raise RuntimeError(payload.get("Error", "unknown error"))
        return base64.b64decode(payload["Response"])

    @staticmethod
    def _report(
        cb: ProgressCallback | None,
        current: int,
        total: int,
        frame_type: str,
        status: str,
    ) -> None:
        if cb:
            cb(current, total, frame_type, status)
