"""Lightweight in-process background job runner.

Used by batch reprocess, multi-select reprocess, and auto-tune — anything
that takes too long for a synchronous API response.  Not a full task queue;
just enough for "run a function in a thread, report progress, fetch result."
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("citrascope.Jobs")


@dataclass
class JobStatus:
    """Snapshot of a background job's progress."""

    job_id: str
    state: str = "pending"  # pending, running, completed, failed, cancelled
    progress: int = 0
    total: int = 0
    result: Any = None
    error: str | None = None
    per_item_results: list[dict] = field(default_factory=list)
    cancelled: bool = False
    _lock: threading.Lock = field(init=False, repr=False, compare=False, default_factory=threading.Lock)

    def append_item_result(self, item: dict) -> None:
        """Thread-safe append to per_item_results."""
        with self._lock:
            self.per_item_results.append(item)

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "job_id": self.job_id,
                "state": self.state,
                "progress": self.progress,
                "total": self.total,
                "result": self.result,
                "error": self.error,
                "per_item_results": list(self.per_item_results),
            }


class BackgroundJobRunner:
    """Runs callables in background threads and tracks their progress.

    Thread-safe: multiple web handlers can poll status concurrently.
    """

    MAX_RETAINED_JOBS = 50

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobStatus] = {}

    def submit(self, fn: Callable[[JobStatus], None], *, total: int = 0) -> JobStatus:
        """Start *fn* in a daemon thread.

        The callable receives a ``JobStatus`` object and is responsible for
        updating ``progress``, ``per_item_results``, and ``result`` as it
        works.  The runner sets ``state`` to ``"running"`` before the call
        and to ``"completed"`` or ``"failed"`` after.
        """
        job_id = uuid.uuid4().hex[:12]
        status = JobStatus(job_id=job_id, state="running", total=total)

        with self._lock:
            self._jobs[job_id] = status
            self._evict_old()

        def _worker() -> None:
            try:
                fn(status)
                status.state = "completed"
            except Exception as exc:
                logger.exception("Background job %s failed", job_id)
                status.state = "failed"
                status.error = str(exc)

        t = threading.Thread(target=_worker, daemon=True, name=f"job-{job_id}")
        t.start()
        return status

    def cancel(self, job_id: str) -> bool:
        """Request cooperative cancellation of a running job.

        Sets the ``cancelled`` flag on the ``JobStatus``.  The worker
        function must check ``status.cancelled`` periodically and exit
        early when it becomes ``True``.

        Returns ``True`` if the job was found and flagged, ``False`` if
        the job ID was unknown or already finished.
        """
        with self._lock:
            status = self._jobs.get(job_id)
        if status is None or status.state not in ("pending", "running"):
            return False
        status.cancelled = True
        return True

    def get_status(self, job_id: str) -> JobStatus | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[JobStatus]:
        with self._lock:
            return list(self._jobs.values())

    def _evict_old(self) -> None:
        """Drop oldest completed/failed jobs when we exceed the retention cap."""
        if len(self._jobs) <= self.MAX_RETAINED_JOBS:
            return
        finished = [(jid, js) for jid, js in self._jobs.items() if js.state in ("completed", "failed", "cancelled")]
        to_remove = len(self._jobs) - self.MAX_RETAINED_JOBS
        for jid, _ in finished[:to_remove]:
            del self._jobs[jid]
