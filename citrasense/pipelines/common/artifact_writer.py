"""Generic artifact-writing helpers for processing pipelines.

All public functions catch exceptions internally and log warnings — they never
block or fail the processing pipeline.  Artifacts are written to the per-task
working directory (``processing/{task_id}/``).

Modality-specific artifact dumping (FITS headers, telescope records, optical
pipeline HTML reports, etc.) lives in the modality package — see
``pipelines.optical.optical_artifacts`` for the optical pipeline.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from citrasense.pipelines.common.processor_result import AggregatedResult

logger = logging.getLogger("citrasense.ArtifactWriter")


def _safe_value(v: Any) -> Any:
    """Coerce a single value to something JSON-serialisable.

    Handles NumPy/Pandas scalar types, NaN, Path, bytes, and Timestamps.
    """
    if v is pd.NA:
        return None
    if isinstance(v, float) and (v != v):  # NaN
        return None
    if isinstance(v, np.generic):
        native = v.item()
        if isinstance(native, float) and (native != native):
            return None
        return native
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


def _sanitize(obj: Any) -> Any:
    """Recursively make a data structure JSON-safe.

    Converts NaN -> None, numpy scalars -> native Python types,
    pd.Timestamp -> ISO string, Path -> str, bytes -> str.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return _safe_value(obj)


def dump_json(working_dir: Path, filename: str, data: Any) -> None:
    """Write *data* as pretty-printed JSON to *working_dir/filename*.

    Best-effort: exceptions are logged and swallowed.
    """
    try:
        path = working_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_sanitize(data), f, indent=2, default=str)
    except Exception as exc:
        logger.warning("Failed to write artifact %s: %s", filename, exc)


def dump_csv(working_dir: Path, filename: str, dataframe: pd.DataFrame) -> None:
    """Write a pandas DataFrame as CSV to *working_dir/filename*.

    Best-effort: exceptions are logged and swallowed.
    """
    try:
        path = working_dir / filename
        dataframe.to_csv(path, index=False)
    except Exception as exc:
        logger.warning("Failed to write artifact %s: %s", filename, exc)


def task_to_dict(task: Any) -> dict:
    """Serialise a Task dataclass to a plain dict, skipping non-serialisable fields."""
    if task is None:
        return {}
    if dataclasses.is_dataclass(task) and not isinstance(task, type):
        d = {}
        for field in dataclasses.fields(task):
            if field.name.startswith("_"):
                continue
            d[field.name] = _safe_value(getattr(task, field.name))
        return d
    return {"repr": repr(task)}


def dump_processor_result(working_dir: Path, filename: str, result: Any, extra: dict | None = None) -> None:
    """Write a ProcessorResult's key fields (plus optional extras) to JSON."""
    data: dict[str, Any] = {
        "processor_name": result.processor_name,
        "confidence": result.confidence,
        "reason": result.reason,
        "processing_time_seconds": result.processing_time_seconds,
        "should_upload": result.should_upload,
        "extracted_data": {k: _safe_value(v) for k, v in result.extracted_data.items()},
    }
    if extra:
        data.update(extra)
    dump_json(working_dir, filename, data)


def dump_processing_summary(working_dir: Path, aggregated: AggregatedResult) -> None:
    """Write the full aggregated processing result to processing_summary.json."""
    processors = []
    for r in aggregated.all_results:
        processors.append(
            {
                "processor_name": r.processor_name,
                "confidence": r.confidence,
                "reason": r.reason,
                "processing_time_seconds": r.processing_time_seconds,
                "should_upload": r.should_upload,
                "extracted_data_keys": list(r.extracted_data.keys()),
            }
        )
    data = {
        "should_upload": aggregated.should_upload,
        "skip_reason": aggregated.skip_reason,
        "total_time": aggregated.total_time,
        "processors": processors,
        "extracted_data": {k: _safe_value(v) for k, v in aggregated.extracted_data.items()},
    }
    dump_json(working_dir, "processing_summary.json", data)
