"""Data classes for pipeline processor output."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProcessorResult:
    """Result returned by processors."""

    should_upload: bool  # False = skip upload
    extracted_data: dict  # Metrics to attach to upload
    confidence: float  # 0.0-1.0 quality score
    reason: str  # Human-readable explanation
    processing_time_seconds: float  # For metrics
    processor_name: str  # Which processor returned this


@dataclass
class AggregatedResult:
    """Combined results from all processors."""

    should_upload: bool  # AND of all processor results
    extracted_data: dict  # Merged extracted data
    all_results: list[ProcessorResult]  # Individual results
    total_time: float  # Total processing time
    skip_reason: str | None  # Why upload was skipped (if any)
