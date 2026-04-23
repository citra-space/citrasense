"""Shared pipeline scaffolding — base classes, context, result types, registry."""

from citrasense.pipelines.common.abstract_processor import AbstractImageProcessor
from citrasense.pipelines.common.processing_context import ProcessingContext
from citrasense.pipelines.common.processor_result import (
    AggregatedResult,
    ProcessorResult,
)

__all__ = [
    "AbstractImageProcessor",
    "AggregatedResult",
    "ProcessingContext",
    "ProcessorResult",
]
