"""Processing pipelines for CitraSense.

Modality-specific processor chains live under subpackages:
- ``pipelines.common`` — shared base classes, context, result types, registry
- ``pipelines.optical`` — telescope / optical image processors
- ``pipelines.radar`` — passive-radar processors (future)
- ``pipelines.rf`` — RF processors (future)
"""

from citrasense.pipelines.common.abstract_processor import AbstractImageProcessor
from citrasense.pipelines.common.pipeline_registry import PipelineRegistry
from citrasense.pipelines.common.processing_context import ProcessingContext
from citrasense.pipelines.common.processor_result import (
    AggregatedResult,
    ProcessorResult,
)

__all__ = [
    "AbstractImageProcessor",
    "AggregatedResult",
    "PipelineRegistry",
    "ProcessingContext",
    "ProcessorResult",
]
