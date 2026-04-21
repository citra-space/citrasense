"""Image processor framework for CitraSense.

This module provides a framework for processing captured images before upload.
Processors can extract readings, check quality, and decide whether to upload images.
"""

from citrasense.processors.abstract_processor import AbstractImageProcessor
from citrasense.processors.processor_registry import ProcessorRegistry
from citrasense.processors.processor_result import (
    AggregatedResult,
    ProcessingContext,
    ProcessorResult,
)

__all__ = [
    "AbstractImageProcessor",
    "AggregatedResult",
    "ProcessingContext",
    "ProcessorRegistry",
    "ProcessorResult",
]
