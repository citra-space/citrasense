"""Abstract base class for image processors."""

from abc import ABC, abstractmethod

from citrasense.processors.processor_result import ProcessingContext, ProcessorResult


class AbstractImageProcessor(ABC):
    """Base class for image processors.

    Processors analyze captured images and return results that include:
    - Upload decision (should this image be uploaded?)
    - Extracted data (metrics, readings, etc.)
    - Confidence score and human-readable reason

    Processors should be stateless and thread-safe.
    """

    name: str  # Processor identifier (e.g., "quality_checker")
    friendly_name: str  # Human-readable name (e.g., "Quality Checker")
    description: str  # Brief description of what the processor does

    @abstractmethod
    def process(self, context: ProcessingContext) -> ProcessorResult:
        """Process an image and return results.

        Args:
            context: ProcessingContext with image data, task, and observatory info

        Returns:
            ProcessorResult with upload decision and extracted data

        Raises:
            Exception: Processors should raise exceptions for fatal errors.
                      The registry will catch them and fail-open (allow upload).
        """
        pass
