"""Unit tests for image processor framework."""

import time
from unittest.mock import Mock

import numpy as np
import pytest

from citrasense.pipelines.common.abstract_processor import AbstractImageProcessor
from citrasense.pipelines.common.pipeline_registry import PipelineRegistry
from citrasense.pipelines.common.processing_context import ProcessingContext
from citrasense.pipelines.common.processor_result import AggregatedResult, ProcessorResult
from citrasense.tasks.task import Task


class MockPassProcessor(AbstractImageProcessor):
    """Mock processor that always passes."""

    name = "mock_pass"
    friendly_name = "Mock Pass Processor"
    description = "Test processor that always passes"

    def process(self, context: ProcessingContext) -> ProcessorResult:
        return ProcessorResult(
            should_upload=True,
            extracted_data={"test_value": 42},
            confidence=0.9,
            reason="Test pass",
            processing_time_seconds=0.001,
            processor_name=self.name,
        )


class MockRejectProcessor(AbstractImageProcessor):
    """Mock processor that always rejects."""

    name = "mock_reject"
    friendly_name = "Mock Reject Processor"
    description = "Test processor that always rejects"

    def process(self, context: ProcessingContext) -> ProcessorResult:
        return ProcessorResult(
            should_upload=False,
            extracted_data={"test_value": 0},
            confidence=0.1,
            reason="Test rejection",
            processing_time_seconds=0.001,
            processor_name=self.name,
        )


class MockErrorProcessor(AbstractImageProcessor):
    """Mock processor that raises an error."""

    name = "mock_error"
    friendly_name = "Mock Error Processor"
    description = "Test processor that raises an error"

    def process(self, context: ProcessingContext) -> ProcessorResult:
        raise RuntimeError("Test error")


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return Mock()


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.processors_enabled = True
    return settings


@pytest.fixture
def mock_task():
    """Mock task for testing."""
    return Task(
        id="test-task-123",
        type="observation",
        status="Scheduled",
        creationEpoch="2024-01-01T00:00:00Z",
        updateEpoch="2024-01-01T00:00:00Z",
        taskStart="2024-01-01T01:00:00Z",
        taskStop="2024-01-01T01:05:00Z",
        userId="user-123",
        username="testuser",
        satelliteId="sat-123",
        satelliteName="Test Satellite",
        telescopeId="tel-123",
        telescopeName="Test Telescope",
        groundStationId="gs-123",
        groundStationName="Test Ground Station",
        assigned_filter_name="Red",
    )


@pytest.fixture
def processing_context(tmp_path, mock_task):
    """Create a processing context with mock data."""
    # Create a dummy FITS file path (doesn't need to exist for most tests)
    image_path = tmp_path / "test_image.fits"

    # Create working directory
    working_dir = tmp_path / "working"
    working_dir.mkdir(exist_ok=True)

    # Create mock image data
    image_data = np.random.randint(0, 1000, size=(100, 100), dtype=np.uint16)

    return ProcessingContext(
        image_path=image_path,
        working_image_path=image_path,
        working_dir=working_dir,
        image_data=image_data,
        task=mock_task,
        telescope_record={"id": "tel-123", "name": "Test Telescope"},
        ground_station_record={"id": "gs-123", "name": "Test Station"},
        settings=Mock(),
    )


class TestProcessorResult:
    """Tests for ProcessorResult data class."""

    def test_processor_result_creation(self):
        """Test creating a ProcessorResult."""
        result = ProcessorResult(
            should_upload=True,
            extracted_data={"key": "value"},
            confidence=0.9,
            reason="Test reason",
            processing_time_seconds=0.5,
            processor_name="test_processor",
        )

        assert result.should_upload is True
        assert result.extracted_data == {"key": "value"}
        assert result.confidence == 0.9
        assert result.reason == "Test reason"
        assert result.processing_time_seconds == 0.5
        assert result.processor_name == "test_processor"


class TestProcessingContext:
    """Tests for ProcessingContext data class."""

    def test_context_with_task(self, processing_context):
        """Test context with task data."""
        assert processing_context.task is not None
        assert processing_context.task.id == "test-task-123"
        assert processing_context.task.satelliteName == "Test Satellite"

    def test_context_without_task(self, tmp_path):
        """Test context without task (manual capture)."""
        working_dir = tmp_path / "working"
        working_dir.mkdir(exist_ok=True)

        context = ProcessingContext(
            image_path=tmp_path / "test.fits",
            working_image_path=tmp_path / "test.fits",
            working_dir=working_dir,
            image_data=None,
            task=None,
            telescope_record=None,
            ground_station_record=None,
            settings=None,
        )

        assert context.task is None


class TestPipelineRegistry:
    """Tests for PipelineRegistry."""

    def test_registry_initialization(self, mock_settings, mock_logger):
        """Test registry initialization."""
        registry = PipelineRegistry(mock_settings, mock_logger)
        assert registry.settings == mock_settings
        assert registry.logger == mock_logger.getChild("PipelineRegistry")
        assert isinstance(registry.processors, list)

    def test_process_all_with_pass_processor(self, mock_settings, mock_logger, processing_context):
        """Test processing with a processor that passes."""
        registry = PipelineRegistry(mock_settings, mock_logger)
        registry.processors = [MockPassProcessor()]

        result = registry.process_all(processing_context)

        assert isinstance(result, AggregatedResult)
        assert result.should_upload is True
        assert "mock_pass.test_value" in result.extracted_data
        assert result.extracted_data["mock_pass.test_value"] == 42
        assert len(result.all_results) == 1

    def test_process_all_with_reject_processor(self, mock_settings, mock_logger, processing_context):
        """Test processing with a processor that rejects."""
        registry = PipelineRegistry(mock_settings, mock_logger)
        registry.processors = [MockRejectProcessor()]

        result = registry.process_all(processing_context)

        assert isinstance(result, AggregatedResult)
        assert result.should_upload is False
        assert result.skip_reason == "mock_reject: Test rejection"

    def test_process_all_with_multiple_processors(self, mock_settings, mock_logger, processing_context):
        """Test processing with multiple processors."""
        registry = PipelineRegistry(mock_settings, mock_logger)
        registry.processors = [MockPassProcessor(), MockPassProcessor()]

        result = registry.process_all(processing_context)

        assert result.should_upload is True
        assert len(result.all_results) == 2

    def test_process_all_reject_wins(self, mock_settings, mock_logger, processing_context):
        """Test that any rejection causes upload to be skipped."""
        registry = PipelineRegistry(mock_settings, mock_logger)
        registry.processors = [MockPassProcessor(), MockRejectProcessor(), MockPassProcessor()]

        result = registry.process_all(processing_context)

        assert result.should_upload is False
        assert result.skip_reason is not None

    def test_process_all_error_handling(self, mock_settings, mock_logger, processing_context):
        """Test that processor errors propagate (triggering retry logic in ProcessingQueue)."""
        registry = PipelineRegistry(mock_settings, mock_logger)
        registry.processors = [MockErrorProcessor(), MockPassProcessor()]

        # Error should propagate and not be caught
        with pytest.raises(RuntimeError, match="Test error"):
            registry.process_all(processing_context)

    def test_aggregated_result_name_prefixing(self, mock_settings, mock_logger, processing_context):
        """Test that extracted data keys are prefixed with processor name."""
        registry = PipelineRegistry(mock_settings, mock_logger)

        # Create two processors with same key name
        class Processor1(MockPassProcessor):
            name = "proc1"
            friendly_name = "Processor 1"
            description = "Test processor 1"

            def process(self, context):
                return ProcessorResult(
                    should_upload=True,
                    extracted_data={"value": 1},
                    confidence=0.9,
                    reason="Test",
                    processing_time_seconds=0.001,
                    processor_name=self.name,
                )

        class Processor2(MockPassProcessor):
            name = "proc2"
            friendly_name = "Processor 2"
            description = "Test processor 2"

            def process(self, context):
                return ProcessorResult(
                    should_upload=True,
                    extracted_data={"value": 2},
                    confidence=0.9,
                    reason="Test",
                    processing_time_seconds=0.001,
                    processor_name=self.name,
                )

        registry.processors = [Processor1(), Processor2()]
        result = registry.process_all(processing_context)

        # Both values should be present with different prefixed keys
        assert result.extracted_data["proc1.value"] == 1
        assert result.extracted_data["proc2.value"] == 2

    def test_timing_measurement(self, mock_settings, mock_logger, processing_context):
        """Test that total processing time is measured."""
        registry = PipelineRegistry(mock_settings, mock_logger)
        registry.processors = [MockPassProcessor()]

        start = time.time()
        result = registry.process_all(processing_context)
        end = time.time()

        assert result.total_time > 0
        assert result.total_time < (end - start) + 0.1  # Allow small margin
