"""Unit tests for processing pipeline diagnostic artifact dumping."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from citrascope.processors.artifact_writer import (
    _read_fits_header,
    _task_to_dict,
    dump_context_artifacts,
    dump_csv,
    dump_json,
    dump_processing_summary,
    dump_processor_result,
)
from citrascope.processors.processor_result import AggregatedResult, ProcessingContext, ProcessorResult
from citrascope.tasks.task import Task


@pytest.fixture
def working_dir(tmp_path):
    d = tmp_path / "processing" / "task-abc"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def sample_task():
    return Task(
        id="task-abc-123",
        type="observation",
        status="Scheduled",
        creationEpoch="2025-03-01T00:00:00Z",
        updateEpoch="2025-03-01T00:00:00Z",
        taskStart="2025-03-01T01:00:00Z",
        taskStop="2025-03-01T01:05:00Z",
        userId="user-1",
        username="observer",
        satelliteId="sat-42",
        satelliteName="STARLINK-1234",
        telescopeId="tel-1",
        telescopeName="TestScope",
        groundStationId="gs-1",
        groundStationName="TestStation",
        assigned_filter_name="Clear",
    )


@pytest.fixture
def sample_fits(tmp_path):
    """Create a minimal FITS file with representative header fields."""
    path = tmp_path / "test_image.fits"
    data = np.zeros((100, 100), dtype=np.uint16)
    hdu = fits.PrimaryHDU(data)
    hdu.header["DATE-OBS"] = "2025-03-01T01:02:03.456"
    hdu.header["CRVAL1"] = 180.0
    hdu.header["CRVAL2"] = 45.0
    hdu.header["XBINNING"] = 2
    hdu.header["EXPTIME"] = 5.0
    hdu.header["FILTER"] = "Clear"
    hdu.header["SITELAT"] = 40.0
    hdu.header["SITELONG"] = -105.0
    hdu.header["SITEALT"] = 1600.0
    hdu.writeto(path, overwrite=True)
    return path


class TestDumpJson:
    def test_writes_valid_json(self, working_dir):
        dump_json(working_dir, "test.json", {"key": "value", "num": 42})
        path = working_dir / "test.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data == {"key": "value", "num": 42}

    def test_handles_path_values(self, working_dir):
        dump_json(working_dir, "paths.json", {"p": Path("/some/path")})
        data = json.loads((working_dir / "paths.json").read_text())
        assert data["p"] == "/some/path"

    def test_handles_nan_values(self, working_dir):
        dump_json(working_dir, "nan.json", {"v": float("nan")})
        data = json.loads((working_dir / "nan.json").read_text())
        assert data["v"] is None

    def test_swallows_write_errors(self, tmp_path):
        nonexistent = tmp_path / "no" / "such" / "dir"
        dump_json(nonexistent, "fail.json", {"x": 1})


class TestDumpCsv:
    def test_writes_csv(self, working_dir):
        df = pd.DataFrame({"ra": [1.0, 2.0], "dec": [3.0, 4.0]})
        dump_csv(working_dir, "test.csv", df)
        path = working_dir / "test.csv"
        assert path.exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == 2
        assert list(loaded.columns) == ["ra", "dec"]

    def test_swallows_write_errors(self, tmp_path):
        nonexistent = tmp_path / "no" / "such" / "dir"
        df = pd.DataFrame({"a": [1]})
        dump_csv(nonexistent, "fail.csv", df)


class TestTaskToDict:
    def test_serializes_task_fields(self, sample_task):
        d = _task_to_dict(sample_task)
        assert d["id"] == "task-abc-123"
        assert d["satelliteName"] == "STARLINK-1234"
        assert d["assigned_filter_name"] == "Clear"
        assert "_status_lock" not in d

    def test_none_task(self):
        assert _task_to_dict(None) == {}


class TestReadFitsHeader:
    def test_reads_expected_keys(self, sample_fits):
        header = _read_fits_header(sample_fits)
        assert header["DATE-OBS"] == "2025-03-01T01:02:03.456"
        assert header["CRVAL1"] == 180.0
        assert header["CRVAL2"] == 45.0
        assert header["NAXIS1"] == 100
        assert header["NAXIS2"] == 100
        assert header["XBINNING"] == 2
        assert header["EXPTIME"] == 5.0
        assert header["SITELAT"] == 40.0

    def test_missing_file(self, tmp_path):
        header = _read_fits_header(tmp_path / "nonexistent.fits")
        assert "_error" in header


class TestDumpContextArtifacts:
    def test_writes_all_context_files(self, working_dir, sample_task, sample_fits):
        location_service = Mock()
        location_service.get_current_location.return_value = {
            "latitude": 40.0,
            "longitude": -105.0,
            "altitude": 1600.0,
        }
        elset_cache = Mock()
        elset_cache.get_elsets.return_value = [
            {"satellite_id": "25544", "name": "ISS", "tle": ["1 line", "2 line"]},
        ]

        satellite_data = {
            "id": "sat-99",
            "name": "STARLINK-1234",
            "most_recent_elset": {
                "tle": ["1 25544U ...", "2 25544 ..."],
                "creationEpoch": "2026-03-09T00:00:00Z",
            },
            "elsets": [{"tle": ["1 25544U ...", "2 25544 ..."], "creationEpoch": "2026-03-09T00:00:00Z"}],
        }

        context = ProcessingContext(
            image_path=sample_fits,
            working_image_path=sample_fits,
            working_dir=working_dir,
            image_data=None,
            task=sample_task,
            telescope_record={"focalLength": 500, "pixelSize": 3.8},
            ground_station_record={"id": "gs-1"},
            settings=None,
            location_service=location_service,
            elset_cache=elset_cache,
            satellite_data=satellite_data,
        )

        dump_context_artifacts(context)

        assert (working_dir / "task.json").exists()
        assert (working_dir / "elset_cache_snapshot.json").exists()
        assert (working_dir / "observer_location.json").exists()
        assert (working_dir / "telescope_record.json").exists()
        assert (working_dir / "fits_header.json").exists()
        assert (working_dir / "target_satellite.json").exists()

        task_data = json.loads((working_dir / "task.json").read_text())
        assert task_data["satelliteName"] == "STARLINK-1234"

        elsets = json.loads((working_dir / "elset_cache_snapshot.json").read_text())
        assert len(elsets) == 1
        assert elsets[0]["name"] == "ISS"

        location = json.loads((working_dir / "observer_location.json").read_text())
        assert location["latitude"] == 40.0

        telescope = json.loads((working_dir / "telescope_record.json").read_text())
        assert telescope["focalLength"] == 500

        fits_hdr = json.loads((working_dir / "fits_header.json").read_text())
        assert fits_hdr["CRVAL1"] == 180.0

        target_sat = json.loads((working_dir / "target_satellite.json").read_text())
        assert target_sat["most_recent_elset"]["tle"] == ["1 25544U ...", "2 25544 ..."]

    def test_handles_missing_services(self, working_dir, sample_fits):
        context = ProcessingContext(
            image_path=sample_fits,
            working_image_path=sample_fits,
            working_dir=working_dir,
            image_data=None,
            task=None,
            telescope_record=None,
            ground_station_record=None,
            settings=None,
        )
        dump_context_artifacts(context)

        assert json.loads((working_dir / "task.json").read_text()) == {}
        assert json.loads((working_dir / "elset_cache_snapshot.json").read_text()) == []
        assert json.loads((working_dir / "observer_location.json").read_text()) == {}
        assert json.loads((working_dir / "telescope_record.json").read_text()) == {}
        assert not (working_dir / "target_satellite.json").exists()


class TestDumpProcessorResult:
    def test_writes_result_json(self, working_dir):
        result = ProcessorResult(
            should_upload=True,
            extracted_data={"plate_solved": True, "ra_center": 180.5},
            confidence=1.0,
            reason="Solved in 2.1s",
            processing_time_seconds=2.1,
            processor_name="plate_solver",
        )
        dump_processor_result(working_dir, "plate_solver_result.json", result)

        data = json.loads((working_dir / "plate_solver_result.json").read_text())
        assert data["processor_name"] == "plate_solver"
        assert data["confidence"] == 1.0
        assert data["extracted_data"]["plate_solved"] is True

    def test_includes_extra_fields(self, working_dir):
        result = ProcessorResult(
            should_upload=True,
            extracted_data={"num_sources": 150},
            confidence=1.0,
            reason="Extracted 150 sources",
            processing_time_seconds=3.0,
            processor_name="source_extractor",
        )
        fwhm_stats = {"fwhm_min": 0.8, "fwhm_max": 12.3, "count_fwhm_lt_1_5": 5}
        dump_processor_result(working_dir, "source_extractor_result.json", result, extra=fwhm_stats)

        data = json.loads((working_dir / "source_extractor_result.json").read_text())
        assert data["fwhm_min"] == 0.8
        assert data["count_fwhm_lt_1_5"] == 5
        assert data["extracted_data"]["num_sources"] == 150


class TestDumpProcessingSummary:
    def test_writes_summary(self, working_dir):
        results = [
            ProcessorResult(
                should_upload=True,
                extracted_data={"plate_solved": True},
                confidence=1.0,
                reason="Solved",
                processing_time_seconds=2.0,
                processor_name="plate_solver",
            ),
            ProcessorResult(
                should_upload=True,
                extracted_data={"num_sources": 100},
                confidence=1.0,
                reason="Extracted",
                processing_time_seconds=3.0,
                processor_name="source_extractor",
            ),
        ]
        aggregated = AggregatedResult(
            should_upload=True,
            extracted_data={"plate_solver.plate_solved": True, "source_extractor.num_sources": 100},
            all_results=results,
            total_time=5.0,
            skip_reason=None,
        )
        dump_processing_summary(working_dir, aggregated)

        data = json.loads((working_dir / "processing_summary.json").read_text())
        assert data["should_upload"] is True
        assert data["total_time"] == 5.0
        assert data["skip_reason"] is None
        assert len(data["processors"]) == 2
        assert data["processors"][0]["processor_name"] == "plate_solver"
        assert data["processors"][1]["processor_name"] == "source_extractor"
        assert "plate_solver.plate_solved" in data["extracted_data"]
