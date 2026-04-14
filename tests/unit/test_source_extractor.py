"""Unit tests for SourceExtractorProcessor CLI override flags."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def processor():
    from citrascope.processors.builtin.source_extractor_processor import SourceExtractorProcessor

    return SourceExtractorProcessor()


@pytest.fixture
def config_dir():
    """Real config directory with bundled SExtractor configs."""
    return Path(__file__).resolve().parents[2] / "citrascope" / "processors" / "builtin" / "sextractor_configs"


@pytest.fixture
def working_dir(tmp_path):
    return tmp_path / "work"


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "test.fits"
    p.touch()
    return p


def _run_extract(processor, image_path, config_dir, working_dir, **kwargs):
    """Call _extract_sources with subprocess.run mocked, return the command list."""
    working_dir.mkdir(parents=True, exist_ok=True)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    captured_cmds: list[list[str]] = []

    def fake_run(cmd, **_kw):
        captured_cmds.append(list(cmd))
        catalog = working_dir / "output.cat"
        catalog.write_text(
            "# 1 ALPHA_J2000\n# 2 DELTA_J2000\n# 3 MAG_AUTO\n# 4 MAGERR_AUTO\n"
            "# 5 FWHM_IMAGE\n# 6 ELONGATION\n"
            "  10.0  20.0  15.0  0.1  3.0  1.1\n"
        )
        return mock_result

    with patch("citrascope.processors.builtin.source_extractor_processor.subprocess.run", side_effect=fake_run):
        processor._extract_sources(
            image_path,
            config_dir,
            working_dir,
            logger=MagicMock(),
            **kwargs,
        )

    assert captured_cmds, "subprocess.run was never called"
    return captured_cmds[0]


class TestSExtractorCLIOverrides:
    def test_no_overrides_uses_defaults(self, processor, image_path, config_dir, working_dir):
        cmd = _run_extract(processor, image_path, config_dir, working_dir)
        assert cmd[0] == "sex"
        assert "-DETECT_THRESH" not in cmd
        assert "-ANALYSIS_THRESH" not in cmd
        assert "-DETECT_MINAREA" not in cmd
        assert "-FILTER_NAME" not in cmd

    def test_detect_thresh_override(self, processor, image_path, config_dir, working_dir):
        cmd = _run_extract(processor, image_path, config_dir, working_dir, detect_thresh=3.0)
        idx = cmd.index("-DETECT_THRESH")
        assert cmd[idx + 1] == "3.0"
        idx2 = cmd.index("-ANALYSIS_THRESH")
        assert cmd[idx2 + 1] == "3.0"

    def test_detect_minarea_override(self, processor, image_path, config_dir, working_dir):
        cmd = _run_extract(processor, image_path, config_dir, working_dir, detect_minarea=5)
        idx = cmd.index("-DETECT_MINAREA")
        assert cmd[idx + 1] == "5"

    def test_filter_name_override(self, processor, image_path, config_dir, working_dir):
        cmd = _run_extract(processor, image_path, config_dir, working_dir, filter_name="gauss_1.5_3x3")
        idx = cmd.index("-FILTER_NAME")
        assert cmd[idx + 1] == "gauss_1.5_3x3.conv"

    def test_filter_name_default_not_appended(self, processor, image_path, config_dir, working_dir):
        cmd = _run_extract(processor, image_path, config_dir, working_dir, filter_name="default")
        assert "-FILTER_NAME" not in cmd

    def test_filter_name_none_not_appended(self, processor, image_path, config_dir, working_dir):
        cmd = _run_extract(processor, image_path, config_dir, working_dir, filter_name=None)
        assert "-FILTER_NAME" not in cmd

    def test_unknown_kernel_falls_back(self, processor, image_path, config_dir, working_dir):
        cmd = _run_extract(processor, image_path, config_dir, working_dir, filter_name="nonexistent_kernel")
        assert "-FILTER_NAME" not in cmd

    def test_all_overrides_combined(self, processor, image_path, config_dir, working_dir):
        cmd = _run_extract(
            processor,
            image_path,
            config_dir,
            working_dir,
            detect_thresh=2.0,
            detect_minarea=10,
            filter_name="tophat_3.0_3x3",
        )
        assert "-DETECT_THRESH" in cmd
        assert "-DETECT_MINAREA" in cmd
        assert "-FILTER_NAME" in cmd
        assert cmd[cmd.index("-DETECT_THRESH") + 1] == "2.0"
        assert cmd[cmd.index("-DETECT_MINAREA") + 1] == "10"
        assert cmd[cmd.index("-FILTER_NAME") + 1] == "tophat_3.0_3x3.conv"
