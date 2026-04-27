"""Tests for processor runtime dependency checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from citrasense import startup_checks
from citrasense.startup_checks import _binary_hint, check_processor_runtime_deps


def _settings(
    *,
    processors_enabled: bool = True,
    use_local_apass_catalog: bool = False,
    enabled_processors: dict[str, bool] | None = None,
):
    """Build a minimal settings stub — the only fields the function touches."""
    sensor = SimpleNamespace(
        processors_enabled=processors_enabled,
        enabled_processors=enabled_processors or {},
    )
    return SimpleNamespace(
        sensors=[sensor],
        use_local_apass_catalog=use_local_apass_catalog,
    )


class TestCheckProcessorRuntimeDeps:
    """check_processor_runtime_deps returns banner-shape dicts for missing deps."""

    def test_everything_present_returns_empty(self, monkeypatch):
        monkeypatch.setattr(startup_checks.importlib.util, "find_spec", lambda name: object())
        monkeypatch.setattr(startup_checks.shutil, "which", lambda name: f"/usr/local/bin/{name}")

        issues = check_processor_runtime_deps(_settings(use_local_apass_catalog=True))

        assert issues == []

    def test_returns_empty_when_processors_globally_disabled(self, monkeypatch):
        """The global processors_enabled kill-switch short-circuits everything."""
        monkeypatch.setattr(startup_checks.importlib.util, "find_spec", lambda name: None)
        monkeypatch.setattr(startup_checks.shutil, "which", lambda name: None)

        settings = _settings(processors_enabled=False, use_local_apass_catalog=True)
        issues = check_processor_runtime_deps(settings)

        assert issues == []

    def test_missing_healpix_when_local_apass_enabled(self, monkeypatch):
        monkeypatch.setattr(
            startup_checks.importlib.util,
            "find_spec",
            lambda name: None if name == "astropy_healpix" else object(),
        )
        monkeypatch.setattr(startup_checks.shutil, "which", lambda name: f"/usr/local/bin/{name}")

        issues = check_processor_runtime_deps(_settings(use_local_apass_catalog=True))

        assert len(issues) == 1
        entry = issues[0]
        assert entry["device_type"] == "processor"
        assert entry["device_name"] == "Photometry Calibrator"
        assert entry["missing_packages"] == "astropy_healpix"
        assert "uv tool install --force" in entry["install_cmd"]

    def test_missing_healpix_ignored_when_local_apass_disabled(self, monkeypatch):
        monkeypatch.setattr(startup_checks.importlib.util, "find_spec", lambda name: None)
        monkeypatch.setattr(startup_checks.shutil, "which", lambda name: f"/usr/local/bin/{name}")

        issues = check_processor_runtime_deps(_settings(use_local_apass_catalog=False))

        assert issues == []

    def test_missing_healpix_ignored_when_photometry_disabled(self, monkeypatch):
        monkeypatch.setattr(startup_checks.importlib.util, "find_spec", lambda name: None)
        monkeypatch.setattr(startup_checks.shutil, "which", lambda name: f"/usr/local/bin/{name}")

        settings = _settings(
            use_local_apass_catalog=True,
            enabled_processors={"photometry": False},
        )
        issues = check_processor_runtime_deps(settings)

        assert issues == []

    def test_missing_solve_field_surfaces_plate_solver(self, monkeypatch):
        monkeypatch.setattr(startup_checks.importlib.util, "find_spec", lambda name: object())
        monkeypatch.setattr(
            startup_checks.shutil,
            "which",
            lambda name: None if name == "solve-field" else f"/usr/local/bin/{name}",
        )

        issues = check_processor_runtime_deps(_settings())

        names = [i["device_name"] for i in issues]
        assert "Plate Solver" in names
        plate = next(i for i in issues if i["device_name"] == "Plate Solver")
        assert plate["missing_packages"] == "solve-field"
        assert plate["install_cmd"]  # platform-specific; concrete string is tested elsewhere

    def test_missing_solve_field_ignored_when_plate_solver_disabled(self, monkeypatch):
        monkeypatch.setattr(startup_checks.importlib.util, "find_spec", lambda name: object())
        monkeypatch.setattr(
            startup_checks.shutil,
            "which",
            lambda name: None if name == "solve-field" else f"/usr/local/bin/{name}",
        )

        issues = check_processor_runtime_deps(_settings(enabled_processors={"plate_solver": False}))

        assert issues == []

    def test_source_extractor_any_of_satisfied_by_sex(self, monkeypatch):
        """If `source-extractor` is missing but `sex` is on PATH, we're good."""
        monkeypatch.setattr(startup_checks.importlib.util, "find_spec", lambda name: object())
        monkeypatch.setattr(
            startup_checks.shutil,
            "which",
            lambda name: "/usr/local/bin/sex" if name == "sex" else None,
        )

        issues = check_processor_runtime_deps(_settings())

        names = [i["device_name"] for i in issues]
        # plate_solver will fail because we patched everything but `sex` away,
        # but Source Extractor must NOT be flagged
        assert "Source Extractor" not in names

    def test_source_extractor_both_missing_flags_entry(self, monkeypatch):
        monkeypatch.setattr(startup_checks.importlib.util, "find_spec", lambda name: object())
        monkeypatch.setattr(
            startup_checks.shutil,
            "which",
            lambda name: None if name in {"source-extractor", "sex"} else f"/usr/local/bin/{name}",
        )

        issues = check_processor_runtime_deps(_settings())

        names = [i["device_name"] for i in issues]
        assert "Source Extractor" in names

    def test_source_extractor_ignored_when_disabled(self, monkeypatch):
        monkeypatch.setattr(startup_checks.importlib.util, "find_spec", lambda name: object())
        monkeypatch.setattr(startup_checks.shutil, "which", lambda name: None)

        settings = _settings(
            enabled_processors={
                "plate_solver": False,
                "source_extractor": False,
            },
        )
        issues = check_processor_runtime_deps(settings)

        assert issues == []


class TestBinaryHint:
    """Platform-aware install hints for missing binaries."""

    @pytest.mark.parametrize(
        ("binary", "expected_substring"),
        [
            ("solve-field", "brew install astrometry-net"),
            ("source-extractor", "brew install sextractor"),
        ],
    )
    def test_darwin_returns_brew_command(self, monkeypatch, binary, expected_substring):
        monkeypatch.setattr(startup_checks.sys, "platform", "darwin")
        assert _binary_hint(binary) == expected_substring

    @pytest.mark.parametrize(
        ("binary", "expected_substring"),
        [
            ("solve-field", "sudo apt install astrometry.net"),
            ("source-extractor", "sudo apt install source-extractor"),
        ],
    )
    def test_linux_returns_apt_command(self, monkeypatch, binary, expected_substring):
        monkeypatch.setattr(startup_checks.sys, "platform", "linux")
        assert _binary_hint(binary) == expected_substring

    def test_unknown_platform_returns_generic_fallback(self, monkeypatch):
        monkeypatch.setattr(startup_checks.sys, "platform", "win32")
        hint = _binary_hint("solve-field")
        assert "solve-field" in hint
        assert "PATH" in hint

    def test_unknown_binary_on_known_platform_returns_fallback(self, monkeypatch):
        monkeypatch.setattr(startup_checks.sys, "platform", "darwin")
        hint = _binary_hint("some-novel-binary")
        assert "some-novel-binary" in hint
