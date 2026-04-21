"""Unit tests for citrasense.version — install type detection and git metadata."""

from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

from citrasense.version import (
    VersionInfo,
    format_version_cli,
    format_version_log,
    get_version_info,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_run(branch: str = "main", sha: str = "abc1234", dirty: str = ""):
    """Return a side_effect for subprocess.run that mimics git commands."""

    def _side_effect(cmd, **_kwargs):
        args = cmd[1:]  # strip "git"
        if args == ["rev-parse", "--short", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{sha}\n", stderr="")
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{branch}\n", stderr="")
        if args == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=dirty, stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error")

    return _side_effect


# ---------------------------------------------------------------------------
# get_version_info
# ---------------------------------------------------------------------------


class TestGetVersionInfo:
    """Tests for the main get_version_info() function."""

    def test_pypi_install(self, tmp_path):
        """No git repo detected -> install_type 'pypi'."""
        with (
            patch("citrasense.version.pkg_version", return_value="1.2.3"),
            patch("citrasense.version._find_git_root", return_value=None),
        ):
            info = get_version_info()
        assert info["version"] == "1.2.3"
        assert info["install_type"] == "pypi"
        assert info["git_hash"] is None
        assert info["git_branch"] is None
        assert info["git_dirty"] is False

    def test_editable_install_clean(self, tmp_path):
        """Metadata found + git repo -> install_type 'editable'."""
        with (
            patch("citrasense.version.pkg_version", return_value="0.9.10"),
            patch("citrasense.version._find_git_root", return_value=tmp_path),
            patch("subprocess.run", side_effect=_fake_run("main", "deadbeef", "")),
        ):
            info = get_version_info()
        assert info["version"] == "0.9.10"
        assert info["install_type"] == "editable"
        assert info["git_hash"] == "deadbeef"
        assert info["git_branch"] == "main"
        assert info["git_dirty"] is False

    def test_editable_install_dirty(self, tmp_path):
        """Dirty working tree is detected."""
        with (
            patch("citrasense.version.pkg_version", return_value="0.9.10"),
            patch("citrasense.version._find_git_root", return_value=tmp_path),
            patch("subprocess.run", side_effect=_fake_run("dev", "1234567", " M foo.py\n")),
        ):
            info = get_version_info()
        assert info["git_dirty"] is True
        assert info["git_branch"] == "dev"

    def test_source_install_no_metadata(self, tmp_path):
        """No pip metadata at all -> install_type 'source'."""
        with (
            patch("citrasense.version.pkg_version", side_effect=PackageNotFoundError("citrasense")),
            patch("citrasense.version._find_git_root", return_value=tmp_path),
            patch("subprocess.run", side_effect=_fake_run()),
        ):
            info = get_version_info()
        assert info["version"] == "development"
        assert info["install_type"] == "source"
        assert info["git_hash"] == "abc1234"

    def test_git_not_installed(self, tmp_path):
        """git binary missing -> falls back gracefully."""

        def _no_git(cmd, **_kwargs):
            raise FileNotFoundError("git not found")

        with (
            patch("citrasense.version.pkg_version", return_value="1.0.0"),
            patch("citrasense.version._find_git_root", return_value=tmp_path),
            patch("subprocess.run", side_effect=_no_git),
        ):
            info = get_version_info()
        assert info["install_type"] == "pypi"
        assert info["git_hash"] is None

    def test_git_timeout(self, tmp_path):
        """git commands timing out -> falls back gracefully."""

        def _timeout(cmd, **_kwargs):
            raise subprocess.TimeoutExpired(cmd, 5)

        with (
            patch("citrasense.version.pkg_version", return_value="1.0.0"),
            patch("citrasense.version._find_git_root", return_value=tmp_path),
            patch("subprocess.run", side_effect=_timeout),
        ):
            info = get_version_info()
        assert info["install_type"] == "pypi"
        assert info["git_hash"] is None


# ---------------------------------------------------------------------------
# format_version_log
# ---------------------------------------------------------------------------


class TestFormatVersionLog:
    def test_pypi(self):
        info = VersionInfo(version="1.2.3", install_type="pypi", git_hash=None, git_branch=None, git_dirty=False)
        assert format_version_log(info) == "1.2.3"

    def test_git_clean(self):
        info = VersionInfo(
            version="0.9.10", install_type="editable", git_hash="abc1234", git_branch="main", git_dirty=False
        )
        assert format_version_log(info) == "0.9.10 (main@abc1234)"

    def test_git_dirty(self):
        info = VersionInfo(
            version="0.9.10", install_type="editable", git_hash="abc1234", git_branch="dev", git_dirty=True
        )
        assert format_version_log(info) == "0.9.10 (dev@abc1234, dirty)"


# ---------------------------------------------------------------------------
# format_version_cli
# ---------------------------------------------------------------------------


class TestFormatVersionCli:
    def test_pypi(self):
        info = VersionInfo(version="1.2.3", install_type="pypi", git_hash=None, git_branch=None, git_dirty=False)
        assert format_version_cli(info) == "1.2.3"

    def test_git_clean(self):
        info = VersionInfo(
            version="0.9.10", install_type="editable", git_hash="abc1234", git_branch="main", git_dirty=False
        )
        assert format_version_cli(info) == "0.9.10 (git: main@abc1234)"

    def test_git_dirty(self):
        info = VersionInfo(
            version="0.9.10", install_type="editable", git_hash="abc1234", git_branch="dev", git_dirty=True
        )
        assert format_version_cli(info) == "0.9.10 (git: dev@abc1234, dirty)"
