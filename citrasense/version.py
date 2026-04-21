"""Version detection for CitraSense, handling both PyPI and git-based installs."""

from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TypedDict


class VersionInfo(TypedDict):
    version: str
    install_type: str  # "pypi", "editable", or "source"
    git_hash: str | None
    git_branch: str | None
    git_dirty: bool


def _find_git_root() -> Path | None:
    """Walk up from the package directory to find a .git directory (or file, for worktrees)."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _run_git(args: list[str], cwd: Path) -> str | None:
    """Run a git command, returning stripped stdout or None on any failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def get_version_info() -> VersionInfo:
    """Detect CitraSense version, install type, and git state.

    Returns a dict with version string, install classification, and git
    metadata (hash, branch, dirty flag) when running from a git checkout.
    """
    try:
        base_version = pkg_version("citrasense")
    except PackageNotFoundError:
        base_version = None

    git_root = _find_git_root()
    git_hash: str | None = None
    git_branch: str | None = None
    git_dirty = False

    if git_root is not None:
        git_hash = _run_git(["rev-parse", "--short", "HEAD"], git_root)
        git_branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], git_root)
        dirty_check = _run_git(["status", "--porcelain"], git_root)
        if dirty_check is not None:
            git_dirty = len(dirty_check) > 0

    if base_version is None:
        install_type = "source"
        version = "development"
    elif git_hash is not None:
        install_type = "editable"
        version = base_version
    else:
        install_type = "pypi"
        version = base_version

    return VersionInfo(
        version=version,
        install_type=install_type,
        git_hash=git_hash,
        git_branch=git_branch,
        git_dirty=git_dirty,
    )


def format_version_log(info: VersionInfo) -> str:
    """Format version info for the daemon startup log line."""
    ver = info["version"]
    if info["git_hash"]:
        branch = info["git_branch"] or "unknown"
        suffix = f" ({branch}@{info['git_hash']}"
        if info["git_dirty"]:
            suffix += ", dirty"
        suffix += ")"
        return f"{ver}{suffix}"
    return ver


def format_version_cli(info: VersionInfo) -> str:
    """Format version info for CLI --version output."""
    ver = info["version"]
    if info["git_hash"]:
        branch = info["git_branch"] or "unknown"
        git_part = f"git: {branch}@{info['git_hash']}"
        if info["git_dirty"]:
            git_part += ", dirty"
        return f"{ver} ({git_part})"
    return ver
