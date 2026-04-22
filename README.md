# CitraSense

[![Pytest](https://github.com/citra-space/citrasense/actions/workflows/pytest.yml/badge.svg)](https://github.com/citra-space/citrasense/actions/workflows/pytest.yml) [![Publish Python Package](https://github.com/citra-space/citrasense/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/citra-space/citrasense/actions/workflows/pypi-publish.yml) [![PyPI version](https://badge.fury.io/py/citrasense.svg)](https://pypi.org/project/citrasense/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/citrasense)](https://pypi.org/project/citrasense/) [![License](https://img.shields.io/github/license/citra-space/citrasense)](https://github.com/citra-space/citrasense/blob/main/LICENSE)

**[Documentation](https://docs.citra.space/citrasense/)** | **[Citra.space](https://citra.space)** | **[PyPI](https://pypi.org/project/citrasense/)** | **[CitraSense-Pi](https://github.com/citra-space/citrasense-pi)**

Photograph satellites from your backyard telescope.

CitraSense turns your astrophotography rig into an autonomous satellite observation station. It connects to [Citra.space](https://citra.space), picks up tasks for specific satellites, slews your telescope, captures images, and processes everything on the edge — plate solving, source extraction, photometry, and satellite matching — before delivering calibrated results back to the platform.

![CitraSense Web UI](citrasense_screenshot.png)

Use [Citra.space](https://citra.space) to search for satellites, schedule observation windows, and task your telescope — CitraSense handles the rest. If you already have a mount, a camera, and software like [N.I.N.A.](https://nighttime-imaging.eu/) or [KStars/Ekos](https://kstars.kde.org/), CitraSense plugs right in. Install it, select your hardware adapter, and you're one click away from photographing the ISS.

## Supported Hardware

| Adapter | Platform | Connects to |
|---------|----------|-------------|
| **Direct** | Any | Composable device adapters (USB cameras, Ximea, Raspberry Pi Camera) |
| **NINA** | Windows | [N.I.N.A.](https://nighttime-imaging.eu/) via Advanced HTTP API |
| **KStars** | Linux, macOS | [KStars/Ekos](https://kstars.kde.org/) via D-Bus |
| **INDI** | Linux | Any [INDI](https://indilib.org/)-compatible device |

## Quick Start

**Requires Python 3.10+.** We recommend [uv](https://docs.astral.sh/uv/) for installation.

```sh
# Install and run
uv tool install citrasense
citrasense
```

Or with pip:

```sh
pip install citrasense
citrasense
```

Open `http://localhost:24872` in your browser to configure your hardware adapter, connect to Citra.space, and start accepting tasks.

**Raspberry Pi?** Check out [citrasense-pi](https://github.com/citra-space/citrasense-pi) for a ready-to-flash SD card image with CitraSense pre-installed.

### Optional Extras

```sh
uv tool install citrasense --with citrasense[indi]        # INDI protocol support
uv tool install citrasense --with citrasense[kstars]      # KStars/Ekos via D-Bus
uv tool install citrasense --with citrasense[usb-camera]  # USB cameras via OpenCV
uv tool install citrasense --with citrasense[rpi]         # Raspberry Pi Camera Module
uv tool install citrasense --with citrasense[all]         # INDI + KStars
```

<details>
<summary>pip equivalents</summary>

```sh
pip install citrasense[indi]
pip install citrasense[kstars]
pip install citrasense[usb-camera]
pip install citrasense[rpi]
pip install citrasense[all]
```
</details>

### CLI Options

```sh
citrasense --help
citrasense --web-port 8080    # Custom web UI port (default: 24872)
```

## Documentation

Full documentation is available at [docs.citra.space/citrasense](https://docs.citra.space/citrasense/). Documentation source is maintained in the [citra-space/docs](https://github.com/citra-space/docs) repository.

## Developer Setup

```sh
uv sync --extra dev
uv run pre-commit install
```

### Dev Container (INDI on macOS/Windows)

If you need to work with the INDI adapter on a non-Linux host, the project includes a [VS Code Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) that provides a Linux environment with `pyindi-client` and its system dependencies pre-installed. Open the project in VS Code and choose **Reopen in Container**.

### Running Tests

```sh
uv run pytest                          # Unit tests (fast, skips slow tests)
uv run pytest -m "not integration"     # Same as above, explicit
uv run pytest --override-ini="addopts=" -m "not integration"  # Include slow tests locally
```

Tests use `pytest` markers to separate fast unit tests from expensive ones:

| Marker | What it covers | Runs locally | Runs in CI |
|--------|---------------|:------------:|:----------:|
| *(none)* | Unit tests | Yes | Yes |
| `@pytest.mark.slow` | Real FITS processing, plate solving | No | Yes |
| `@pytest.mark.integration` | Live hardware or services | No | Opt-in |

### Pre-commit Hooks

The project uses [pre-commit](https://pre-commit.com/) with **Ruff** (linting + import sorting), **Black** (formatting), and **Pyright** (type checking).

```sh
uv run pre-commit run --all-files    # Run all checks manually
```

### VS Code Launch Configs

The `.vscode/launch.json` includes pre-configured debug configurations:
- **Python: citrasense** — Runs the daemon with default settings
- **Python: citrasense (custom port)** — Runs with web interface on port 8080

### Releasing

```sh
bump-my-version bump patch    # 0.1.3 → 0.1.4
bump-my-version bump minor    # 0.1.3 → 0.2.0
bump-my-version bump major    # 0.1.3 → 1.0.0
git push && git push --tags
```

Tagging triggers `create-release.yml` which creates a GitHub release, which in turn triggers `pypi-publish.yml`.

## License

[MIT](LICENSE)
