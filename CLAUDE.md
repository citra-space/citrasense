# CLAUDE.md — Agent Guide for CitraSense

## What is this project?

CitraSense is a Python daemon that runs at a telescope. It polls a backend API for observation tasks, controls hardware (mount, camera, filter wheel, focuser) to capture images, runs an edge-processing pipeline (plate solving, source extraction, photometry, satellite matching), and uploads results. It also serves a local web UI for operators to monitor and configure the system.

**Backend API**: https://dev.api.citra.space/docs (Swagger)

## Quick start

```bash
uv sync --extra dev --extra test
uv run pytest -m "not integration"
uv run citrasense  # starts the daemon
```

Web UI runs on port **24872** (CITRA on a phone keypad).

Config lives at `~/Library/Application Support/citrasense/config.json` (macOS) or platform equivalent via `platformdirs`.

### Useful paths for debugging

The running daemon exposes its file paths via `GET /api/config` (`config_file_path`, `log_file_path`). Typical macOS locations:

- **Local API**: `http://localhost:24872/api/` (try `/api/config`, `/api/status`)
- **Log file**: `~/Library/Logs/citrasense/citrasense-YYYY-MM-DD.log`
- **Config file**: `~/Library/Application Support/citrasense/config.json`

## Architecture overview

```
__main__.py (CLI via Click)
  └─ citrasense_daemon.py (lifecycle, main loop)
       ├─ api/ (CitraApiClient, DummyApiClient for local testing)
       ├─ hardware/ (adapter pattern — abstract base + concrete implementations)
       │    ├─ abstract_astro_hardware_adapter.py (the contract)
       │    ├─ adapter_registry.py (add new adapters here)
       │    ├─ filter_sync.py, dummy_adapter.py (shared utilities)
       │    ├─ nina/ (adapter, event_listener, survey_template)
       │    ├─ kstars/ (adapter, scheduler_template, sequence_template)
       │    ├─ indi/ (adapter)
       │    ├─ direct/ (adapter)
       │    └─ devices/ (direct USB/RPi/Ximea cameras)
       ├─ sensors/
       │    ├─ abstract_sensor.py (AbstractSensor, SensorCapabilities, acquisition modes)
       │    ├─ sensor_manager.py (manages multiple AbstractSensor instances)
       │    ├─ sensor_runtime.py (SensorRuntime — per-sensor execution silo: owns queues + managers)
       │    └─ telescope_sensor.py (adapter bridge to AbstractAstroHardwareAdapter)
       ├─ acquisition/
       │    ├─ base_work_queue.py (BaseWorkQueue — retry, backoff, threading)
       │    ├─ acquisition_queue.py (AcquisitionQueue — imaging/capture stage)
       │    ├─ processing_queue.py (ProcessingQueue — edge-processing stage)
       │    └─ upload_queue.py (UploadQueue — result upload stage)
       ├─ tasks/
       │    ├─ task_dispatcher.py (TaskDispatcher — site-level: API polling, heap, routing, safety, web facade)
       │    ├─ autofocus_manager.py (dedicated autofocus scheduling/execution)
       │    ├─ scope/ (base_telescope_task.py, sidereal/tracking variants)
       │    └─ views/ (TelescopeTaskView, RfTaskView, RadarTaskView — per-modality typed accessors)
       ├─ pipelines/
       │    ├─ common/ (abstract_processor, pipeline_registry, processing_context, processor_result, artifact_writer)
       │    ├─ optical/ (optical_processing_context, calibration → plate_solver → source_extractor → photometry → satellite_matcher → annotated_image, optical_artifacts, report_generator)
       │    ├─ radar/ (future — #307)
       │    └─ rf/ (future)
       ├─ settings/ (CitraSenseSettings — Pydantic BaseModel, persisted to JSON)
       └─ web/ (FastAPI app, Alpine.js frontend, WebSocket log streaming)
```

## Critical conventions

### RA/Dec are always in degrees internally

Every function, API call, and data structure uses **degrees** for both RA and Dec. Conversions to/from hours happen only at hardware boundaries:
- **INDI**: expects RA in hours → convert `ra / 15.0` before sending, `ra * 15.0` when reading
- **KStars D-Bus**: same as INDI
- **NINA Advanced API**: accepts degrees natively, no conversion needed
- **Citra API**: degrees
- **FITS WCS, Astropy, Skyfield**: degrees

If you touch any coordinate code, verify the units. This has been a source of real bugs.

### Hardware adapters

All adapters implement `AbstractAstroHardwareAdapter`. Key methods:
- `connect()` / `disconnect()`
- `do_autofocus(target_ra, target_dec, on_progress)` — all in degrees
- `perform_observation_sequence(task, satellite_data)` — returns file paths
- `point_telescope(ra, dec)` — degrees

To add a new adapter: create the class in `citrasense/hardware/`, register it in `adapter_registry.py`.

### Hardware device probes (GIL-safe detection)

Any device that enumerates native hardware (ctypes/SDK calls) in `get_settings_schema()` **must** use `AbstractHardwareDevice._cached_hardware_probe()`. It runs the probe in a subprocess (separate GIL) with a timeout, so a hung USB device can't freeze the web server. See the docstring on that method for the full contract and `MoravianCamera`/`ZwoEafFocuser`/`UsbCamera` for working examples.

The NINA adapter uses a **WebSocket event listener** (`nina/event_listener.py`) for reactive hardware control instead of polling. The pattern is: `event.clear()` → issue command via REST → `event.wait(timeout=...)`. Key events: `SEQUENCE-FINISHED`, `IMAGE-SAVE`, `FILTERWHEEL-CHANGED`, `AUTOFOCUS-FINISHED`.

### Work queues

Tasks flow through three serial queues owned by a **SensorRuntime**: **AcquisitionQueue** → **ProcessingQueue** → **UploadQueue**. Each has background worker threads. Use `queue.is_idle()` to check if a queue has pending work. The **TaskDispatcher** sits above runtimes handling API polling, task scheduling (heap), routing tasks to the right runtime, safety evaluation, and stage tracking. It exposes web-compat facade properties (`imaging_queue`, `autofocus_manager`, etc.) that delegate to the default runtime so existing web routes work unchanged.

### Web UI

- **Dark theme required** — operators use this at night, preserve night vision
- **Mobile-friendly** — the UI is used on tablets/phones in the field
- **FastAPI** backend in `web/app.py`, **Alpine.js** frontend in `web/static/`
- Real-time updates via WebSocket at `/ws` (`WebLogHandler` broadcasts logs to connected clients)
- Templates in `web/templates/` (Jinja2 partials prefixed with `_`)
- The web server runs in a **daemon thread** with its own event loop (`web/server.py`). Use thread-safe mechanisms when accessing daemon state from web handlers. The daemon only calls `web_server.start()` — all web complexity stays in `web/server.py`.
- **Toast notifications**: Use `web_server.send_toast(message, type, id)` from daemon threads to push Bootstrap toasts to the browser. `type` is `"success"`, `"info"`, `"warning"`, or `"danger"`. `danger`/`warning` toasts persist until dismissed; `success`/`info` auto-hide. Pass an `id` to deduplicate (only one toast with that id shown at a time). Wire via an `on_toast` callback attribute — see `TaskManager.on_toast` and `AutofocusManager.on_toast` for the pattern. **Use toasts for safety-critical events** so operators who return later see what happened.

### Adding a new setting

`CitraSenseSettings` is a **Pydantic BaseModel**. Each persisted setting is a single field declaration. The API endpoint (`GET /api/config`) and JS save function (`saveConfiguration()`) derive from `to_dict()` / `store.config` automatically — no manual field lists to maintain.

1. **`settings/citrasense_settings.py`** — add a Pydantic field with type and default:
   ```python
   my_setting: str = "default_value"
   ```
   If the field needs validation, add a `@field_validator` with warn-and-fallback semantics (see existing validators for the pattern).

2. **(If UI-editable)** Add the input/control to the relevant HTML template in `web/templates/`.

That's it. `to_dict()` (via `model_dump()`), `GET /api/config`, `saveConfiguration()`, and `update_and_save()` all pick up the new field automatically.

### Adapter settings persistence

`_all_adapter_settings` in `CitraSenseSettings` is a nested dict keyed by adapter name (e.g., `"NinaAdvancedHttpAdapter"`). Each entry contains adapter-specific settings *including* a `"filters"` key saved by `_save_filter_config()`. The web form sends flat adapter_settings for only the **current** adapter — `update_and_save()` must **merge** incoming settings with existing ones, not replace the entire dict. Replacing it will silently wipe filters and other adapter-specific state.

### Task lifecycle and race conditions

After imaging completes, a task is in-flight through processing and upload queues. The next poll cycle can see it's not the `current_task_id` and not yet removed from the API response — re-adding it causes duplicate imaging and 404 errors on the second upload attempt. The `_stage_lock` vs `heap_lock` ordering matters: `remove_task_from_all_stages` acquires `_stage_lock` then `heap_lock`, so any code that needs both must follow the same order to avoid deadlocks.

### Module boundaries and encapsulation

**We have shipped violations of every rule below. These are not aspirational — they are load-bearing guardrails.**

#### Pass narrow dependencies, not god objects

A class or function should receive only the specific objects it actually uses — not a large parent object it can reach into. If you're writing `self.thing.foo.bar`, the dependency is too wide. Pass `foo` directly, or define a small Protocol/dataclass that bundles just the fields the consumer needs.

#### Use abstractions — never bypass them

If a module defines an abstract interface, all external consumers must go through it. Don't use `getattr()` to fish out a concrete implementation's internal objects and call methods on them directly — that silently breaks every other implementation of the abstraction. If the interface doesn't expose what you need, **extend the interface**.

#### Respect private fields (`_prefix`)

Never read or write `_private` attributes from outside the owning class. If external code needs the information:

1. Add a public method or property to the class that owns the state.
2. Put the decision logic where the data lives — if a check depends on private state, the check belongs on the class that holds it, not on the caller.

#### Don't reach into another object's locks or internal data structures

If you need a snapshot of another object's state, call a **public query method** that handles its own locking and returns a copy. Never acquire another object's locks from outside, iterate its internal collections directly, or depend on the shape of its private data structures. Lock ordering and data layout must stay encapsulated in the owning class.

#### Dependency direction flows downward

Lower-level modules must not import from higher-level ones. If a lower layer needs data that originates in a higher one, define a Protocol or lightweight data class in the lower layer (or a shared module) instead of importing the higher-level type. When you find yourself adding an import that points upward, stop and introduce a narrow interface.

## Coding standards

- **Python 3.10+**, line length **120** (Black)
- **Type hints** on public methods. Use `Type | None` for nullable attrs.
- Use `from __future__ import annotations` + `if TYPE_CHECKING:` to avoid circular imports with type hints.
- Use `assert self.thing is not None` to satisfy type checkers when an attribute is guaranteed set after `connect()`.
- **Logging**: use the project logger (`self.logger`), never `print()`.
- **Tests**: `pytest`, files named `test_*.py` in `tests/unit/` or `tests/integration/`.
- Run tests: `uv run pytest` (slow tests auto-skipped locally via `addopts`; CI runs everything)
- Run slow tests too: `uv run pytest -m ""` or `uv run pytest --override-ini="addopts="`

### Static analysis — keep it green

The full stack runs at three layers. New code must pass all three cleanly.

| Layer | Formatting | Linting | Type Checking |
|---|---|---|---|
| **Editor (save)** | Black | Ruff | Pyright (Pylance) |
| **Pre-commit** | Black | Ruff | Pyright |
| **CI** | — | Ruff | Pyright |

**Ruff** handles linting and import sorting. Auto-fixes run on save (`.vscode/settings.json`) and at pre-commit (`ruff --fix`). Config is in `pyproject.toml` under `[tool.ruff]`.

**Black** handles formatting. Runs on save and at pre-commit. Config is in `pyproject.toml` under `[tool.black]`.

**Pyright** (basic mode) handles type checking. Config is in `pyrightconfig.json`. The same engine powers Pylance in the editor, so red squiggles in Cursor match what CI enforces.

> **Agent note:** Cursor's inline lint panel surfaces Pyright diagnostics but **does not run the full ruff ruleset** the project opts into (`select = ["E", "F", "W", "I", "UP", "PT", "B", "T20", "C4", "RUF"]`). That means editor-clean code can still fail pre-commit on things like `PT011` (`pytest.raises` too broad), `B` (bugbear), `T20` (stray `print`), or `C4` (comprehension style). After editing Python, run `uv run ruff check <files>` before handing back — don't rely on the editor's lint panel alone.

Common type-checking patterns used in this codebase:

- `assert self.x is not None` — narrow `T | None` attrs that are guaranteed set after initialization (e.g., after `connect()`, after `_initialize_components()`). Preferred over `if` guards when the None case is a programming error.
- `# type: ignore[reportMissingImports]` — only for platform-specific SDK imports (`PyIndi`, `dbus`, `cv2`, `ximea`) that aren't installed everywhere.
- `# type: ignore[attr-defined]` / `# type: ignore[arg-type]` — only for third-party library stub gaps (astroquery, astropy.units, FITS header values). Never use these to silence errors in our own code.
- `assert isinstance(hdul[0], fits.PrimaryHDU)` — narrow opaque `HDUList.__getitem__` returns for FITS processing.
- `from __future__ import annotations` + `if TYPE_CHECKING:` — for circular import avoidance only.

## Testing philosophy

### Markers

| Marker | Meaning | Runs locally | Runs in CI |
|---|---|---|---|
| *(none)* | Normal unit test | Yes | Yes |
| `@pytest.mark.slow` | Expensive tests (plate solving, real FITS processing) | **No** (skipped by default) | Yes |
| `@pytest.mark.integration` | Requires live hardware or services | No | No (opt-in) |

Slow tests are skipped locally via `addopts = ["-m", "not slow"]` in `pyproject.toml`. CI overrides this with `--override-ini="addopts="` to run the full suite.

### What makes a good test here

**Write tests that validate real logic** — math, parsing, state machines, file I/O, retry behavior. These catch real bugs.

**Don't write mock theater.** If a test assembles 6 MagicMocks, calls one method, and asserts that mock #3 was called with the right argument — that's testing your mocks, not your code. Orchestration code (daemon init, task runner loops) is better served by integration tests.

Good test targets in this codebase:
- Pure functions (`filter_sync.py`, `elset_cache._normalize_api_response`)
- Math/science (`angular_distance`, time health thresholds, GPS parsing)
- Stateful logic with real side effects (`BaseWorkQueue` retry/backoff, settings file round-tripping)
- API contract validation (FastAPI `TestClient` route tests — response shapes and status codes)
- Pipeline processing (real FITS through plate solver → source extractor → photometry, marked `@pytest.mark.slow`)

Bad test targets:
- Testing test doubles (DummyApiClient is itself a fake)
- Daemon/runner lifecycle with fully mocked dependencies
- Trivial assertions like "abstract method raises NotImplementedError"

### Test-to-source ratio

Aim for tests that are roughly 1:1 or shorter than the code they test. If you need 300 lines of test setup to exercise 50 lines of source, the test is probably mock wiring — consider an integration test instead.

## Common pitfalls

- **DummyApiClient** returns snake_case keys (matching the real Citra API), not camelCase. The NINA API uses camelCase — don't confuse the two.
- **Task.from_dict** parses camelCase keys from the Citra API (e.g., `assignedFilterName`).
- **NINA WebSocket event ordering**: `SEQUENCE-FINISHED` arrives before `IMAGE-SAVE`. Don't query image history until IMAGE-SAVE confirms the file is written.
- **NINA's IMAGE-SAVE payload** does not include an `Index` field despite what the wiki says. Match by `Filename`, then look up the index via `/image-history`.
- **Focuser move** has no WebSocket event — still requires polling.
- **AutofocusManager** gates on `imaging_queue.is_idle()` to prevent hardware collisions with active imaging tasks.
- **`formatLastAutofocus`** (JS) has a sanity check for timestamps before 2020-01-01 — returns "Never" for obviously wrong values.
- **NINA WebSocket is always available** if the REST API is. There's no configuration to disable it separately. Don't add HTTP polling fallbacks — they add complexity for a case that can't happen.
- **NINA autofocus can run indefinitely** without a timeout. Always enforce a deadline when waiting for autofocus results.
- **Alpine.js `x-show` vs HTML5 `required`**: `x-show` keeps hidden elements in the DOM, but `required` attributes still fire validation on them. Use `:required="visible"` or `x-if`.
- **Alpine store load ordering**: If a `<select>` depends on API-fetched options (e.g., autofocus presets), fetch those *before* setting the config that references them, or the UI renders with stale/empty options.
- **After renaming a parameter**, search the entire method/file for the old name. Long methods have had stale references survive "first pass" refactors, causing `NameError`s at runtime.
- **Don't invent redundant settings.** Before adding one, check if an existing setting already controls the same behavior.
- **Pre-commit hooks**: The repo has a 500KB file size limit. FITS files and large binaries will be rejected.
- **ZWO AM5 mount protocol** has many gotchas — see the docstring in `citrasense/hardware/devices/mount/zwo_am_protocol.py` for the full reference (supported commands, broken commands, error codes, and authoritative sources).
- **Encapsulation shortcuts compound.** Before passing a large object, bypassing an abstraction, reading a private field, grabbing another object's lock, or adding an upward import — re-read *Module boundaries and encapsulation* above. Every one of these has caused real bugs here.

## Key dependencies

- **Click**: CLI entry point (`python -m citrasense`)
- **FastAPI** + **Uvicorn**: Web UI and API, runs in a daemon thread
- **Requests**: HTTP calls to Citra API and NINA REST API
- **websockets**: NINA WebSocket event listener
- **Skyfield**: Astronomical calculations (celestial positions, satellite passes)
- **Astropy**: FITS file I/O, WCS, coordinate transforms
- **SEP**: Source Extraction and Photometry (used for HFR computation in autofocus)
- **Alpine.js**: Reactive frontend (loaded via CDN, no build step)

### System dependencies (not pip-installable)

- **astrometry.net** (`solve-field`): Plate solving. Install: `brew install astrometry-net` (macOS) / `apt install astrometry.net` (Linux). Requires [index files](https://data.astrometry.net/) (4100-series for wide FOV).
- **SExtractor** (`sex` / `source-extractor`): Source extraction. Install: `brew install sextractor` (macOS) / `apt install sextractor` (Linux).

**uv** is the project's Python toolchain — it manages the virtualenv, lockfile, and dependency resolution. Dev tools: **ruff** (linter + import sorting), **Black** (formatter), **pyright** (type checking, basic mode), **pytest** + **pytest-cov** (testing). Pre-commit hooks run ruff, black, and pyright automatically. CI runs ruff and pyright as separate jobs.

Full dependency list in `pyproject.toml`, pinned versions in `uv.lock`.

## External APIs

- **Citra API**: `https://dev.api.citra.space/docs` — task polling, filter sync, observation upload
- **NINA Advanced API**: REST at `http://<host>:1888/v2/api`, WebSocket at `ws://<host>:1888/v2/socket` — [docs](https://github.com/christian-photo/ninaAPI/wiki/Websocket-V2)
- **INDI**: XML protocol over TCP, via `pyindi-client`
- **KStars**: D-Bus interface, requires `dbus-python`

## Documentation

Project documentation is maintained in the [citra-space/docs](https://github.com/citra-space/docs) repository under `docs/citrasense/`.
