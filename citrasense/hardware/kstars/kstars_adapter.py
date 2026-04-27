import json
import logging
import shutil
import time
from pathlib import Path

import dbus  # type: ignore[reportMissingImports]
from platformdirs import user_cache_dir, user_data_dir

from citrasense.hardware.abstract_astro_hardware_adapter import (
    AbstractAstroHardwareAdapter,
    ObservationStrategy,
    SettingSchemaEntry,
)


class KStarsDBusAdapter(AbstractAstroHardwareAdapter):
    """
    Adapter for controlling astronomical equipment through KStars via DBus.

    .. warning::

        **Multi-sensor limitation — one KStars/Ekos instance per rig.**

        The Ekos Scheduler is a *singleton inside one KStars process*. When
        we load a job we call ``scheduler.removeAllJobs()`` first to avoid
        state conflicts from stale jobs in the scheduler. If two
        ``KStarsDBusAdapter`` instances (two sensors) point at the *same*
        KStars DBus bus they will repeatedly wipe each other's queued jobs.

        For multi-sensor deployments each telescope must run its own
        KStars/Ekos process on its own DBus session bus (or equivalent
        isolation such as a separate user / X display). The operator is
        expected to configure distinct ``bus_name``/environment per rig.

    DBus Interface Documentation (from introspection):

    Mount Interface (org.kde.kstars.Ekos.Mount):
      Methods:
        - slew(double RA, double DEC) -> bool: Slew telescope to coordinates
        - sync(double RA, double DEC) -> bool: Sync telescope at coordinates
        - abort() -> bool: Abort current slew
        - park() -> bool: Park telescope
        - unpark() -> bool: Unpark telescope

      Properties:
        - equatorialCoords (ad): Current RA/Dec as list of doubles [RA, Dec]
        - slewStatus (i): Current slew status (0=idle, others=slewing)
        - status (i): Mount status enumeration
        - canPark (b): Whether mount supports parking

    Scheduler Interface (org.kde.kstars.Ekos.Scheduler):
      Methods:
        - loadScheduler(string fileURL) -> bool: Load ESL scheduler file
        - setSequence(string sequenceFileURL): Set sequence file (ESQ)
        - start(): Start scheduler execution
        - stop(): Stop scheduler
        - removeAllJobs(): Clear all jobs
        - resetAllJobs(): Reset job states

      Properties:
        - status (i): Scheduler state enumeration
        - currentJobName (s): Name of currently executing job
        - jsonJobs (s): JSON representation of all jobs

      Signals:
        - newStatus(int status): Emitted when scheduler state changes
    """

    def __init__(self, logger: logging.Logger, images_dir: Path, **kwargs):
        """
        Initialize the KStars DBus adapter.

        Args:
            logger: Logger instance for logging messages
            images_dir: Path to the images directory
            **kwargs: Configuration including bus_name, ccd_name, filter_wheel_name
        """
        super().__init__(images_dir=images_dir, **kwargs)
        self.logger: logging.Logger = logger
        self.bus_name = kwargs.get("bus_name") or "org.kde.kstars"
        self.ccd_name = kwargs.get("ccd_name") or "CCD Simulator"
        self.filter_wheel_name = kwargs.get("filter_wheel_name") or ""
        self.optical_train_name = kwargs.get("optical_train_name") or "Primary"

        # Capture parameters
        self.exposure_time = kwargs.get("exposure_time", 5.0)
        self.frame_count = kwargs.get("frame_count", 1)
        self.binning_x = kwargs.get("binning_x", 1)
        self.binning_y = kwargs.get("binning_y", 1)
        self.image_format = kwargs.get("image_format", "Mono")

        self.bus: dbus.SessionBus | None = None
        self.kstars: dbus.Interface | None = None
        self.ekos: dbus.Interface | None = None
        self._ekos_mount: dbus.Interface | None = None
        self._ekos_camera: dbus.Interface | None = None
        self.scheduler: dbus.Interface | None = None

    @classmethod
    def get_settings_schema(cls, **kwargs) -> list[SettingSchemaEntry]:
        """
        Return a schema describing configurable settings for the KStars DBus adapter.
        """
        return [
            {
                "name": "bus_name",
                "friendly_name": "D-Bus Service Name",
                "type": "str",
                "default": "org.kde.kstars",
                "description": "D-Bus service name for KStars (default: org.kde.kstars)",
                "required": False,
                "placeholder": "org.kde.kstars",
                "group": "Connection",
            },
            {
                "name": "ccd_name",
                "friendly_name": "Camera/CCD Device Name",
                "type": "str",
                "default": "CCD Simulator",
                "description": (
                    "Name of the camera device in your Ekos profile "
                    "(check Ekos logs on connect for available devices)"
                ),
                "required": False,
                "placeholder": "CCD Simulator",
                "group": "Devices",
            },
            {
                "name": "filter_wheel_name",
                "friendly_name": "Filter Wheel Device Name",
                "type": "str",
                "default": "",
                "description": "Name of the filter wheel device (leave empty if no filter wheel)",
                "required": False,
                "placeholder": "Filter Simulator",
                "group": "Devices",
            },
            {
                "name": "optical_train_name",
                "friendly_name": "Optical Train Name",
                "type": "str",
                "default": "Primary",
                "description": (
                    "Name of the optical train in your Ekos profile "
                    "(check Ekos logs on connect for available trains)"
                ),
                "required": False,
                "placeholder": "Primary",
                "group": "Devices",
            },
            {
                "name": "exposure_time",
                "friendly_name": "Exposure Time (seconds)",
                "type": "float",
                "default": 1.0,
                "description": "Exposure duration in seconds for each frame",
                "required": False,
                "placeholder": "1.0",
                "min": 0.001,
                "max": 300.0,
                "group": "Imaging",
            },
            {
                "name": "frame_count",
                "friendly_name": "Frame Count",
                "type": "int",
                "default": 1,
                "description": "Number of frames to capture per observation",
                "required": False,
                "placeholder": "1",
                "min": 1,
                "max": 100,
                "group": "Imaging",
            },
            {
                "name": "binning_x",
                "friendly_name": "Binning X",
                "type": "int",
                "default": 1,
                "description": "Horizontal pixel binning (1=no binning, 2=2x2, etc.)",
                "required": False,
                "placeholder": "1",
                "min": 1,
                "max": 4,
                "group": "Imaging",
            },
            {
                "name": "binning_y",
                "friendly_name": "Binning Y",
                "type": "int",
                "default": 1,
                "description": "Vertical pixel binning (1=no binning, 2=2x2, etc.)",
                "required": False,
                "placeholder": "1",
                "min": 1,
                "max": 4,
                "group": "Imaging",
            },
            {
                "name": "image_format",
                "friendly_name": "Image Format",
                "type": "str",
                "default": "Mono",
                "description": "Camera image format (Mono for monochrome, RGGB/RGB for color cameras)",
                "required": False,
                "placeholder": "Mono",
                "options": ["Mono", "RGGB", "RGB"],
                "group": "Imaging",
            },
        ]

    def _do_point_telescope(self, ra: float, dec: float):
        """
        Point the telescope to the specified RA/Dec coordinates.

        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees

        Raises:
            RuntimeError: If mount is not connected or slew fails
        """
        if not self._ekos_mount:
            raise RuntimeError("Mount interface not connected. Call connect() first.")

        try:
            # Convert RA from degrees to hours for KStars (KStars expects RA in hours)
            ra_hours = ra / 15.0

            self.logger.info(f"Slewing telescope to RA={ra_hours:.4f}h ({ra:.4f}°), Dec={dec:.4f}°")

            # Call the slew method via DBus
            success = self._ekos_mount.slew(ra_hours, dec)

            if not success:
                raise RuntimeError(f"Mount slew command failed for RA={ra_hours}h, Dec={dec}°")

            self.logger.info("Slew command sent successfully")

        except Exception as e:
            self.logger.error(f"Failed to slew telescope: {e}")
            raise RuntimeError(f"Telescope slew failed: {e}") from e

    def get_observation_strategy(self) -> ObservationStrategy:
        return ObservationStrategy.SEQUENCE_TO_CONTROLLER

    def _load_template(self, template_name: str) -> str:
        """Load a template file from the hardware directory."""
        template_path = Path(__file__).parent / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        return template_path.read_text()

    def _create_sequence_file(self, task_id: str, satellite_data: dict, output_dir: Path, task=None) -> Path:
        """
        Create an ESQ sequence file from template.

        Args:
            task_id: Unique task identifier
            satellite_data: Dictionary containing target information
            output_dir: Base output directory for captures
            task: Optional task object containing filter assignment

        Returns:
            Path to the created sequence file
        """
        template = self._load_template("sequence_template.esq")

        # Extract target info
        target_name = satellite_data.get("name", "Unknown").replace(" ", "_")

        # Generate job blocks based on filter configuration
        jobs_xml = self._generate_job_blocks(output_dir, task)

        # Replace placeholders
        sequence_content = template.replace("{{JOBS}}", jobs_xml)
        sequence_content = sequence_content.replace("{{OUTPUT_DIR}}", str(output_dir))
        sequence_content = sequence_content.replace("{{TASK_ID}}", task_id)
        sequence_content = sequence_content.replace("{{TARGET_NAME}}", target_name)
        sequence_content = sequence_content.replace("{{CCD_NAME}}", self.ccd_name)
        sequence_content = sequence_content.replace("{{FILTER_WHEEL_NAME}}", self.filter_wheel_name)
        sequence_content = sequence_content.replace("{{OPTICAL_TRAIN}}", self.optical_train_name)

        # Write to temporary file
        temp_dir = Path(user_cache_dir("citrasense")) / "kstars"
        temp_dir.mkdir(exist_ok=True, parents=True)
        sequence_file = temp_dir / f"{task_id}_sequence.esq"
        sequence_file.write_text(sequence_content)

        self.logger.info(f"Created sequence file: {sequence_file}")
        return sequence_file

    def _generate_job_blocks(self, output_dir: Path, task=None) -> str:
        """
        Generate XML job blocks for each filter in filter_map.
        If no filters discovered, generates single job with no filter.

        Args:
            output_dir: Base output directory for captures
            task: Optional task object containing filter assignment

        Returns:
            XML string containing one or more <Job> blocks
        """
        job_template = """  <Job>
    <Exposure>{exposure}</Exposure>
    <Format>{format}</Format>
    <Encoding>FITS</Encoding>
    <Binning>
      <X>{binning_x}</X>
      <Y>{binning_y}</Y>
    </Binning>
    <Frame>
      <X>0</X>
      <Y>0</Y>
      <W>0</W>
      <H>0</H>
    </Frame>
    <Temperature force='false'>0</Temperature>
    <Filter>{filter_name}</Filter>
    <Type>Light</Type>
    <Count>{count}</Count>
    <Delay>0</Delay>
    <GuideDitherPerJob>0</GuideDitherPerJob>
    <FITSDirectory>{output_dir}</FITSDirectory>
    <PlaceholderFormat>%t_%F</PlaceholderFormat>
    <PlaceholderSuffix>0</PlaceholderSuffix>
    <UploadMode>0</UploadMode>
    <Properties>
</Properties>
    <Calibration>
      <PreAction>
        <Type>1</Type>
      </PreAction>
      <FlatDuration dark='false'>
        <Type>Manual</Type>
      </FlatDuration>
    </Calibration>
  </Job>
"""

        jobs = []

        # Select filters to use for this task (allow_no_filter for KStars '--' fallback)
        filters_to_use = self.select_filters_for_task(task, allow_no_filter=True)

        if filters_to_use is None:
            # No filters available - use '--' for no filter wheel
            filter_name = "--" if not self.filter_wheel_name else "Luminance"
            task_id_str = task.id if task else "unknown"
            self.logger.info(f"Using fallback filter '{filter_name}' for task {task_id_str}")

            job_xml = job_template.format(
                exposure=self.exposure_time,
                format=self.image_format,
                binning_x=self.binning_x,
                binning_y=self.binning_y,
                filter_name=filter_name,
                count=self.frame_count,
                output_dir=str(output_dir),
            )
            jobs.append(job_xml)
            return "\n".join(jobs)

        # Generate jobs for selected filters
        for filter_idx in sorted(filters_to_use.keys()):
            filter_info = filters_to_use[filter_idx]
            filter_name = filter_info["name"]

            job_xml = job_template.format(
                exposure=self.exposure_time,
                format=self.image_format,
                binning_x=self.binning_x,
                binning_y=self.binning_y,
                filter_name=filter_name,
                count=self.frame_count,
                output_dir=str(output_dir),
            )
            jobs.append(job_xml)

        return "\n".join(jobs)

    def _create_scheduler_job(self, task_id: str, satellite_data: dict, sequence_file: Path) -> Path:
        """
        Create an ESL scheduler job file from template.

        Args:
            task_id: Unique task identifier
            satellite_data: Dictionary containing target coordinates
            sequence_file: Path to the ESQ sequence file

        Returns:
            Path to the created scheduler job file
        """
        template = self._load_template("scheduler_template.esl")

        # Extract target info
        target_name = satellite_data.get("name", "Unknown")
        ra_deg = satellite_data.get("ra", 0.0)  # RA in degrees
        dec_deg = satellite_data.get("dec", 0.0)  # Dec in degrees

        # Convert RA from degrees to hours for Ekos
        ra_hours = ra_deg / 15.0

        self.logger.info(f"Target: {target_name} at RA={ra_deg:.4f}° ({ra_hours:.4f}h), Dec={dec_deg:.4f}°")

        # Replace placeholders
        job_name = f"CitraSense: {target_name} (Task: {task_id})"
        scheduler_content = template.replace("{{JOB_NAME}}", job_name)
        scheduler_content = scheduler_content.replace("{{TARGET_RA}}", f"{ra_hours:.6f}")
        scheduler_content = scheduler_content.replace("{{TARGET_DEC}}", f"{dec_deg:.6f}")
        scheduler_content = scheduler_content.replace("{{SEQUENCE_FILE}}", str(sequence_file))
        scheduler_content = scheduler_content.replace("{{MIN_ALTITUDE}}", "0")  # 0° minimum altitude for satellites

        # Write to temporary file
        temp_dir = Path(user_cache_dir("citrasense")) / "kstars"
        temp_dir.mkdir(exist_ok=True, parents=True)
        job_file = temp_dir / f"{task_id}_job.esl"
        job_file.write_text(scheduler_content)

        self.logger.info(f"Created scheduler job: {job_file}")
        return job_file

    def _wait_for_job_completion(self, timeout: int = 300, task_id: str = "", output_dir: Path | None = None) -> bool:
        """
        Poll the scheduler status until job completes or times out.
        With Loop completion, we poll for images and stop when we have all expected images.
        For multi-filter sequences, waits until images from all filters are captured.

        Args:
            timeout: Maximum time to wait in seconds
            task_id: Task identifier for image detection
            output_dir: Output directory for image detection

        Returns:
            True if job completed successfully, False otherwise
        """
        if not self.scheduler:
            raise RuntimeError("Scheduler interface not connected")

        assert self.bus is not None

        # Calculate expected number of images based on enabled filters
        enabled_filter_count = (
            sum(1 for f in self.filter_map.values() if f.get("enabled", True)) if self.filter_map else 1
        )
        expected_total_images = enabled_filter_count * self.frame_count

        self.logger.info(
            f"Waiting for scheduler job completion (timeout: {timeout}s, "
            f"expecting {expected_total_images} images across {enabled_filter_count} filters)..."
        )
        start_time = time.time()

        # Get scheduler object for property access
        scheduler_obj = self.bus.get_object(self.bus_name, "/KStars/Ekos/Scheduler")
        props = dbus.Interface(scheduler_obj, "org.freedesktop.DBus.Properties")

        while time.time() - start_time < timeout:
            try:
                # Get scheduler status (0=Idle, 1=Running, 2=Paused, etc.)
                status = int(props.Get("org.kde.kstars.Ekos.Scheduler", "status"))
                current_job = props.Get("org.kde.kstars.Ekos.Scheduler", "currentJobName")

                self.logger.debug(f"Scheduler status: {status}, Current job: {current_job}")

                # Check for images if we're using Loop completion
                if task_id and output_dir:
                    images = self._retrieve_captured_images(task_id, output_dir)
                    if len(images) >= expected_total_images:
                        self.logger.info(
                            f"Found {len(images)} images (expected {expected_total_images}), stopping scheduler"
                        )
                        self.scheduler.stop()
                        time.sleep(1)  # Give it time to stop
                        return True
                    elif images:
                        self.logger.debug(f"Found {len(images)}/{expected_total_images} images so far, continuing...")

                # Status 0 = Idle, meaning job finished or not started
                # If we were running and now idle, job completed
                if status == 0 and current_job == "":
                    self.logger.info("Scheduler job completed")
                    return True

                time.sleep(5)  # Poll every 5 seconds (slower since we're checking files)

            except dbus.DBusException as e:
                if "ServiceUnknown" in str(e) or "NoReply" in str(e):
                    self.logger.error("KStars appears to have crashed or disconnected")
                    return False
                self.logger.warning(f"Error checking scheduler status: {e}")
                time.sleep(2)
            except Exception as e:
                self.logger.warning(f"Error checking scheduler status: {e}")
                time.sleep(2)

        self.logger.error(f"Scheduler job did not complete within {timeout}s")
        return False

    def _retrieve_captured_images(self, task_id: str, output_dir: Path) -> list[str]:
        """
        Find and return paths to captured images for this task.

        Args:
            task_id: Unique task identifier
            output_dir: Base output directory where images were saved

        Returns:
            List of absolute paths to captured FITS files
        """
        self.logger.debug(f"Looking for captured images in: {output_dir}")

        # Check if base output directory exists
        if not output_dir.exists():
            self.logger.warning(f"Base output directory does not exist: {output_dir}")
            # List parent directory to see what's there
            parent = output_dir.parent
            if parent.exists():
                self.logger.debug(f"Parent directory contents: {list(parent.iterdir())}")
            return []

        # List what's in the base directory
        self.logger.debug(f"Base directory contents: {list(output_dir.iterdir())}")

        # Look for images in task-specific subdirectory
        task_dir = output_dir / task_id

        if not task_dir.exists():
            self.logger.error(f"Task directory does not exist: {task_dir}")
            self.logger.error("This likely indicates Ekos failed to create the capture directory")
            self.logger.error(f"Expected directory structure: {output_dir}/{task_id}/")
            raise RuntimeError(
                f"Task-specific capture directory not found: {task_dir}. "
                f"Ekos may have failed to start the capture sequence."
            )

        # Find all FITS files in task directory and subdirectories
        fits_files = list(task_dir.rglob("*.fits")) + list(task_dir.rglob("*.fit"))

        # Since files are in task-specific directory, we don't need to filter by filename
        matching_files = [str(f.absolute()) for f in fits_files]

        self.logger.info(f"Found {len(matching_files)} captured images for task {task_id}")
        for img_path in matching_files:
            self.logger.debug(f"  - {img_path}")

        return matching_files

    def perform_observation_sequence(self, task, satellite_data: dict) -> list[str]:
        """
        Execute a complete observation sequence using Ekos Scheduler.

        Args:
            task: Task object containing id and filter assignment
            satellite_data: Dictionary with keys: 'name', and either 'ra'/'dec' or TLE data

        Returns:
            List of paths to captured FITS files

        Raises:
            RuntimeError: If scheduler not connected or job execution fails
        """
        if not self.scheduler:
            raise RuntimeError("Scheduler interface not connected. Call connect() first.")

        # Calculate current position if not already provided
        if "ra" not in satellite_data or "dec" not in satellite_data:
            # For now, require RA/Dec to be provided by caller
            # TODO: Add TLE propagation capability to adapter for full autonomy
            raise ValueError("satellite_data must include 'ra' and 'dec' keys (in degrees)")

        try:
            # Setup output directory
            output_dir = Path(user_data_dir("citrasense")) / "kstars_captures"
            output_dir.mkdir(exist_ok=True, parents=True)

            # Clear task-specific directory to prevent Ekos from thinking job is already done
            task_output_dir = output_dir / task.id
            if task_output_dir.exists():
                shutil.rmtree(task_output_dir)
                self.logger.info(f"Cleared existing output directory: {task_output_dir}")

            # Create task directory for this observation
            task_output_dir.mkdir(exist_ok=True, parents=True)
            self.logger.info(f"Output directory: {task_output_dir}")

            # Create sequence and scheduler job files (use task-specific directory)
            sequence_file = self._create_sequence_file(task.id, satellite_data, task_output_dir, task)
            job_file = self._create_scheduler_job(task.id, satellite_data, sequence_file)

            # Ensure temp files are cleaned up even on failure
            try:
                self._execute_observation(task.id, output_dir, sequence_file, job_file)
            finally:
                # Cleanup temp files
                self._cleanup_temp_files(sequence_file, job_file)

            # Retrieve and return captured images
            image_paths = self._retrieve_captured_images(task.id, output_dir)
            if not image_paths:
                raise RuntimeError(f"No images captured for task {task.id}")

            self.logger.info(f"Observation sequence complete: {len(image_paths)} images captured")
            return image_paths

        except Exception as e:
            self.logger.error(f"Failed to execute observation sequence: {e}")
            raise

    def _execute_observation(self, task_id: str, output_dir: Path, sequence_file: Path, job_file: Path):
        """Execute the observation by loading scheduler job and waiting for completion.

        Args:
            task_id: Task identifier
            output_dir: Base output directory
            sequence_file: Path to ESQ sequence file
            job_file: Path to ESL scheduler job file
        """
        assert self.scheduler is not None
        assert self.bus is not None

        # Load scheduler job via DBus
        self.logger.info(f"Loading scheduler job: {job_file}")

        # Verify files exist and have content
        if not job_file.exists():
            raise RuntimeError(f"Scheduler job file does not exist: {job_file}")
        if not sequence_file.exists():
            raise RuntimeError(f"Sequence file does not exist: {sequence_file}")

        self.logger.debug(f"Job file size: {job_file.stat().st_size} bytes")
        self.logger.debug(f"Sequence file size: {sequence_file.stat().st_size} bytes")

        # Load the scheduler job
        try:
            # Clear any existing jobs first to prevent state conflicts.
            #
            # NOTE: ``removeAllJobs`` wipes *every* queued job on the Ekos
            # scheduler. Sharing one KStars instance between two CitraSense
            # telescope sensors is therefore unsafe — each sensor would
            # clobber the other's queue here. Deployments with >1 telescope
            # must run one KStars process per rig (distinct DBus bus).
            try:
                self.scheduler.removeAllJobs()
                self.logger.info(
                    "Cleared existing scheduler jobs (Ekos singleton — "
                    "one KStars process per telescope sensor required)"
                )
                time.sleep(0.5)
            except Exception as clear_error:
                self.logger.warning(f"Could not clear jobs (might not exist): {clear_error}")

            success = self.scheduler.loadScheduler(str(job_file))
            self.logger.debug(f"loadScheduler() returned: {success}")
        except Exception as dbus_error:
            self.logger.error(f"DBus error calling loadScheduler: {dbus_error}")
            raise RuntimeError(f"DBus error loading scheduler job: {dbus_error}") from dbus_error

        if not success:
            # Log file contents for debugging
            self.logger.error("Scheduler rejected job file. Contents:")
            self.logger.error(job_file.read_text()[:500])  # First 500 chars
            raise RuntimeError(f"Ekos Scheduler rejected job file: {job_file}")

        self.logger.info("Scheduler job loaded successfully")

        # Verify what was loaded before starting
        try:
            scheduler_obj = self.bus.get_object(self.bus_name, "/KStars/Ekos/Scheduler")
            props = dbus.Interface(scheduler_obj, "org.freedesktop.DBus.Properties")
            json_jobs = props.Get("org.kde.kstars.Ekos.Scheduler", "jsonJobs")
            self.logger.info(f"Loaded jobs: {json_jobs}")

            # Parse and validate the job looks correct
            jobs = json.loads(str(json_jobs))
            if jobs:
                job = jobs[0]  # We only load one job at a time
                self.logger.info(f"Loaded {len(jobs)} job(s):")
                self.logger.info(f"  Name: {job.get('name', 'Unknown')}")
                self.logger.info(f"  State: {job.get('state', 'Unknown')}")
                self.logger.info(f"  RA: {job.get('targetRA', 'N/A')}h, Dec: {job.get('targetDEC', 'N/A')}°")
                self.logger.info(f"  Altitude: {job.get('altitudeFormatted', 'N/A')}")
                self.logger.info(f"  Repeats: {job.get('repeatsRemaining', 0)}/{job.get('repeatsRequired', 0)}")
                self.logger.info(f"  Completed: {job.get('completedCount', 0)}")
            else:
                self.logger.warning("No jobs found in scheduler after loading!")
        except Exception as e:
            self.logger.warning(f"Could not validate loaded jobs: {e}")

        # Start scheduler
        self.logger.info("Starting scheduler execution...")
        self.scheduler.start()

        # Give it a moment to start
        time.sleep(1)

        # Check scheduler logs immediately after starting
        try:
            scheduler_obj = self.bus.get_object(self.bus_name, "/KStars/Ekos/Scheduler")
            props = dbus.Interface(scheduler_obj, "org.freedesktop.DBus.Properties")
            log_lines = props.Get("org.kde.kstars.Ekos.Scheduler", "logText")
            if log_lines:
                self.logger.info("Scheduler logs after start:")
                for line in log_lines[-10:]:  # Last 10 lines
                    self.logger.info(f"  Ekos: {line}")
        except Exception as e:
            self.logger.debug(f"Could not read scheduler logs: {e}")

        # Wait for completion (with Loop mode, this polls for images and stops when found)
        if not self._wait_for_job_completion(timeout=300, task_id=task_id, output_dir=output_dir):
            raise RuntimeError("Scheduler job did not complete in time")

    def _cleanup_temp_files(self, sequence_file: Path, job_file: Path):
        """Clean up temporary ESQ and ESL files.

        Args:
            sequence_file: Path to ESQ sequence file
            job_file: Path to ESL scheduler job file
        """
        try:
            if sequence_file.exists():
                sequence_file.unlink()
                self.logger.debug(f"Cleaned up sequence file: {sequence_file.name}")
            if job_file.exists():
                job_file.unlink()
                self.logger.debug(f"Cleaned up job file: {job_file.name}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp files: {e}")

    def connect(self) -> bool:
        """
        Connect to KStars via DBus and initialize the Ekos session.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Connect to the session bus
            self.logger.info("Connecting to DBus session bus...")
            self.bus = dbus.SessionBus()
            assert self.bus is not None

            # Get the KStars service
            try:
                kstars_obj = self.bus.get_object(self.bus_name, "/KStars")
                self.kstars = dbus.Interface(kstars_obj, dbus_interface="org.kde.kstars")
                self.logger.info("Connected to KStars DBus interface")
            except dbus.DBusException as e:
                self.logger.error(f"Failed to connect to KStars: {e}")
                self.logger.error("Make sure KStars is running and DBus is enabled")
                return False

            # Get the Ekos interface
            try:
                ekos_obj = self.bus.get_object(self.bus_name, "/KStars/Ekos")
                self.ekos = dbus.Interface(ekos_obj, dbus_interface="org.kde.kstars.Ekos")
                self.logger.info("Connected to Ekos interface")
            except dbus.DBusException as e:
                self.logger.warning(f"Failed to connect to Ekos interface: {e}")
                self.logger.warning("Attempting to start Ekos...")

                # Try to start Ekos if it's not running
                try:
                    assert self.kstars is not None
                    self.kstars.startEkos()
                    time.sleep(2)  # Give Ekos time to start
                    ekos_obj = self.bus.get_object(self.bus_name, "/KStars/Ekos")
                    self.ekos = dbus.Interface(ekos_obj, dbus_interface="org.kde.kstars.Ekos")
                    self.logger.info("Started and connected to Ekos interface")
                except Exception as start_error:
                    self.logger.error(f"Failed to start Ekos: {start_error}")
                    return False

            # Get Mount interface
            try:
                mount_obj = self.bus.get_object(self.bus_name, "/KStars/Ekos/Mount")
                self._ekos_mount = dbus.Interface(mount_obj, dbus_interface="org.kde.kstars.Ekos.Mount")
                self.logger.info("Connected to Mount interface")
            except dbus.DBusException as e:
                self.logger.warning(f"Mount interface not available: {e}")

            # Get Camera interface
            try:
                camera_obj = self.bus.get_object(self.bus_name, "/KStars/Ekos/Camera")
                self._ekos_camera = dbus.Interface(camera_obj, dbus_interface="org.kde.kstars.Ekos.Camera")
                self.logger.info("Connected to Camera interface")
            except dbus.DBusException as e:
                self.logger.warning(f"Camera interface not available: {e}")

            # Get Scheduler/Sequence interface
            try:
                scheduler_obj = self.bus.get_object(self.bus_name, "/KStars/Ekos/Scheduler")
                self.scheduler = dbus.Interface(scheduler_obj, dbus_interface="org.kde.kstars.Ekos.Scheduler")
                self.logger.info("Connected to Scheduler interface")
            except dbus.DBusException as e:
                self.logger.warning(f"Scheduler interface not available: {e}")

            # Validate devices and imaging train
            self._validate_devices()

            # Discover available filters (non-fatal if fails)
            self.discover_filters()

            self.logger.info("Successfully connected to KStars via DBus")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to KStars via DBus: {e}")
            return False

    def _validate_devices(self):
        """Check what optical train/devices are configured in Ekos."""
        try:
            assert self.bus is not None
            # Use Capture module (not Camera)
            capture_obj = self.bus.get_object(self.bus_name, "/KStars/Ekos/Capture")
            props = dbus.Interface(capture_obj, "org.freedesktop.DBus.Properties")

            optical_train = props.Get("org.kde.kstars.Ekos.Capture", "opticalTrain")
            camera_name = props.Get("org.kde.kstars.Ekos.Capture", "camera")
            filter_wheel = props.Get("org.kde.kstars.Ekos.Capture", "filterWheel")

            self.logger.info(f"Ekos optical train: {optical_train}")
            self.logger.info(f"Ekos camera device: {camera_name}")
            self.logger.info(f"Ekos filter wheel: {filter_wheel}")

        except Exception as e:
            self.logger.warning(f"Could not read Ekos devices: {e}")
            # Non-fatal - continue with defaults

    def discover_filters(self):
        """Discover available filters from Ekos filter wheel via INDI interface.

        This is called during connect() to populate filter_map.
        Uses INDI interface to query FILTER_NAME properties for each slot.
        If no filter wheel is configured or discovery fails, filter_map remains empty
        and adapter falls back to single-filter behavior.
        """
        try:
            if not self.bus:
                self.logger.debug("Cannot discover filters: DBus not connected")
                return

            self.logger.info("Attempting to discover filters...")

            # Get filter wheel device name from Capture module
            capture_obj = self.bus.get_object(self.bus_name, "/KStars/Ekos/Capture")
            capture_props = dbus.Interface(capture_obj, "org.freedesktop.DBus.Properties")

            try:
                filter_wheel_name = capture_props.Get("org.kde.kstars.Ekos.Capture", "filterWheel")
                if not filter_wheel_name or filter_wheel_name == "--":
                    self.logger.info("No filter wheel configured in Capture module")
                    return
                self.logger.info(f"Filter wheel detected: {filter_wheel_name}")
            except Exception as e:
                self.logger.debug(f"Could not get filter wheel name: {e}")
                return

            # Use INDI interface to query filter properties
            indi_obj = self.bus.get_object(self.bus_name, "/KStars/INDI")
            indi_iface = dbus.Interface(indi_obj, "org.kde.kstars.INDI")

            # Get all properties for the filter wheel device
            properties = indi_iface.getProperties(filter_wheel_name)

            # Find FILTER_NAME properties (FILTER_SLOT_NAME_1, FILTER_SLOT_NAME_2, etc.)
            filter_slots = []
            for prop in properties:
                if "FILTER_NAME.FILTER_SLOT_NAME_" in prop:
                    slot_num = prop.split("_")[-1]
                    try:
                        filter_slots.append(int(slot_num))
                    except ValueError:
                        continue

            if not filter_slots:
                self.logger.warning(f"No FILTER_NAME properties found for {filter_wheel_name}")
                return

            # Query each filter slot name and merge with pre-populated filter_map
            filter_slots.sort()
            for slot_num in filter_slots:
                try:
                    filter_name = indi_iface.getText(filter_wheel_name, "FILTER_NAME", f"FILTER_SLOT_NAME_{slot_num}")
                    # Use 0-based indexing for filter_map (slot 1 -> index 0)
                    filter_idx = slot_num - 1

                    # If filter already in map (from saved settings), preserve focus position and enabled state
                    if filter_idx in self.filter_map:
                        focus_position = self.filter_map[filter_idx].get("focus_position", 0)
                        enabled = self.filter_map[filter_idx].get("enabled", True)
                        self.logger.debug(
                            f"Filter slot {slot_num} ({filter_name}): using saved focus "
                            f"position {focus_position}, enabled: {enabled}"
                        )
                    else:
                        focus_position = 0
                        enabled = True  # Default new filters to enabled
                        self.logger.debug(
                            f"Filter slot {slot_num} ({filter_name}): new filter, using default focus position"
                        )

                    self.filter_map[filter_idx] = {
                        "name": filter_name,
                        "focus_position": focus_position,
                        "enabled": enabled,
                    }
                except Exception as e:
                    self.logger.warning(f"Could not read filter slot {slot_num}: {e}")

            if self.filter_map:
                self.logger.info(
                    f"Discovered {len(self.filter_map)} filters: {[f['name'] for f in self.filter_map.values()]}"
                )
            else:
                self.logger.warning("No filters discovered from filter wheel")

        except Exception as e:
            self.logger.info(f"Filter discovery failed (non-fatal): {e}")
            # Leave filter_map empty, use single-filter mode

    def supports_autofocus(self) -> bool:
        """Indicates that KStars adapter does not support autofocus yet.

        Returns:
            bool: False (autofocus not implemented).
        """
        return False

    def supports_filter_management(self) -> bool:
        """Indicates whether this adapter supports filter/focus management.

        Returns:
            bool: True if filters were discovered, False otherwise.
        """
        return bool(self.filter_map)

    def disconnect(self):
        raise NotImplementedError

    def is_telescope_connected(self) -> bool:
        """Check if telescope is connected and responsive."""
        if not self._ekos_mount or not self.bus:
            return False
        try:
            # Actually test the connection by reading a property
            mount_obj = self.bus.get_object(self.bus_name, "/KStars/Ekos/Mount")
            props = dbus.Interface(mount_obj, "org.freedesktop.DBus.Properties")
            props.Get("org.kde.kstars.Ekos.Mount", "status")
            return True
        except (dbus.DBusException, Exception):
            return False

    def is_camera_connected(self) -> bool:
        """Check if camera is connected and responsive."""
        if not self._ekos_camera or not self.bus:
            return False
        try:
            # Actually test the connection by reading a property
            capture_obj = self.bus.get_object(self.bus_name, "/KStars/Ekos/Capture")
            props = dbus.Interface(capture_obj, "org.freedesktop.DBus.Properties")
            props.Get("org.kde.kstars.Ekos.Capture", "status")
            return True
        except (dbus.DBusException, Exception):
            return False

    def get_current_binning(self) -> tuple[int, int]:
        """Return configured imaging binning ``(binning_x, binning_y)``.

        These values are templated into the capture XML sent to KStars at
        every imaging site (see ``binning_x`` / ``binning_y`` in the sequence
        template), so they describe what the adapter actually asks for.
        Without this override, ``assess_config_health`` would assume ``(1, 1)``
        and false-flag a pixel-scale mismatch for any binned KStars imaging.
        """
        try:
            bx = max(1, int(self.binning_x))
            by = max(1, int(self.binning_y))
        except (TypeError, ValueError):
            return (1, 1)
        return (bx, by)

    def list_devices(self) -> list[str]:
        raise NotImplementedError

    def select_telescope(self, device_name: str) -> bool:
        raise NotImplementedError

    def get_telescope_direction(self) -> tuple[float, float]:
        """
        Get the current telescope pointing direction.

        Returns:
            tuple[float, float]: Current (RA, Dec) in degrees

        Raises:
            RuntimeError: If mount is not connected or position query fails
        """
        if not self._ekos_mount:
            raise RuntimeError("Mount interface not connected. Call connect() first.")

        assert self.bus is not None

        try:
            # Get the mount object for property access
            mount_obj = self.bus.get_object(self.bus_name, "/KStars/Ekos/Mount")
            props = dbus.Interface(mount_obj, "org.freedesktop.DBus.Properties")

            # Get equatorial coordinates property (returns list [RA in hours, Dec in degrees])
            coords = props.Get("org.kde.kstars.Ekos.Mount", "equatorialCoords")

            if not coords or len(coords) < 2:
                raise RuntimeError("Failed to retrieve valid coordinates from mount")

            # coords[0] is RA in hours, coords[1] is Dec in degrees
            ra_hours = float(coords[0])
            dec_deg = float(coords[1])

            # Convert RA from hours to degrees
            ra_deg = ra_hours * 15.0

            self.logger.debug(f"Current telescope position: RA={ra_deg:.4f}° ({ra_hours:.4f}h), Dec={dec_deg:.4f}°")

            return (ra_deg, dec_deg)

        except Exception as e:
            self.logger.error(f"Failed to get telescope position: {e}")
            raise RuntimeError(f"Failed to get telescope position: {e}") from e

    def telescope_is_moving(self) -> bool:
        """
        Check if the telescope is currently slewing.

        Returns:
            bool: True if telescope is slewing, False if idle or tracking

        Raises:
            RuntimeError: If mount is not connected or status query fails
        """
        if not self._ekos_mount:
            raise RuntimeError("Mount interface not connected. Call connect() first.")

        assert self.bus is not None

        try:
            # Get the mount object for property access
            mount_obj = self.bus.get_object(self.bus_name, "/KStars/Ekos/Mount")
            props = dbus.Interface(mount_obj, "org.freedesktop.DBus.Properties")

            # Get slewStatus property (0 = idle, non-zero = slewing)
            slew_status = props.Get("org.kde.kstars.Ekos.Mount", "slewStatus")

            is_slewing = int(slew_status) != 0

            self.logger.debug(f"Mount slew status: {slew_status} (is_slewing={is_slewing})")

            return is_slewing

        except Exception as e:
            self.logger.error(f"Failed to get telescope slew status: {e}")
            raise RuntimeError(f"Failed to get telescope slew status: {e}") from e

    def select_camera(self, device_name: str) -> bool:
        raise NotImplementedError

    def take_image(self, task_id: str, exposure_duration_seconds=1) -> str:
        raise NotImplementedError

    def set_custom_tracking_rate(self, ra_rate: float, dec_rate: float):
        raise NotImplementedError

    def get_tracking_rate(self) -> tuple[float, float]:
        raise NotImplementedError
