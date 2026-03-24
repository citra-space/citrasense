# Test FITS Injection

Drop real FITS images into this folder to run them through the full citrascope
processing pipeline without a telescope connected.

## Quick start

1. Copy one or more `.fits` files into this directory.
2. Open the citrascope web UI (`http://localhost:24872`).
3. In **Configuration**:
   - Set **Hardware Adapter** to **Dummy Adapter**.
   - Under the Dummy Adapter settings, set **Image Source** to **Real FITS Images**.
     The **FITS Directory** will appear pre-filled with this folder's path.
   - Enable **Image Processing Pipeline**.
   - Check **Use Dummy API** to avoid uploading results to a real server.
   - Set **Number of Exposures** to `1` if you only have one FITS file.
4. With the Dummy API, tasks are auto-generated for DirecTV GEO satellites. If
   you need a task targeting a specific satellite (matching your FITS image),
   use the dev API endpoint instead and create a matching task there.
5. Start the daemon (`python -m citrascope`). It will pick up the next due task,
   "image" by serving your FITS file, and run the full processing pipeline.
6. Check the web UI for the annotated image and the logs for satellite matcher
   results.

## How it works

When the Dummy Adapter's **FITS Directory** is set, `take_image()` copies the
next file from this folder (round-robin) instead of generating a synthetic
starfield. The processing pipeline then runs normally: plate solving, source
extraction, photometry, satellite matching, and annotation.

The copy step is important — the processing pipeline may modify the FITS file
in place (e.g. updating WCS headers after plate solving), so your originals in
this folder are never touched.

## Notes

- FITS files in this folder are **gitignored** (they're too large to commit).
  Only this README is tracked.
- If you have multiple FITS files, they are served in sorted filename order and
  cycle back to the first when exhausted.
- The dev API endpoint (`https://dev.api.citra.space`) is useful for testing
  without affecting production data.
