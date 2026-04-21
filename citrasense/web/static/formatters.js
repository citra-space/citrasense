/**
 * CitraSense Formatter Utilities
 *
 * Shared formatting functions for the dashboard UI.
 * These functions are exposed in the Alpine.js store for use in templates.
 */

/**
 * Strip ANSI color codes from text
 * @param {string} text - Text containing ANSI codes
 * @returns {string} Text with ANSI codes removed
 */
export function stripAnsiCodes(text) {
    const esc = String.fromCharCode(27);
    return text.replace(new RegExp(esc + '\\[\\d+m', 'g'), '').replace(/\[\d+m/g, '');
}

/**
 * Format ISO date string to local time
 * @param {string} isoString - ISO 8601 date string
 * @returns {string} Formatted local time string
 */
export function formatLocalTime(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true
    });
}

/**
 * Compact local-time formatter for table cells.
 *
 * Shows just the time of day (`12:39:16 PM`) when the timestamp falls on
 * today's date — the date prefix is redundant in dense lists where every row
 * is "now-ish".  When the date differs from today, prepends `Apr 17 ` so the
 * day rollover is still legible.  Use `formatLocalTime` for cases where the
 * full date should always be visible.
 *
 * @param {string} isoString - ISO 8601 date string
 * @returns {string} e.g. "12:39:16 PM" or "Apr 17 12:39:16 AM"
 */
export function formatLocalTimeShort(isoString) {
    const date = new Date(isoString);
    const time = date.toLocaleTimeString(undefined, {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true
    });
    const now = new Date();
    if (date.toDateString() === now.toDateString()) return time;
    const day = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    return `${day} ${time}`;
}

/**
 * Bootstrap text-color class for an altitude readout.
 *
 * Color bands are keyed off the configured minimum observing elevation when
 * that's available, so red means "below the current observing floor" rather
 * than an arbitrary number.  When the threshold is unknown, falls back to a
 * conservative default so the bands are still meaningful.
 *
 *   * red     -- below the minimum elevation
 *   * yellow  -- within 10° of the threshold (low pass, marginal)
 *   * green   -- comfortably above
 *
 * @param {number|null|undefined} altDeg
 * @param {number|null|undefined} minElevationDeg - From status.telescope_min_elevation.
 * @returns {string} Bootstrap text-color utility class.
 */
export function skyAltClass(altDeg, minElevationDeg = null) {
    if (altDeg == null) return 'text-muted';
    // 15° matches the alignment_manager default when no minimum elevation is available.
    const limit = minElevationDeg != null ? minElevationDeg : 15;
    if (altDeg < limit) return 'text-danger';
    if (altDeg < limit + 10) return 'text-warning';
    return 'text-success';
}

/**
 * Convert the configured minimum-elevation floor into a dashed-ring radius on
 * the polar compass plot, so operators can see the "safe band" at a glance.
 * Returns `null` when the threshold is unknown or would coincide with the
 * outer rim, so the template can `x-if` the ring away.
 *
 * @param {number|null|undefined} minElevationDeg - From status.telescope_min_elevation.
 * @param {number} [size=40] - SVG side in px (matches skyCompassDot default).
 * @returns {number|null} Ring radius in px, or null.
 */
export function skyHorizonRingRadius(minElevationDeg, size = 40) {
    if (minElevationDeg == null || minElevationDeg <= 0 || minElevationDeg >= 90) return null;
    const center = size / 2;
    const radius = center - 3;  // matches skyCompassDot's outer radius
    return radius * (1 - minElevationDeg / 90);
}

/**
 * Convert a task's `(sky_az_deg, sky_alt_deg)` into the `(cx, cy)` for a dot
 * on a square polar-compass SVG.
 *
 * The plot uses standard astronomer convention: zenith at the center, horizon
 * at the outer ring, North at the top, East at the right (azimuth measured
 * clockwise from N).  Distance from center scales linearly with `1 - alt/90`,
 * which keeps high-elevation targets cleanly clustered near the middle.
 *
 * Returns `null` when the task has no sky position attached so the template
 * can `x-if` the dot away.
 *
 * @param {Object} task - Task dict with sky_alt_deg / sky_az_deg.
 * @param {number} [size=40] - SVG width/height in px.
 * @returns {{cx: number, cy: number}|null}
 */
export function skyCompassDot(task, size = 40) {
    if (!task || task.sky_alt_deg == null || task.sky_az_deg == null) return null;
    const center = size / 2;
    const radius = center - 3;  // leave 3px breathing room inside the rim
    const altClamped = Math.max(0, Math.min(90, task.sky_alt_deg));
    const r = radius * (1 - altClamped / 90);
    const azRad = (task.sky_az_deg * Math.PI) / 180;
    return {
        cx: center + r * Math.sin(azRad),
        cy: center - r * Math.cos(azRad),
    };
}

/**
 * Tooltip text for the Sky cell -- everything we computed but didn't render
 * visually.  Returns an empty string when no sky data is attached so the
 * `:title` binding stays inert.
 *
 * @param {Object} task - Task dict with optional sky_* fields.
 * @returns {string} Human-readable summary like "45° SSW · Rising · Peaks at 52° · 28° slew".
 */
export function formatSkyTitle(task) {
    if (!task || task.sky_alt_deg == null) return '';
    const parts = [];
    parts.push(`${Math.round(task.sky_alt_deg)}° ${task.sky_compass || ''}`.trim());
    if (task.sky_trend && task.sky_trend !== 'flat') {
        parts.push(task.sky_trend === 'rising' ? 'Rising' : 'Setting');
    }
    if (task.sky_max_alt_deg != null && Math.abs(task.sky_max_alt_deg - task.sky_alt_deg) > 1) {
        parts.push(`Peaks at ${Math.round(task.sky_max_alt_deg)}°`);
    }
    if (task.slew_from_current_deg != null) {
        parts.push(`${Math.round(task.slew_from_current_deg)}° slew from current`);
    }
    return parts.join(' · ');
}

/**
 * Format milliseconds as countdown string
 * @param {number} milliseconds - Time in milliseconds
 * @returns {string} Formatted countdown string (e.g., "2h 30m 15s")
 */
export function formatCountdown(milliseconds) {
    const totalSeconds = Math.floor(milliseconds / 1000);
    if (totalSeconds < 0) return 'Starting soon...';
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
    if (minutes > 0) return `${minutes}m ${seconds}s`;
    return `${seconds}s`;
}

/**
 * Format elapsed time for "X ago" display
 * @param {number} milliseconds - Elapsed time in milliseconds
 * @returns {string} Human-readable elapsed time (e.g., "2 hours ago")
 */
export function formatElapsedTime(milliseconds) {
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    if (days > 0) return `${days} day${days !== 1 ? 's' : ''} ago`;
    if (hours > 0) return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
    if (minutes > 0) return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`;
    return 'just now';
}

/**
 * Format minutes as "Xh Ym" display
 * @param {number} minutes - Time in minutes
 * @returns {string} Formatted time string (e.g., "2h 30m")
 */
export function formatMinutes(minutes) {
    const hours = Math.floor(minutes / 60);
    const mins = Math.floor(minutes % 60);
    if (hours > 0) return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
    return `${mins}m`;
}

/**
 * Format last autofocus timestamp
 * @param {Object} status - Status object containing last_autofocus_timestamp
 * @returns {string} Formatted autofocus time or "Never"
 */
export function formatLastAutofocus(status) {
    if (!status || !status.last_autofocus_timestamp) return 'Never';
    return formatTimestamp(status.last_autofocus_timestamp);
}

/**
 * Format a Unix timestamp (seconds) as a relative elapsed time string.
 * Returns "Never" for missing/invalid timestamps.
 */
export function formatTimestamp(ts) {
    if (!ts || ts < 1577836800) return 'Never'; // 2020-01-01T00:00:00Z
    const elapsed = Date.now() - ts * 1000;
    if (elapsed < 0) return 'Never';
    return formatElapsedTime(elapsed);
}

/**
 * Format time offset with source information - compact format for status pill
 * @param {Object} timeHealth - Time health object with offset_ms, source, and optional metadata
 * @returns {string} Formatted time offset (e.g., "17ns, 10 sats" or "+2ms, ntp")
 */
export function formatTimeOffset(timeHealth) {
    if (!timeHealth || timeHealth.offset_ms == null) return 'Unknown';

    const o = timeHealth.offset_ms;
    const abs = Math.abs(o);
    const s = o >= 0 ? '+' : '';

    // Format offset with appropriate units
    let offsetStr;
    if (abs < 0.001) {
        // Sub-microsecond: show as nanoseconds
        offsetStr = `${s}${Math.round(abs * 1000000)}ns`;
    } else if (abs < 1) {
        // Sub-millisecond: show as microseconds
        offsetStr = `${s}${Math.round(abs * 1000)}µs`;
    } else if (abs < 1000) {
        // Milliseconds
        offsetStr = `${s}${abs.toFixed(0)}ms`;
    } else {
        // Seconds
        offsetStr = `${s}${(abs / 1000).toFixed(1)}s`;
    }

    // Add source/satellite info
    if (timeHealth.source === 'gps' && timeHealth.metadata?.satellites != null) {
        // GPS with satellite count
        return `${offsetStr}, ${timeHealth.metadata.satellites} sats`;
    } else if (timeHealth.source && timeHealth.source !== 'unknown') {
        // Other sources (ntp, chrony)
        return `${offsetStr}, ${timeHealth.source}`;
    } else {
        // No source info
        return offsetStr;
    }
}

/**
 * Format GPS location information with three distinct states:
 *   1. null/undefined gpsLocation  → "Not detected"  (gpsd unreachable)
 *   2. gpsLocation exists, no lat  → "Searching…"    (gpsd running, no fix)
 *   3. gpsLocation with coords     → accuracy/sats   (has fix)
 *
 * @param {Object} gpsLocation - GPS location object with satellites, fix_mode, sep
 * @returns {string} Formatted GPS status string
 */
export function formatGPSLocation(gpsLocation) {
    if (!gpsLocation) {
        return 'Not detected';
    }

    const sats = gpsLocation.satellites || 0;
    const fixMode = gpsLocation.fix_mode || 0;

    if (gpsLocation.latitude == null) {
        if (fixMode <= 1) return `Searching\u2026 (${sats} sats)`;
        return `No fix (${sats} sats)`;
    }

    const fixTypes = ['No fix', 'No fix', '2D fix', '3D fix'];
    const fixType = fixTypes[Math.min(fixMode, 3)];

    // Add accuracy if available (prefer SEP - spherical error probable)
    let accuracy = '';
    if (gpsLocation.sep != null) {
        const accuracyFt = Math.round(gpsLocation.sep * 3.28084); // meters to feet
        accuracy = `\u00b1${accuracyFt}ft, `;
    } else if (gpsLocation.eph != null) {
        const accuracyFt = Math.round(gpsLocation.eph * 3.28084); // meters to feet
        accuracy = `\u00b1${accuracyFt}ft, `;
    }

    return `${accuracy}${sats} sats, ${fixType}`;
}

/**
 * Format operating location as compact coordinate string for the Telescope card.
 * @param {Object} status - SystemStatus object with location_latitude/longitude/altitude
 * @returns {string} e.g. "38.840°N, 104.821°W, 2743m"
 */
export function formatCompactLocation(status) {
    if (status?.location_latitude == null || status?.location_longitude == null) return '-';
    const lat = status.location_latitude;
    const lon = status.location_longitude;
    const ns = lat >= 0 ? 'N' : 'S';
    const ew = lon >= 0 ? 'E' : 'W';
    let s = `${Math.abs(lat).toFixed(3)}\u00b0${ns}, ${Math.abs(lon).toFixed(3)}\u00b0${ew}`;
    if (status.location_altitude != null) {
        s += `, ${Math.round(status.location_altitude)}m`;
    }
    return s;
}

/**
 * Return a Bootstrap text color class for a pointing accuracy value,
 * scaled relative to the telescope's short-axis FOV.
 *
 * Green: < 10% of FOV (pointing well within the field)
 * Yellow: < 33% of FOV (marginal)
 * Red: >= 33% of FOV (significant error)
 *
 * @param {number|null} deg - Pointing error in degrees
 * @param {number|null|undefined} fovShortDeg - Short-axis FOV in degrees
 * @returns {string} Bootstrap text class
 */
export function pointingAccuracyClass(deg, fovShortDeg) {
    if (deg == null) return 'text-muted';
    const green = fovShortDeg ? fovShortDeg * 0.10 : 0.05;
    const yellow = fovShortDeg ? fovShortDeg * 0.33 : 0.15;
    if (deg < green) return 'text-success';
    if (deg < yellow) return 'text-warning';
    return 'text-danger';
}
