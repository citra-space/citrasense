/**
 * CitraScope Formatter Utilities
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
 * Format GPS location information
 * @param {Object} gpsLocation - GPS location object with satellites, fix_mode, sep
 * @returns {string} Formatted GPS location (e.g., "±102ft, 6 sats, 3D fix")
 */
export function formatGPSLocation(gpsLocation) {
    if (!gpsLocation || gpsLocation.latitude == null) {
        return 'GPS unavailable';
    }

    const sats = gpsLocation.satellites || 0;
    const fixMode = gpsLocation.fix_mode || 0;
    const fixTypes = ['No fix', 'No fix', '2D fix', '3D fix'];
    const fixType = fixTypes[Math.min(fixMode, 3)];

    // Add accuracy if available (prefer SEP - spherical error probable)
    let accuracy = '';
    if (gpsLocation.sep != null) {
        const accuracyFt = Math.round(gpsLocation.sep * 3.28084); // meters to feet
        accuracy = `±${accuracyFt}ft, `;
    } else if (gpsLocation.eph != null) {
        const accuracyFt = Math.round(gpsLocation.eph * 3.28084); // meters to feet
        accuracy = `±${accuracyFt}ft, `;
    }

    return `${accuracy}${sats} sats, ${fixType}`;
}

/**
 * Compute slippy-map tile coordinates for a given lat/lon.
 * Returns { url, px, py } where url is the tile image URL and px/py are the
 * pixel offsets of the point within that tile.
 * @param {number} lat - Latitude in degrees
 * @param {number} lon - Longitude in degrees
 * @param {number} [zoom=12] - Zoom level
 * @param {number} [tileSize=256] - Tile pixel size
 * @returns {{url: string, px: number, py: number}}
 */
export function tileCoords(lat, lon, zoom = 12, tileSize = 256) {
    lat = Math.max(-85.0511, Math.min(85.0511, lat));
    lon = ((lon % 360) + 540) % 360 - 180;
    const n = Math.pow(2, zoom);
    const tx = Math.floor((lon + 180) / 360 * n);
    const r = lat * Math.PI / 180;
    const ty = Math.floor((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2 * n);
    const xFrac = (lon + 180) / 360 * n;
    const yFrac = (1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2 * n;
    const px = Math.round((xFrac - tx) * tileSize);
    const py = Math.round((yFrac - ty) * tileSize);
    const url = `https://a.basemaps.cartocdn.com/dark_all/${zoom}/${tx}/${ty}.png`;
    return { url, px, py };
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
