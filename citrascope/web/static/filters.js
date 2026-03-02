/**
 * Filter color utilities for visual wavelength representation.
 *
 * Maps standard astronomical filters to their representative colors
 * based on spectral wavelength. Colors are darker variants suitable
 * for badge backgrounds with white text.
 */

/**
 * Standard filter name to color mappings.
 * Colors chosen to be dark enough for white text readability.
 */
export const FILTER_COLORS = {
    // Johnson-Cousins UBVRI
    'U': '#33005D',      // Ultraviolet (365nm)
    'B': '#002A6E',      // Blue (445nm)
    'V': '#005500',      // Visual/Green (551nm)
    'R': '#6E0000',      // Red (658nm)
    'I': '#550000',      // Near-infrared (806nm)

    // Sloan ugriz (bare letters + sloan_ prefixed aliases used by Citra API)
    'u': '#3B0066',      // Ultraviolet (354nm)
    'sloan_u': '#3B0066',
    'g': '#00442A',      // Green (477nm)
    'sloan_g': '#00442A',
    'r': '#6E2200',      // Orange-Red (623nm)
    'sloan_r': '#6E2200',
    'i': '#5D0000',      // Near-infrared (763nm)
    'sloan_i': '#5D0000',
    'z': '#440000',      // Infrared (913nm)
    'sloan_z': '#440000',

    // RGB filters
    'Red': '#770000',
    'Green': '#006600',
    'Blue': '#003380',
    'Clear': '#4C4C4C',  // Gray for broadband clear
    'Luminance': '#4C4C4C',  // Gray for luminance

    // Narrowband emission line filters
    'Ha': '#6E002A',     // H-alpha (656.3nm) - characteristic deep red
    'Hb': '#004C66',     // H-beta (486.1nm) - cyan
    'OIII': '#005D44',   // OIII (500.7nm) - teal
    'SII': '#6E112A',    // SII (672.4nm) - pink-red
};

/**
 * Get color for a filter based on name.
 *
 * @param {string} name - Filter name (e.g., "Ha", "Red", "V")
 * @returns {string} Hex color string suitable for dark badge backgrounds
 */
export function getFilterColor(name) {
    // Check known filters first
    if (FILTER_COLORS[name]) {
        return FILTER_COLORS[name];
    }
    // Default to gray if no match
    return '#777777';
}
