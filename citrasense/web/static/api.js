// API client for CitraSense backend

/**
 * Wrapper around fetch that handles JSON parsing and error responses
 * @param {string} url - The URL to fetch
 * @param {object} options - Fetch options (method, headers, body, etc.)
 * @returns {Promise<{ok: boolean, status: number, data: any}>}
 */
export async function fetchJSON(url, options = {}) {
    try {
        const response = await fetch(url, options);
        const data = await response.json();

        return {
            ok: response.ok,
            status: response.status,
            data: data
        };
    } catch (error) {
        console.error(`API request failed: ${url}`, error);
        throw error;
    }
}

/**
 * Get current configuration
 */
export async function getConfig() {
    const result = await fetchJSON('/api/config');
    return result.data;
}

/**
 * Save configuration
 */
export async function saveConfig(config) {
    return await fetchJSON('/api/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(config)
    });
}

/**
 * Get configuration status
 */
export async function getConfigStatus() {
    const result = await fetchJSON('/api/config/status');
    return result.data;
}

/**
 * Get available hardware adapters
 */
export async function getHardwareAdapters() {
    const result = await fetchJSON('/api/hardware-adapters');
    return result.data;
}

/**
 * Get available processors
 */
export async function getProcessors() {
    try {
        const result = await fetchJSON('/api/processors');
        return result.data;
    } catch (error) {
        console.error('Error fetching processors:', error);
        return [];
    }
}

/**
 * Get hardware adapter schema
 */
export async function getAdapterSchema(adapterName) {
    const result = await fetchJSON(`/api/hardware-adapters/${adapterName}/schema`);
    return result.data;
}

/**
 * Get task queue
 */
export async function getTasks() {
    const result = await fetchJSON('/api/tasks');
    return result.data;
}

/**
 * Reconnect hardware adapter using current settings
 */
export async function reconnectHardware() {
    return await fetchJSON('/api/hardware/reconnect', { method: 'POST' });
}

/**
 * Get recent logs
 */
export async function getLogs(limit = 100) {
    const result = await fetchJSON(`/api/logs?limit=${limit}`);
    return result.data;
}
