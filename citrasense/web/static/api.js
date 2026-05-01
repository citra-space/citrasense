// Unified API client for CitraSense backend
//
// Every endpoint the frontend calls is wrapped here with a uniform return
// shape: { ok, status, data, error }.  Callers should never use raw fetch().

/**
 * @typedef {Object} ApiResult
 * @property {boolean} ok - True if HTTP status is 2xx.
 * @property {number} status - HTTP status code (0 on network error).
 * @property {any} data - Parsed JSON body.
 * @property {string|null} error - Error message on failure, null on success.
 */

/**
 * Core fetch wrapper — handles JSON parsing, error extraction, and network failures.
 * @param {string} url
 * @param {RequestInit} [options]
 * @returns {Promise<ApiResult>}
 */
export async function fetchJSON(url, options = {}) {
    try {
        const response = await fetch(url, options);
        let data = null;
        const ct = response.headers.get('content-type') || '';
        if (ct.includes('application/json')) {
            data = await response.json();
        } else {
            const text = await response.text();
            try { data = JSON.parse(text); } catch { data = text; }
        }
        return {
            ok: response.ok,
            status: response.status,
            data,
            error: response.ok ? null : (data?.error || data?.detail || `HTTP ${response.status}`),
        };
    } catch (error) {
        return {
            ok: false,
            status: 0,
            data: null,
            error: error.message || 'Network error',
        };
    }
}

function sensorUrl(sensorId, path = '') {
    return `/api/sensors/${encodeURIComponent(sensorId)}${path}`;
}

function jsonPost(url, body) {
    return fetchJSON(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
}

function jsonPatch(url, body) {
    return fetchJSON(url, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
}

function post(url) {
    return fetchJSON(url, { method: 'POST' });
}

function del(url) {
    return fetchJSON(url, { method: 'DELETE' });
}

// ─── Config ───────────────────────────────────────────────────────────────────

export async function getConfig() {
    const r = await fetchJSON('/api/config');
    return r.data;
}

export async function saveConfig(config) {
    return jsonPost('/api/config', config);
}

export async function getConfigStatus() {
    const r = await fetchJSON('/api/config/status');
    return r.data;
}

export async function addSensor(sensorData) {
    return jsonPost('/api/config/sensors', sensorData);
}

export async function deleteSensor(sensorId) {
    return del('/api/config/sensors/' + encodeURIComponent(sensorId));
}

// ─── Hardware ─────────────────────────────────────────────────────────────────

export async function getHardwareAdapters() {
    const r = await fetchJSON('/api/hardware-adapters');
    return r.data;
}

export async function getAdapterSchema(adapterName, settingsParam = '') {
    const r = await fetchJSON(`/api/hardware-adapters/${encodeURIComponent(adapterName)}/schema${settingsParam}`);
    return r.data;
}

export async function reconnectSensor(sensorId) {
    return post(sensorUrl(sensorId, '/reconnect'));
}

export async function connectSensor(sensorId) {
    return post(sensorUrl(sensorId, '/connect'));
}

export async function disconnectSensor(sensorId) {
    return post(sensorUrl(sensorId, '/disconnect'));
}

export async function scanHardware(adapterName, currentSettings = {}) {
    return jsonPost('/api/hardware/scan', { adapter_name: adapterName, current_settings: currentSettings });
}

// ─── Sensor Types ─────────────────────────────────────────────────────────────

export async function getSensorTypes() {
    const r = await fetchJSON('/api/sensor-types');
    return r.data;
}

export async function getSensorTypeSchema(sensorType, currentSettings = null) {
    let url = `/api/sensor-types/${encodeURIComponent(sensorType)}/schema`;
    if (currentSettings && Object.keys(currentSettings).length > 0) {
        url += `?current_settings=${encodeURIComponent(JSON.stringify(currentSettings))}`;
    }
    const r = await fetchJSON(url);
    return r.data;
}

// ─── Processors ───────────────────────────────────────────────────────────────

export async function getProcessors(sensorId) {
    if (!sensorId) {
        console.error('getProcessors called without sensorId');
        return [];
    }
    try {
        const qs = `?sensor_id=${encodeURIComponent(sensorId)}`;
        const r = await fetchJSON(`/api/processors${qs}`);
        return r.data;
    } catch {
        return [];
    }
}

// ─── Tasks ────────────────────────────────────────────────────────────────────

export async function getTasks() {
    const r = await fetchJSON('/api/tasks');
    return r.data;
}

export async function cancelTask(taskId) {
    return post('/api/tasks/' + encodeURIComponent(taskId) + '/cancel');
}

export async function pauseTasks(sensorId) {
    return post(sensorUrl(sensorId, '/tasks/pause'));
}

export async function resumeTasks(sensorId) {
    return post(sensorUrl(sensorId, '/tasks/resume'));
}

// ─── Observing Session / Self-Tasking / Scheduling ────────────────────────────

export async function toggleObservingSession(sensorId, enabled) {
    return jsonPatch(sensorUrl(sensorId, '/observing-session'), { enabled });
}

export async function toggleSelfTasking(sensorId, enabled) {
    return jsonPatch(sensorUrl(sensorId, '/self-tasking'), { enabled });
}

export async function toggleScheduling(sensorId, enabled) {
    return jsonPatch(sensorUrl(sensorId, '/scheduling'), { enabled });
}

export async function requestSelfTaskNow(sensorId) {
    return post(sensorUrl(sensorId, '/self-tasking/request-now'));
}

// ─── Camera ───────────────────────────────────────────────────────────────────

export async function captureImage(sensorId, duration) {
    return jsonPost(sensorUrl(sensorId, '/camera/capture'), { duration });
}

export async function capturePreview(sensorId, duration, flipHorizontal = false) {
    return jsonPost(sensorUrl(sensorId, '/camera/preview'), {
        duration,
        flip_horizontal: flipHorizontal,
    });
}

// ─── Allsky ───────────────────────────────────────────────────────────────────

export async function setAllskyStreaming(sensorId, enabled) {
    return jsonPost(sensorUrl(sensorId, '/allsky/streaming'), { enabled });
}

// ─── Filters ──────────────────────────────────────────────────────────────────

export async function getFilters(sensorId) {
    return fetchJSON(sensorUrl(sensorId, '/filters'));
}

export async function updateFiltersBatch(sensorId, updates) {
    return jsonPost(sensorUrl(sensorId, '/filters/batch'), updates);
}

export async function syncFilters(sensorId) {
    return post(sensorUrl(sensorId, '/filters/sync'));
}

export async function setFilter(sensorId, position) {
    return jsonPost(sensorUrl(sensorId, '/filter/set'), { position: parseInt(position) });
}

// ─── Focuser ──────────────────────────────────────────────────────────────────

export async function moveFocuser(sensorId, position) {
    return jsonPost(sensorUrl(sensorId, '/focuser/move'), { position: parseInt(position) });
}

export async function moveFocuserRelative(sensorId, steps) {
    return jsonPost(sensorUrl(sensorId, '/focuser/move'), { relative: parseInt(steps) });
}

export async function abortFocuser(sensorId) {
    return post(sensorUrl(sensorId, '/focuser/abort'));
}

// ─── Mount ────────────────────────────────────────────────────────────────────

export async function mountMove(sensorId, action, direction) {
    return jsonPost(sensorUrl(sensorId, '/mount/move'), { action, direction });
}

export async function mountGoto(sensorId, ra, dec) {
    return jsonPost(sensorUrl(sensorId, '/mount/goto'), { ra, dec });
}

export async function mountHome(sensorId) {
    return post(sensorUrl(sensorId, '/mount/home'));
}

export async function mountUnwind(sensorId) {
    return post(sensorUrl(sensorId, '/mount/unwind'));
}

export async function mountTracking(sensorId, enabled) {
    return jsonPost(sensorUrl(sensorId, '/mount/tracking'), { enabled });
}

// ─── Autofocus ────────────────────────────────────────────────────────────────

export async function getAutofocusPresets(sensorId) {
    return fetchJSON(sensorUrl(sensorId, '/autofocus/presets'));
}

export async function triggerAutofocus(sensorId, options = {}) {
    return jsonPost(sensorUrl(sensorId, '/autofocus'), options);
}

export async function cancelAutofocus(sensorId) {
    return post(sensorUrl(sensorId, '/autofocus/cancel'));
}

// ─── Alignment ────────────────────────────────────────────────────────────────

export async function triggerAlignment(sensorId) {
    return post(sensorUrl(sensorId, '/alignment'));
}

export async function cancelAlignment(sensorId) {
    return post(sensorUrl(sensorId, '/alignment/cancel'));
}

export async function syncTelescope(sensorId, ra, dec) {
    return jsonPost(sensorUrl(sensorId, '/sync'), { ra, dec });
}

// ─── Pointing Model ──────────────────────────────────────────────────────────

export async function calibratePointingModel(sensorId) {
    return post(sensorUrl(sensorId, '/pointing-model/calibrate'));
}

export async function resetPointingModel(sensorId) {
    return post(sensorUrl(sensorId, '/pointing-model/reset'));
}

export async function cancelPointingModelCalibration(sensorId) {
    return post(sensorUrl(sensorId, '/pointing-model/calibrate/cancel'));
}

// ─── Safety ───────────────────────────────────────────────────────────────────

export async function emergencyStop() {
    return post('/api/emergency-stop');
}

export async function clearOperatorStop() {
    return post('/api/safety/operator-stop/clear');
}

export async function resetCableWrap(sensorId) {
    return post(sensorUrl(sensorId, '/safety/cable-wrap/reset'));
}

// ─── Calibration ──────────────────────────────────────────────────────────────

export async function getCalibrationStatus(sensorId) {
    return fetchJSON(sensorUrl(sensorId, '/calibration/status'));
}

export async function startCalibrationCapture(sensorId, options = {}) {
    return jsonPost(sensorUrl(sensorId, '/calibration/capture'), options);
}

export async function cancelCalibrationCapture(sensorId) {
    return post(sensorUrl(sensorId, '/calibration/cancel'));
}

export async function startCalibrationSuite(sensorId, options = {}) {
    return jsonPost(sensorUrl(sensorId, '/calibration/capture-suite'), options);
}

export async function uploadCalibration(sensorId, queryString, file) {
    return fetchJSON(sensorUrl(sensorId, `/calibration/upload?${queryString}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/fits' },
        body: file,
    });
}

export async function deleteMasterCalibration(sensorId, options = {}) {
    return fetchJSON(sensorUrl(sensorId, '/calibration/master'), {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(options),
    });
}

// ─── Radar ────────────────────────────────────────────────────────────────────

export async function getRadarDetections(sensorId, since) {
    return fetchJSON(sensorUrl(sensorId, `/radar/detections?since=${since}`));
}

// ─── Version ──────────────────────────────────────────────────────────────────

export async function getVersion() {
    return fetchJSON('/api/version');
}

export async function checkForUpdates() {
    return fetchJSON('/api/version/check-updates');
}

// ─── Twilight ─────────────────────────────────────────────────────────────────

export async function getTwilight() {
    return fetchJSON('/api/twilight');
}

// ─── Logs ─────────────────────────────────────────────────────────────────────

export async function getLogs(limit = 100) {
    const r = await fetchJSON(`/api/logs?limit=${limit}`);
    return r.data;
}

// ─── Analysis ─────────────────────────────────────────────────────────────────

export async function getAnalysisStats(params) {
    return fetchJSON('/api/analysis/stats?' + params.toString());
}

export async function getAnalysisTasks(params) {
    return fetchJSON('/api/analysis/tasks?' + params.toString());
}

export async function getAnalysisTask(taskId) {
    return fetchJSON('/api/analysis/tasks/' + encodeURIComponent(taskId));
}

export async function reprocessTask(taskId, options = {}) {
    return jsonPost('/api/analysis/tasks/' + encodeURIComponent(taskId) + '/reprocess', options);
}

export async function reprocessTaskUpload(taskId) {
    return post('/api/analysis/tasks/' + encodeURIComponent(taskId) + '/reprocess/upload');
}

export async function reprocessBatch(options) {
    return jsonPost('/api/analysis/reprocess-batch', options);
}

// ─── Jobs ─────────────────────────────────────────────────────────────────────

export async function getJob(jobId) {
    return fetchJSON('/api/jobs/' + encodeURIComponent(jobId));
}

export async function cancelJob(jobId) {
    return post('/api/jobs/' + encodeURIComponent(jobId) + '/cancel');
}

// ─── Autotune ─────────────────────────────────────────────────────────────────

export async function runAutotune(sensorId, url, options = {}) {
    return jsonPost(url, options);
}

export async function applyAutotune(sensorId, settings) {
    return jsonPost(sensorUrl(sensorId, '/autotune/apply'), settings);
}
