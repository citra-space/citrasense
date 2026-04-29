// CitraSense Dashboard - Main Application (Alpine.js)
import { connectWebSocket } from './websocket.js';
import { initConfig, initFilterConfig, setupAutofocusButton, showToast } from './config.js';
import { getTasks, getLogs } from './api.js';

// Store and components are registered in store-init.js (loaded before Alpine)

// --- Store update handlers (replace DOM manipulation) ---
function updateStoreFromStatus(status) {
    const store = Alpine.store('citrasense');
    store.status = status;

    // Sync sensor list from status.sensors dict (spread all enriched data)
    if (status.sensors && typeof status.sensors === 'object') {
        const list = Object.entries(status.sensors).map(([id, info]) => ({
            id,
            ...info,
        }));
        store.sensors = list;
    }

    // Per-sensor map of actively-executing tasks. Used by taskRow.isActive
    // (and the cancel-button gate) so every sensor's in-flight task is
    // flagged. There is intentionally no scalar ``current_task`` fallback —
    // the site-level scalar was removed because "first non-None wins" was
    // ambiguous in multi-sensor deployments.
    const taskIdMap = status.current_task_ids && typeof status.current_task_ids === 'object'
        ? status.current_task_ids
        : {};
    store.currentTaskIds = taskIdMap;
    store.activeTaskIdSet = new Set(Object.values(taskIdMap));

    // Per-sensor task preview: push the newest annotated image into
    // ``previewDataUrls[sensor_id]`` so each sensor card shows its own
    // latest result.  There is no singular ``previewDataUrl`` — the
    // fullscreen modal reads ``previewDataUrls[activePreviewSensorId]``
    // set by the card's click handler.
    if (status.sensors) {
        if (!store._lastTaskImageUrlBySensor) store._lastTaskImageUrlBySensor = {};
        const next = { ...store.previewDataUrls };
        let changed = false;
        for (const [sid, info] of Object.entries(status.sensors)) {
            if (store.isLoopingFor(sid)) continue;
            const url = info?.latest_task_image_url;
            if (url && url !== store._lastTaskImageUrlBySensor[sid]) {
                store._lastTaskImageUrlBySensor[sid] = url;
                next[sid] = url;
                changed = true;
            }
        }
        if (changed) store.previewDataUrls = next;
    }
}

function updateStoreFromTasks(tasks) {
    const store = Alpine.store('citrasense');
    const sorted = [...(tasks || [])].sort((a, b) => new Date(a.start_time) - new Date(b.start_time));
    store.tasks = sorted;
}

const MAX_CLIENT_LOGS = 500;
let logSeq = 0;

function appendLogToStore(log) {
    const store = Alpine.store('citrasense');
    log._id = ++logSeq;
    const logs = [...store.logs, log];
    store.logs = logs.length > MAX_CLIENT_LOGS ? logs.slice(-MAX_CLIENT_LOGS) : logs;
    store.latestLog = log;
}

function updatePreviewFromPush(dataUrl, source, sensorId) {
    const store = Alpine.store('citrasense');
    if (!sensorId) {
        console.warn('updatePreviewFromPush called without sensorId; dropping preview push');
        return;
    }
    if (store.isLoopingFor(sensorId)) return;
    store.previewDataUrls = { ...store.previewDataUrls, [sensorId]: dataUrl };
    store.previewSource = source || '';
}

function handleBackendToast(data) {
    showToast(data.message, data.toast_type || 'info', { id: data.id });
}

function handleRadarDetection(sensorId, det) {
    if (!sensorId || !det) return;
    const store = Alpine.store('citrasense');
    store.appendRadarDetection(sensorId, det);
}

function updateStoreFromConnection(connected, reconnectAt = 0) {
    const store = Alpine.store('citrasense');
    store.wsConnected = connected;
    store.wsReconnecting = !connected && reconnectAt > 0;
    store.wsReconnectAt = reconnectAt;
    if (connected) {
        store.wsLastMessage = Date.now();
    }
}

// --- 1Hz heartbeat: bumps $store.citrasense.now ---
//
// Reactive Alpine getters that depend on wall-clock time read
// `$store.citrasense.now` so they re-render once per second without each
// component running its own setInterval.  The Scheduled Tasks countdown,
// for instance, is just a getter on `taskRow` that reads `now` and the
// task's start/stop times — so the heartbeat no longer computes a
// site-wide countdown string (that field was never read by any
// template, and was ambiguous across sensors anyway).
let countdownInterval = null;

function startCountdownUpdater() {
    if (countdownInterval) return;
    countdownInterval = setInterval(() => {
        const store = Alpine.store('citrasense');
        store.now = Date.now();
    }, 1000);
}

// --- Version checking ---
async function fetchVersion() {
    try {
        const response = await fetch('/api/version');
        const data = await response.json();
        const store = Alpine.store('citrasense');
        store.versionData = data;
        if (data.version) {
            if (data.version === 'development') {
                store.version = 'dev';
            } else if (data.git_branch && data.git_branch !== 'main' && data.install_type !== 'pypi') {
                store.version = `v${data.version} (${data.git_branch})`;
            } else {
                store.version = 'v' + data.version;
            }
        } else {
            store.version = 'v?';
        }
    } catch (error) {
        console.error('Error fetching version:', error);
        Alpine.store('citrasense').version = 'v?';
    }
}

// --- Navigation (History API, path-based) ---
//
// The backend serves dashboard.html for any non-/api/*, non-/static/*, non-/ws
// GET via the SPA catchall in citrasense/web/routes/spa_fallback.py, so the
// browser URL can be a real path (/monitoring, /analysis, /config,
// /sensors/<id>) rather than a hash.  This function owns the mapping in
// both directions: reading location.pathname -> {section, sensorId}, and
// (via navigateTo on the store) pushing state when the user clicks an
// in-app link.  ``popstate`` fires for browser back/forward.
const STATIC_PATH_SECTIONS = {
    '/': 'monitoring',
    '/monitoring': 'monitoring',
    '/analysis': 'analysis',
    '/config': 'config',
};

// Browser tab titles per static section.  Sensor routes are titled
// dynamically (placeholder = sensor id, upgraded to the real name once
// the WebSocket delivers the sensor list — see store-init.js).
const SECTION_TITLES = {
    monitoring: 'Monitoring',
    analysis: 'Analysis',
    config: 'Config',
};

// Allow hyphen, dot, underscore, and URL-encoded chars in sensor IDs.
// SensorConfig.id validation is stricter than this, but the router just
// needs to capture whatever the backend considers valid.
const SENSOR_PATH_RE = /^\/sensors\/([^/?#]+)$/;

function parsePath(pathname) {
    const p = pathname.replace(/\/+$/, '') || '/';
    if (STATIC_PATH_SECTIONS[p]) {
        return { section: STATIC_PATH_SECTIONS[p], sensorId: null };
    }
    const m = p.match(SENSOR_PATH_RE);
    if (m) {
        try {
            return { section: 'sensor', sensorId: decodeURIComponent(m[1]) };
        } catch {
            return { section: 'sensor', sensorId: m[1] };
        }
    }
    return null;
}

function applyRoute() {
    const parsed = parsePath(window.location.pathname);
    const store = Alpine.store('citrasense');
    if (!parsed) {
        // Unknown path: rewrite silently to /monitoring rather than force a
        // refresh (the SPA shell already rendered).
        history.replaceState({}, '', '/monitoring');
        store.currentSection = 'monitoring';
        store.currentSensorId = null;
        document.title = `${SECTION_TITLES.monitoring} | CitraSense`;
        return;
    }
    store.currentSection = parsed.section;
    store.currentSensorId = parsed.sensorId;
    if (parsed.section === 'sensor') {
        // Placeholder title: show the raw sensor id immediately so the
        // tab is distinguishable before the WebSocket delivers sensor
        // data.  An Alpine.effect in store-init.js upgrades this to the
        // human-readable sensor name once ``currentSensor`` resolves.
        document.title = `${parsed.sensorId} | CitraSense`;
    } else {
        document.title = `${SECTION_TITLES[parsed.section] || 'Dashboard'} | CitraSense`;
    }
    if (parsed.section === 'config') initFilterConfig();
    if (parsed.section === 'analysis' && window.loadAnalysisData) {
        window.loadAnalysisData();
    }
}

function initNavigation() {
    // Expose applyRoute so store.navigateTo can re-run routing after a
    // pushState without re-importing this module.
    window.__spaApplyRoute = applyRoute;
    window.addEventListener('popstate', applyRoute);
    applyRoute();
}

// Config module will need to update store.config when loaded - we'll handle in config.js

// Camera control and version modal moved to Alpine store methods

// Camera capture moved to Alpine store method

// --- Initialize ---
// Expose showToast globally so inline Alpine x-data handlers in Jinja
// templates can fire toasts without a dynamic import per click.  The
// established pattern inside ES modules is still to import it directly
// from ./config.js; only template-scoped code should reach for the
// window-level hook.
window.showToast = showToast;

document.addEventListener('DOMContentLoaded', async () => {
    initNavigation();
    await initConfig();
    await initFilterConfig();
    setupAutofocusButton();
    fetchVersion();
    Alpine.store('citrasense').checkForUpdates();
    setInterval(() => Alpine.store('citrasense').checkForUpdates(), 3600000);

    connectWebSocket({
        onStatus: updateStoreFromStatus,
        onLog: appendLogToStore,
        onTasks: updateStoreFromTasks,
        onPreview: updatePreviewFromPush,
        onToast: handleBackendToast,
        onRadarDetection: handleRadarDetection,
        onConnectionChange: updateStoreFromConnection
    });

    const tasksData = await getTasks();
    const tasks = Array.isArray(tasksData) ? tasksData : (tasksData?.tasks || []);
    updateStoreFromTasks(tasks);

    const logsData = await getLogs(100);
    const store = Alpine.store('citrasense');
    store.logs = (logsData.logs || []).map(log => ({ ...log, _id: ++logSeq }));
    if (store.logs.length > 0) {
        store.latestLog = store.logs[store.logs.length - 1];
    }

    startCountdownUpdater();

    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    for (const el of tooltipTriggerList) {
        new bootstrap.Tooltip(el);
    }

    // Toggle switches now use Alpine @change directives in templates
});
