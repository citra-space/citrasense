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
    // flagged, not just the first one ``current_task`` happens to surface.
    const taskIdMap = status.current_task_ids && typeof status.current_task_ids === 'object'
        ? status.current_task_ids
        : {};
    store.currentTaskIds = taskIdMap;
    store.activeTaskIdSet = new Set(Object.values(taskIdMap));

    if (status.current_task && status.current_task !== 'None') {
        store.isTaskActive = true;
        store.currentTaskId = status.current_task;
        store.nextTaskStartTime = null;
    } else {
        store.isTaskActive = store.activeTaskIdSet.size > 0;
        store.currentTaskId = null;
    }

    if (!store.isTaskActive && store.tasks.length > 0) {
        const sorted = [...store.tasks].sort((a, b) => new Date(a.start_time) - new Date(b.start_time));
        store.nextTaskStartTime = sorted[0].start_time;
    }

    // Per-sensor task preview: push the newest annotated image into
    // ``previewDataUrls[sensor_id]`` so each sensor card shows its own
    // latest result. We intentionally no longer touch the site-wide
    // ``store.previewDataUrl`` slot here — that slot is reserved for the
    // full-screen modal and gets set explicitly by the card's click
    // handler. Previously this line would flip ``previewDataUrl`` between
    // sensors every status tick, corrupting the modal and the "card
    // fallback" display.
    if (status.sensors && !store.isLooping) {
        if (!store._lastTaskImageUrlBySensor) store._lastTaskImageUrlBySensor = {};
        const next = { ...store.previewDataUrls };
        let changed = false;
        for (const [sid, info] of Object.entries(status.sensors)) {
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

    if (!store.isTaskActive && sorted.length > 0) {
        store.nextTaskStartTime = sorted[0].start_time;
    } else if (store.isTaskActive) {
        store.nextTaskStartTime = null;
    }
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
    if (store.isLooping) return;
    if (sensorId) {
        store.previewDataUrls = { ...store.previewDataUrls, [sensorId]: dataUrl };
    }
    store.previewDataUrl = dataUrl;
    store.previewSource = source || '';
}

function handleBackendToast(data) {
    showToast(data.message, data.toast_type || 'info', { id: data.id });
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

// --- 1Hz heartbeat: bumps store.now and refreshes store.countdown ---
//
// Reactive Alpine getters that depend on wall-clock time read
// `$store.citrasense.now` so they re-render once per second without each
// component running its own setInterval.  The Scheduled Tasks countdown,
// for instance, is just a getter on `taskRow` that reads `now` and the
// task's start/stop times.
let countdownInterval = null;

function startCountdownUpdater() {
    if (countdownInterval) return;
    countdownInterval = setInterval(() => {
        const store = Alpine.store('citrasense');
        store.now = Date.now();

        if (!store.nextTaskStartTime || store.isTaskActive) {
            store.countdown = '';
            return;
        }
        const timeUntil = new Date(store.nextTaskStartTime) - store.now;
        store.countdown = timeUntil > 0 ? store.formatCountdown(timeUntil) : 'Starting soon...';
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

// --- Navigation (Alpine-driven in Phase 3, keep hash sync for now) ---
function navigateToSection(section) {
    const store = Alpine.store('citrasense');
    store.currentSection = section;
    window.location.hash = section;
    if (section === 'config') {
        initFilterConfig();
    }
}

function initNavigation() {
    window.addEventListener('hashchange', () => {
        const hash = window.location.hash.substring(1);
        if (hash && (hash === 'monitoring' || hash === 'config' || hash === 'analysis')) {
            const store = Alpine.store('citrasense');
            store.currentSection = hash;
            if (hash === 'config') initFilterConfig();
            if (hash === 'analysis' && window.loadAnalysisData) window.loadAnalysisData();
        }
    });

    const hash = window.location.hash.substring(1);
    if (hash && (hash === 'monitoring' || hash === 'config' || hash === 'analysis')) {
        navigateToSection(hash);
    } else {
        navigateToSection('monitoring');
    }
}

// Config module will need to update store.config when loaded - we'll handle in config.js

// Camera control and version modal moved to Alpine store methods

// Camera capture moved to Alpine store method

// --- Initialize ---
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
