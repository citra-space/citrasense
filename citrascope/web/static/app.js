// CitraScope Dashboard - Main Application (Alpine.js)
import { connectWebSocket } from './websocket.js';
import { initConfig, initFilterConfig, setupAutofocusButton, showToast } from './config.js';
import { getTasks, getLogs } from './api.js';

// Store and components are registered in store-init.js (loaded before Alpine)

// --- Store update handlers (replace DOM manipulation) ---
function updateStoreFromStatus(status) {
    const store = Alpine.store('citrascope');
    store.status = status;

    if (status.current_task && status.current_task !== 'None') {
        store.isTaskActive = true;
        store.currentTaskId = status.current_task;
        store.nextTaskStartTime = null;
    } else {
        store.isTaskActive = false;
        store.currentTaskId = null;
    }

    // Set nextTaskStartTime from tasks if we have them and no active task
    if (!store.isTaskActive && store.tasks.length > 0) {
        const sorted = [...store.tasks].sort((a, b) => new Date(a.start_time) - new Date(b.start_time));
        store.nextTaskStartTime = sorted[0].start_time;
    }

    // Update Optics pane with latest annotated task image (don't interrupt live loop)
    if (status.latest_task_image_url && status.latest_task_image_url !== store._lastTaskImageUrl) {
        store._lastTaskImageUrl = status.latest_task_image_url;
        if (!store.isLooping) {
            store.previewDataUrl = status.latest_task_image_url;
        }
    }
}

function updateStoreFromTasks(tasks) {
    const store = Alpine.store('citrascope');
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
    const store = Alpine.store('citrascope');
    log._id = ++logSeq;
    const logs = [...store.logs, log];
    store.logs = logs.length > MAX_CLIENT_LOGS ? logs.slice(-MAX_CLIENT_LOGS) : logs;
    store.latestLog = log;
}

function updatePreviewFromPush(dataUrl, source) {
    const store = Alpine.store('citrascope');
    if (store.isLooping) return;
    store.previewDataUrl = dataUrl;
    store.previewSource = source || '';
}

function handleBackendToast(data) {
    showToast(data.message, data.toast_type || 'info', { id: data.id });
}

function updateStoreFromConnection(connected, reconnectInfo = '') {
    const store = Alpine.store('citrascope');
    store.wsConnected = connected;
    store.wsReconnecting = !!reconnectInfo;
}

// --- Countdown tick (updates store.countdown) ---
let countdownInterval = null;

function startCountdownUpdater() {
    if (countdownInterval) return;
    countdownInterval = setInterval(() => {
        const store = Alpine.store('citrascope');
        if (!store.nextTaskStartTime || store.isTaskActive) {
            store.countdown = '';
            return;
        }
        const now = new Date();
        const timeUntil = new Date(store.nextTaskStartTime) - now;
        store.countdown = timeUntil > 0 ? store.formatCountdown(timeUntil) : 'Starting soon...';
    }, 1000);
}

// --- Version checking ---
async function fetchVersion() {
    try {
        const response = await fetch('/api/version');
        const data = await response.json();
        const store = Alpine.store('citrascope');
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
        Alpine.store('citrascope').version = 'v?';
    }
}

// --- Navigation (Alpine-driven in Phase 3, keep hash sync for now) ---
function navigateToSection(section) {
    const store = Alpine.store('citrascope');
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
            const store = Alpine.store('citrascope');
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
    Alpine.store('citrascope').checkForUpdates();
    setInterval(() => Alpine.store('citrascope').checkForUpdates(), 3600000);

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
    const store = Alpine.store('citrascope');
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
