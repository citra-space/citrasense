// WebSocket connection management for CitraSense

let ws = null;
let reconnectAttempts = 0;
let reconnectTimer = null;
let connectionTimer = null;
let heartbeatTimer = null;
const connectionTimeout = 5000;
const heartbeatInterval = 15000;
const maxReconnectDelay = 30000;

let onStatusUpdate = null;
let onLogMessage = null;
let onTasksUpdate = null;
let onPreviewImage = null;
let onToastMessage = null;
let onConnectionChange = null;

/**
 * Initialize WebSocket connection
 * @param {object} handlers - Event handlers {onStatus, onLog, onTasks, onPreview, onToast, onConnectionChange}
 */
export function connectWebSocket(handlers = {}) {
    onStatusUpdate = handlers.onStatus || null;
    onLogMessage = handlers.onLog || null;
    onTasksUpdate = handlers.onTasks || null;
    onPreviewImage = handlers.onPreview || null;
    onToastMessage = handlers.onToast || null;
    onConnectionChange = handlers.onConnectionChange || null;

    connect();

    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                reconnectAttempts = 0;
                connect();
            }
        }
    });
}

function connect() {
    if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }
    if (connectionTimer) {
        clearTimeout(connectionTimer);
        connectionTimer = null;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    try {
        if (ws && ws.readyState !== WebSocket.CLOSED) {
            ws.onclose = null;
            ws.onerror = null;
            ws.onmessage = null;
            ws.close();
        }

        ws = new WebSocket(wsUrl);

        connectionTimer = setTimeout(() => {
            if (ws && ws.readyState !== WebSocket.OPEN) {
                ws.close();
                scheduleReconnect();
            }
        }, connectionTimeout);

        ws.onopen = (event) => {
            if (event.target !== ws) return;
            if (connectionTimer) {
                clearTimeout(connectionTimer);
                connectionTimer = null;
            }
            reconnectAttempts = 0;
            startHeartbeat();
            notifyConnectionChange(true);
        };

        ws.onmessage = (event) => {
            if (event.target !== ws) return;
            let message;
            try {
                message = JSON.parse(event.data);
            } catch (e) {
                console.warn('Malformed WebSocket message:', e);
                return;
            }

            notifyMessageReceived();

            if (message.type === 'status' && onStatusUpdate) {
                onStatusUpdate(message.data);
            } else if (message.type === 'log' && onLogMessage) {
                onLogMessage(message.data);
            } else if (message.type === 'tasks' && onTasksUpdate) {
                onTasksUpdate(message.data);
            } else if (message.type === 'preview' && onPreviewImage) {
                onPreviewImage(message.data, message.source, message.sensor_id);
            } else if (message.type === 'preview_url' && onPreviewImage) {
                onPreviewImage(message.url, message.source, message.sensor_id);
            } else if (message.type === 'toast' && onToastMessage) {
                onToastMessage(message.data);
            }
        };

        ws.onclose = (event) => {
            if (event.target !== ws) return;
            console.log('WebSocket closed', event.code, event.reason);
            if (connectionTimer) {
                clearTimeout(connectionTimer);
                connectionTimer = null;
            }
            stopHeartbeat();
            ws = null;
            scheduleReconnect();
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    } catch (error) {
        console.error('Failed to create WebSocket:', error);
        if (connectionTimer) {
            clearTimeout(connectionTimer);
            connectionTimer = null;
        }
        ws = null;
        scheduleReconnect();
    }
}

function startHeartbeat() {
    stopHeartbeat();
    heartbeatTimer = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            try { ws.send('ping'); } catch (_) { /* onclose will fire */ }
        }
    }, heartbeatInterval);
}

function stopHeartbeat() {
    if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
    }
}

function scheduleReconnect() {
    let delay;
    if (reconnectAttempts === 0) {
        delay = 0;
    } else {
        const base = Math.min(1000 * Math.pow(2, reconnectAttempts - 1), maxReconnectDelay);
        delay = base + Math.random() * 1000;
    }

    const reconnectAt = Date.now() + delay;
    reconnectAttempts++;
    notifyConnectionChange(false, reconnectAt);

    reconnectTimer = setTimeout(connect, delay);
}

function notifyConnectionChange(connected, reconnectAt = 0) {
    if (onConnectionChange) {
        onConnectionChange(connected, reconnectAt);
    }
}

function notifyMessageReceived() {
    try {
        const store = window.Alpine?.store('citrasense');
        if (store) store.wsLastMessage = Date.now();
    } catch (_) { /* Alpine not ready yet */ }
}
