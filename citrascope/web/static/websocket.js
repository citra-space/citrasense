// WebSocket connection management for CitraScope

let ws = null;
let reconnectAttempts = 0;
let reconnectTimer = null;
let connectionTimer = null;
const reconnectDelay = 5000; // Fixed 5 second delay between reconnect attempts
const connectionTimeout = 5000; // 5 second timeout for connection attempts

// Callbacks for handling messages
let onStatusUpdate = null;
let onLogMessage = null;
let onTasksUpdate = null;
let onPreviewImage = null;
let onConnectionChange = null;

/**
 * Initialize WebSocket connection
 * @param {object} handlers - Event handlers {onStatus, onLog, onTasks, onConnectionChange}
 */
export function connectWebSocket(handlers = {}) {
    onStatusUpdate = handlers.onStatus || null;
    onLogMessage = handlers.onLog || null;
    onTasksUpdate = handlers.onTasks || null;
    onPreviewImage = handlers.onPreview || null;
    onConnectionChange = handlers.onConnectionChange || null;

    connect();
}

function connect() {
    // Clear any existing reconnect timer
    if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }

    // Clear any existing connection timeout
    if (connectionTimer) {
        clearTimeout(connectionTimer);
        connectionTimer = null;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log('Attempting WebSocket connection to:', wsUrl);

    try {
        // Close existing connection if any
        if (ws && ws.readyState !== WebSocket.CLOSED) {
            ws.close();
        }

        ws = new WebSocket(wsUrl);

        // Set a timeout for connection attempt
        connectionTimer = setTimeout(() => {
            console.log('WebSocket connection timeout');
            if (ws && ws.readyState !== WebSocket.OPEN) {
                ws.close();
                scheduleReconnect();
            }
        }, connectionTimeout);

        ws.onopen = () => {
            console.log('WebSocket connected successfully');
            if (connectionTimer) {
                clearTimeout(connectionTimer);
                connectionTimer = null;
            }
            reconnectAttempts = 0;
            notifyConnectionChange(true);
        };

        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'status' && onStatusUpdate) {
                onStatusUpdate(message.data);
            } else if (message.type === 'log' && onLogMessage) {
                onLogMessage(message.data);
            } else if (message.type === 'tasks' && onTasksUpdate) {
                onTasksUpdate(message.data);
            } else if (message.type === 'preview' && onPreviewImage) {
                onPreviewImage(message.data, message.source);
            }
        };

        ws.onclose = (event) => {
            console.log('WebSocket closed', event.code, event.reason);
            if (connectionTimer) {
                clearTimeout(connectionTimer);
                connectionTimer = null;
            }
            ws = null;
            scheduleReconnect();
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            console.log('WebSocket readyState:', ws?.readyState);
            // Close will be called automatically after error
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

function scheduleReconnect() {
    // Fixed 5 second delay between reconnect attempts
    const delay = reconnectDelay;

    notifyConnectionChange(false, 'reconnecting');

    console.log(`Scheduling reconnect in ${delay/1000}s... (attempt ${reconnectAttempts + 1})`);

    reconnectAttempts++;
    reconnectTimer = setTimeout(connect, delay);
}

function notifyConnectionChange(connected, reconnectInfo = '') {
    if (onConnectionChange) {
        onConnectionChange(connected, reconnectInfo);
    }
}
