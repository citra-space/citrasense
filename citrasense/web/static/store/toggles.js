// Store methods: sensor toggle handlers (processing, observing, self-tasking, scheduling)

import { showToast } from '../toast.js';
import * as api from '../api.js';

export const toggleMethods = {
    async toggleProcessing(enabled, sensorId) {
        if (!sensorId) {
            console.error('toggleProcessing called without sensorId');
            showToast('Cannot toggle task processing: missing sensor_id', 'danger');
            return;
        }
        try {
            const result = enabled ? await api.resumeTasks(sensorId) : await api.pauseTasks(sensorId);
            if (!result.ok) {
                showToast(result.error || 'Failed to toggle task processing', 'danger');
                if (sensorId && this.status?.sensors?.[sensorId]) {
                    this.status.sensors[sensorId].task_processing_paused = !enabled;
                }
            }
        } catch (error) {
            console.error('Error toggling processing:', error);
            showToast('Error toggling task processing', 'danger');
        }
    },

    async toggleObservingSession(enabled, sensorId) {
        if (!sensorId) {
            console.error('toggleObservingSession called without sensorId');
            showToast('Cannot toggle observing session: missing sensor_id', 'danger');
            return;
        }
        try {
            const result = await api.toggleObservingSession(sensorId, enabled);
            if (!result.ok) {
                showToast(result.error || 'Failed to toggle observing session', 'danger');
                if (sensorId && this.status?.sensors?.[sensorId]) {
                    this.status.sensors[sensorId].observing_session_enabled = !enabled;
                }
            }
        } catch (error) {
            console.error('Error toggling observing session:', error);
            showToast('Error toggling observing session', 'danger');
        }
    },

    async toggleSelfTasking(enabled, sensorId) {
        if (!sensorId) {
            console.error('toggleSelfTasking called without sensorId');
            showToast('Cannot toggle self-tasking: missing sensor_id', 'danger');
            return;
        }
        try {
            const result = await api.toggleSelfTasking(sensorId, enabled);
            if (!result.ok) {
                showToast(result.error || 'Failed to toggle self-tasking', 'danger');
                if (sensorId && this.status?.sensors?.[sensorId]) {
                    this.status.sensors[sensorId].self_tasking_enabled = !enabled;
                }
            }
        } catch (error) {
            console.error('Error toggling self-tasking:', error);
            showToast('Error toggling self-tasking', 'danger');
        }
    },

    async toggleStreaming(enabled, sensorId) {
        // Allsky capture-loop on/off (issue #342).  The backend route
        // is allsky-scoped (POST /api/sensors/{id}/allsky/streaming)
        // and the persisted flag lives on the generic
        // ``SensorConfig.streaming_enabled`` so a future radar/audio
        // streaming sensor can opt in with a UI-only change.
        if (!sensorId) {
            console.error('toggleStreaming called without sensorId');
            showToast('Cannot toggle streaming: missing sensor_id', 'danger');
            return;
        }
        try {
            const result = await api.setAllskyStreaming(sensorId, enabled);
            if (!result.ok) {
                showToast(result.error || 'Failed to toggle streaming', 'danger');
                if (sensorId && this.status?.sensors?.[sensorId]) {
                    this.status.sensors[sensorId].streaming_enabled = !enabled;
                }
            }
        } catch (error) {
            console.error('Error toggling streaming:', error);
            showToast('Error toggling streaming', 'danger');
        }
    },

    async toggleAutomatedScheduling(enabled, sensorId) {
        if (!sensorId) {
            console.error('toggleAutomatedScheduling called without sensorId');
            showToast('Cannot toggle automated scheduling: missing sensor_id', 'danger');
            return;
        }
        try {
            const result = await api.toggleScheduling(sensorId, enabled);
            if (!result.ok) {
                showToast(result.error || 'Failed to toggle automated scheduling', 'danger');
                if (sensorId && this.status?.sensors?.[sensorId]) {
                    this.status.sensors[sensorId].automated_scheduling = !enabled;
                }
            }
        } catch (error) {
            console.error('Error toggling automated scheduling:', error);
            showToast('Error toggling automated scheduling', 'danger');
        }
    },

    async reconnectSensor(sensorId) {
        // Per-sensor disconnect+connect cycle.  The HTTP request returns
        // 202 immediately — the actual reconnect runs on the daemon's
        // async init worker and the green/red toast on completion comes
        // through the websocket toast channel via the runtime's
        // ``on_init_state_change`` callback.  No client-side spinner
        // here beyond what the per-sensor card already shows from
        // ``sensor.init_state === 'connecting'``.
        if (!sensorId) {
            console.error('reconnectSensor called without sensorId');
            showToast('Cannot reconnect: missing sensor_id', 'danger');
            return;
        }
        try {
            const result = await api.reconnectSensor(sensorId);
            if (!result.ok && result.status !== 409) {
                // 409 means a reconnect is already in flight; don't toast
                // an error since the existing one will resolve normally.
                showToast(result.error || `Reconnect failed for ${sensorId}`, 'danger');
            }
        } catch (error) {
            console.error('Reconnect error:', error);
            showToast(`Reconnect failed for ${sensorId}: ${error.message}`, 'danger');
        }
    },
};
