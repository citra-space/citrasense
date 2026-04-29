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
                    this.status.sensors[sensorId].task_processing_paused = enabled;
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

    async reconnectHardware() {
        if (this.isReconnecting) return;
        this.isReconnecting = true;
        try {
            const result = await api.reconnectHardware();
            if (result.ok) {
                showToast('Hardware reconnected successfully', 'success');
            } else {
                showToast(result.data?.error || 'Reconnect failed', 'danger');
            }
        } catch (error) {
            console.error('Reconnect error:', error);
            showToast('Reconnect failed: ' + error.message, 'danger');
        } finally {
            this.isReconnecting = false;
        }
    },
};
