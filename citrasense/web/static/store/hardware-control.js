// Store methods: hardware control (mount, focuser, autofocus, alignment, safety, filters)

import { showToast } from '../toast.js';
import * as api from '../api.js';

export const hardwareControlMethods = {
    async triggerAutofocus(sensorId) {
        const ss = this.status?.sensors?.[sensorId] || {};
        const shouldCancel = ss?.autofocus_requested || ss?.autofocus_running;

        if (shouldCancel) {
            try {
                const result = await api.cancelAutofocus(sensorId);
                if (result.ok && result.data?.success) {
                    showToast('Autofocus cancelled', 'info');
                } else {
                    showToast('Nothing to cancel', 'warning');
                }
            } catch (error) {
                console.error('Error cancelling autofocus:', error);
                showToast('Failed to cancel autofocus', 'danger');
            }
            return;
        }

        this._setAutofocusing(sensorId, true);
        try {
            const result = await api.triggerAutofocus(sensorId);
            if (result.ok) {
                showToast('Autofocus queued', 'success');
            } else {
                showToast(result.error || 'Autofocus request failed', 'danger');
            }
        } catch (error) {
            console.error('Error triggering autofocus:', error);
            showToast('Failed to trigger autofocus', 'danger');
        } finally {
            this._setAutofocusing(sensorId, false);
        }
    },

    async triggerAlignment(sensorId) {
        const ss = this.status?.sensors?.[sensorId] || {};
        const isCancel = ss?.alignment_requested;

        if (isCancel) {
            try {
                const result = await api.cancelAlignment(sensorId);
                if (result.ok && result.data?.success) {
                    showToast('Alignment cancelled', 'info');
                } else {
                    showToast('Nothing to cancel', 'warning');
                }
            } catch (error) {
                console.error('Error cancelling alignment:', error);
                showToast('Failed to cancel alignment', 'danger');
            }
            return;
        }

        try {
            const result = await api.triggerAlignment(sensorId);
            if (result.ok) {
                showToast('Alignment queued', 'success');
            } else {
                showToast(result.error || 'Alignment request failed', 'danger');
            }
        } catch (error) {
            console.error('Error triggering alignment:', error);
            showToast('Failed to trigger alignment', 'danger');
        }
    },

    async manualSync(sensorId, ra, dec) {
        if (ra === '' || ra == null || dec === '' || dec == null) {
            showToast('Enter RA and Dec values before syncing', 'warning');
            return;
        }
        try {
            const result = await api.syncTelescope(sensorId, parseFloat(ra), parseFloat(dec));
            if (result.ok && result.data?.success) {
                showToast(result.data.message || 'Mount synced', 'success');
            } else {
                showToast(result.error || 'Sync failed', 'danger');
            }
        } catch (error) {
            console.error('Error syncing mount:', error);
            showToast('Failed to sync mount', 'danger');
        }
    },

    async homeMount(sensorId) {
        try {
            const result = await api.mountHome(sensorId);
            if (result.ok) {
                showToast('Mount homing initiated', 'success');
            } else {
                showToast(result.error || 'Failed to home mount', 'danger');
            }
        } catch (error) {
            console.error('Error homing mount:', error);
            showToast('Failed to home mount', 'danger');
        }
    },

    async triggerCableUnwind(sensorId) {
        try {
            const result = await api.mountUnwind(sensorId);
            if (result.ok) {
                showToast('Cable unwind started — monitor progress in the Telescope card', 'success');
            } else {
                showToast(result.error || 'Cable unwind failed', 'danger');
            }
        } catch (error) {
            console.error('Error triggering cable unwind:', error);
            showToast('Failed to trigger cable unwind', 'danger');
        }
    },

    async resetCableWrap(sensorId) {
        try {
            const result = await api.resetCableWrap(sensorId);
            if (result.ok) {
                showToast('Cable wrap counter reset to 0°', 'success');
            } else {
                showToast(result.error || 'Reset failed', 'danger');
            }
        } catch (error) {
            console.error('Error resetting cable wrap:', error);
            showToast('Failed to reset cable wrap', 'danger');
        }
    },

    async emergencyStop() {
        try {
            const result = await api.emergencyStop();
            if (result.ok) {
                showToast(result.data?.message || 'Emergency stop executed', 'warning');
            } else {
                showToast(result.error || 'Emergency stop failed', 'danger');
            }
        } catch (error) {
            console.error('Emergency stop error:', error);
            showToast('Failed to execute emergency stop', 'danger');
        }
    },

    async clearOperatorStop() {
        try {
            const result = await api.clearOperatorStop();
            if (result.ok) {
                showToast(result.data?.message || 'Operator stop cleared', 'success');
            } else {
                showToast(result.error || 'Failed to clear operator stop', 'danger');
            }
        } catch (error) {
            console.error('Clear operator stop error:', error);
            showToast('Failed to clear operator stop', 'danger');
        }
    },

    async changeFilterPosition(sensorId, position) {
        const wasLooping = this.isLoopingFor(sensorId);
        if (wasLooping) this.stopFocusLoop();

        try {
            const result = await api.setFilter(sensorId, parseInt(position));
            if (result.ok) {
                showToast(`Filter changed to ${result.data?.name || position}`, 'success');
            } else {
                showToast(result.error || 'Filter change failed', 'danger');
            }
        } catch (error) {
            console.error('Filter change error:', error);
            showToast('Failed to change filter', 'danger');
        }

        if (wasLooping) this.startFocusLoop(sensorId);
    },

    async moveFocuserRelative(sensorId, steps) {
        try {
            const result = await api.moveFocuserRelative(sensorId, parseInt(steps));
            if (!result.ok) {
                showToast(result.error || 'Focuser move failed', 'danger');
            }
        } catch (error) {
            console.error('Focuser move error:', error);
            showToast('Failed to move focuser', 'danger');
        }
    },

    async moveFocuserAbsolute(sensorId, position) {
        try {
            const result = await api.moveFocuser(sensorId, parseInt(position));
            if (result.ok) {
                showToast(`Focuser moved to ${result.data?.position ?? position}`, 'success');
            } else {
                showToast(result.error || 'Focuser move failed', 'danger');
            }
        } catch (error) {
            console.error('Focuser move error:', error);
            showToast('Failed to move focuser', 'danger');
        }
    },

    async abortFocuser(sensorId) {
        try {
            const result = await api.abortFocuser(sensorId);
            if (result.ok) {
                showToast('Focuser stopped', 'warning');
            } else {
                showToast(result.error || 'Failed to stop focuser', 'danger');
            }
        } catch (error) {
            console.error('Focuser abort error:', error);
            showToast('Failed to stop focuser', 'danger');
        }
    },

    async mountMove(sensorId, action, direction) {
        try {
            const result = await api.mountMove(sensorId, action, direction);
            if (!result.ok) {
                console.error('Mount move error:', result.error);
            }
        } catch (error) {
            console.error('Mount move error:', error);
        }
    },

    async mountGoto(sensorId, ra, dec) {
        try {
            const result = await api.mountGoto(sensorId, parseFloat(ra), parseFloat(dec));
            if (result.ok) {
                showToast(`Slewing to RA=${parseFloat(ra).toFixed(2)}°, Dec=${parseFloat(dec).toFixed(2)}°`, 'info');
            } else {
                showToast(result.error || 'Goto failed', 'danger');
            }
        } catch (error) {
            console.error('Mount goto error:', error);
            showToast('Failed to send goto command', 'danger');
        }
    },

    async mountSetTracking(sensorId, enabled) {
        try {
            const result = await api.mountTracking(sensorId, enabled);
            if (result.ok) {
                const sd = this.status?.sensors?.[sensorId];
                if (sd) sd.mount_tracking = enabled;
                showToast(enabled ? 'Sidereal tracking started' : 'Tracking stopped', 'info');
            } else {
                showToast(result.error || 'Tracking command failed', 'danger');
            }
        } catch (error) {
            console.error('Mount tracking error:', error);
            showToast('Failed to change tracking', 'danger');
        }
    },

    async calibratePointingModel(sensorId) {
        try {
            const result = await api.calibratePointingModel(sensorId);
            if (result.ok && result.data?.success) {
                showToast('Pointing calibration started', 'success');
            } else {
                showToast(result.error || 'Failed to start calibration', 'danger');
            }
        } catch (error) {
            console.error('Error starting pointing calibration:', error);
            showToast('Failed to start pointing calibration', 'danger');
        }
    },

    async resetPointingModel(sensorId) {
        try {
            const result = await api.resetPointingModel(sensorId);
            if (result.ok && result.data?.success) {
                showToast('Pointing model reset', 'success');
            } else {
                showToast(result.error || 'Failed to reset pointing model', 'danger');
            }
        } catch (error) {
            console.error('Error resetting pointing model:', error);
            showToast('Failed to reset pointing model', 'danger');
        }
    },

    async cancelPointingCalibration(sensorId) {
        try {
            const result = await api.cancelPointingModelCalibration(sensorId);
            if (result.ok && result.data?.success) {
                showToast('Pointing calibration cancelled', 'success');
            } else {
                showToast(result.error || 'Failed to cancel calibration', 'danger');
            }
        } catch (error) {
            console.error('Error cancelling pointing calibration:', error);
            showToast('Failed to cancel pointing calibration', 'danger');
        }
    },

    async removeSensor(sid) {
        if (!confirm("Remove sensor '" + sid + "'? You will need to Save & Reload to finalize.")) return;
        try {
            const result = await api.deleteSensor(sid);
            if (!result.ok) { showToast(result.error || 'Failed to remove sensor', 'danger'); return; }
            const sensors = this.config.sensors;
            const idx = sensors.findIndex(s => s.id === sid);
            if (idx >= 0) sensors.splice(idx, 1);
            delete this.savedAdapters[sid];
            if (this.configSensorId === sid) {
                this.configSensorId = sensors[0]?.id || null;
            }
        } catch (e) {
            showToast(e.message, 'danger');
        }
    },
};
