// Alpine.data component: calibrationPane
// Extracted from _config_calibration.html inline x-data

import * as api from '../api.js';
import { showToast } from '../toast.js';

export function calibrationPane() {
    return {
        calStatus: null,
        calLoading: false,
        calPollTimer: null,
        calFrameType: 'bias',
        calExposure: 2.0,
        calFilterPosition: null,
        calUploadFrameType: 'flat',
        calUploadFilter: '',
        calUploadNormalize: true,
        calUploadBusy: false,
        calUploadResult: null,
        twilightInfo: null,
        twilightTimer: null,

        async loadCalStatus() {
            try {
                const result = await api.getCalibrationStatus(this.$store.citrasense.configSensorId);
                this.calStatus = result.ok ? result.data : null;
            } catch (e) { this.calStatus = null; }
        },

        async loadTwilight() {
            try {
                const result = await api.getTwilight();
                this.twilightInfo = result.ok ? result.data : null;
            } catch (e) { this.twilightInfo = null; }
        },

        startTwilightPolling() {
            if (this.twilightTimer) return;
            this.loadTwilight();
            this.twilightTimer = setInterval(() => this.loadTwilight(), 60000);
        },

        stopTwilightPolling() {
            if (this.twilightTimer) { clearInterval(this.twilightTimer); this.twilightTimer = null; }
        },

        formatTwilight() {
            const t = this.twilightInfo;
            if (!t || !t.location_available) return null;
            if (t.in_flat_window && t.flat_window) {
                return { text: 'Flat window now — ends in ' + Math.round(t.flat_window.remaining_minutes) + 'm', cls: 'text-success' };
            }
            if (t.next_flat_window) {
                const start = new Date(t.next_flat_window.start);
                const now = new Date();
                const diffMin = Math.round((start - now) / 60000);
                const timeStr = start.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                const typeStr = t.next_flat_window.type === 'morning' ? 'Morning' : 'Evening';
                if (diffMin < 120) {
                    return { text: typeStr + ' flat window in ' + diffMin + 'm (' + timeStr + ')', cls: 'text-info' };
                }
                const hours = Math.floor(diffMin / 60);
                const mins = diffMin % 60;
                return { text: typeStr + ' flat window in ' + hours + 'h ' + mins + 'm (' + timeStr + ')', cls: 'text-muted' };
            }
            return { text: 'Sun alt: ' + t.current_sun_altitude + '°', cls: 'text-muted' };
        },

        startPolling() {
            if (this.calPollTimer) return;
            this.calPollTimer = setInterval(async () => {
                await this.loadCalStatus();
                if (!this.calStatus?.capture_running && !this.calStatus?.capture_requested) {
                    this.stopPolling();
                }
            }, 2000);
        },

        stopPolling() {
            if (this.calPollTimer) { clearInterval(this.calPollTimer); this.calPollTimer = null; }
        },

        async captureFrames() {
            try {
                let body = {
                    frame_type: this.calFrameType,
                    count: this.calStatus?.frame_count_setting || 30,
                    gain: this.calStatus?.current_gain,
                    binning: this.calStatus?.current_binning,
                };
                if (this.calFrameType === 'dark' || this.calFrameType === 'flat') {
                    body.exposure_time = this.calExposure;
                }
                if (this.calFrameType === 'flat') {
                    body.filter_position = this.calFilterPosition;
                }
                await api.startCalibrationCapture(this.$store.citrasense.configSensorId, body);
                await this.loadCalStatus();
                this.startPolling();
            } catch (e) { console.error(e); }
        },

        async cancelCapture() {
            await api.cancelCalibrationCapture(this.$store.citrasense.configSensorId);
        },

        async captureSuite(suiteName) {
            try {
                const result = await api.startCalibrationSuite(this.$store.citrasense.configSensorId, { suite: suiteName });
                if (!result.ok) {
                    showToast(result.error || 'Failed to start suite', 'danger');
                    return;
                }
                await this.loadCalStatus();
                this.startPolling();
            } catch (e) { console.error(e); }
        },

        async uploadMaster() {
            const input = this.$refs.calUploadFile;
            if (!input || !input.files || !input.files[0]) {
                this.calUploadResult = { error: 'Select a FITS file first' };
                return;
            }
            const file = input.files[0];
            this.calUploadBusy = true;
            this.calUploadResult = null;
            try {
                const sensorId = this.$store.citrasense.configSensorId;
                const qs = new URLSearchParams({
                    frame_type: this.calUploadFrameType,
                    normalize_flat: this.calUploadNormalize ? 'true' : 'false',
                    override_filter: this.calUploadFilter || '',
                }).toString();
                const result = await api.uploadCalibration(sensorId, qs, file);
                if (!result.ok) {
                    this.calUploadResult = { error: result.error || ('Upload failed (' + result.status + ')') };
                } else {
                    this.calUploadResult = result.data;
                    input.value = '';
                    await this.loadCalStatus();
                }
            } catch (e) {
                this.calUploadResult = { error: String(e) };
            } finally {
                this.calUploadBusy = false;
            }
        },

        async deleteMaster(type, entry) {
            if (!confirm('Delete this master frame?')) return;
            await api.deleteMasterCalibration(this.$store.citrasense.configSensorId, {
                frame_type: type,
                camera_id: this.calStatus?.camera_id,
                gain: entry.gain,
                binning: entry.binning,
                exposure_time: entry.exposure_time || 0,
                temperature: entry.temperature,
                filter_name: entry.filter || '',
                read_mode: entry.read_mode || ''
            });
            this.loadCalStatus();
        },
    };
}
