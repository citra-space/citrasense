// Store methods: camera capture and preview loop

import { showToast } from '../toast.js';
import * as api from '../api.js';

export const cameraMethods = {
    async captureImage(sensorId) {
        const duration = this.previewExposureFor(sensorId);
        if (Number.isNaN(duration) || duration <= 0) {
            showToast('Invalid exposure duration', 'danger');
            return;
        }

        this._setSaving(sensorId, true);
        try {
            const result = await api.captureImage(sensorId, duration);
            if (result.ok && result.data?.success) {
                this.captureResult = result.data;
                showToast('Image captured successfully', 'success');
            } else {
                showToast(result.error || 'Failed to capture image', 'danger');
            }
        } catch (error) {
            console.error('Capture error:', error);
            showToast('Failed to capture image: ' + error.message, 'danger');
        } finally {
            this._setSaving(sensorId, false);
        }
    },

    async capturePreview(sensorId) {
        if (this.isImagingTaskActive(sensorId)) {
            if (this.isLoopingFor(sensorId)) this.loopingSensorId = null;
            return;
        }
        try {
            const result = await api.capturePreview(
                sensorId,
                this.previewExposureFor(sensorId),
                this.previewFlipHFor(sensorId),
            );
            if (result.status === 409) {
                if (this.isLoopingFor(sensorId)) {
                    setTimeout(() => this.capturePreview(sensorId), 250);
                }
                return;
            }
            if (result.ok && result.data?.image_data) {
                this.previewDataUrls = { ...this.previewDataUrls, [sensorId]: result.data.image_data };
                this.loopCount++;
            } else {
                showToast(result.error || 'Preview failed', 'danger');
                if (this.isLoopingFor(sensorId)) this.loopingSensorId = null;
                return;
            }
        } catch (error) {
            console.error('Preview error:', error);
            if (this.isLoopingFor(sensorId)) this.loopingSensorId = null;
            return;
        }

        if (this.isLoopingFor(sensorId)) {
            requestAnimationFrame(() => this.capturePreview(sensorId));
        }
    },

    startFocusLoop(sensorId) {
        if (!sensorId) return;
        if (this.loopingSensorId !== null || this.isImagingTaskActive(sensorId)) return;
        this.loopingSensorId = sensorId;
        this.loopCount = 0;
        this.capturePreview(sensorId);
    },

    stopFocusLoop() {
        this.loopingSensorId = null;
    },

    async singlePreview(sensorId) {
        if (this.loopingSensorId !== null) return;
        this._setCapturing(sensorId, true);
        try {
            const result = await api.capturePreview(
                sensorId,
                this.previewExposureFor(sensorId),
                this.previewFlipHFor(sensorId),
            );
            if (result.ok && result.data?.image_data) {
                this.previewDataUrls = { ...this.previewDataUrls, [sensorId]: result.data.image_data };
                this.loopCount++;
            } else {
                showToast(result.error || 'Preview failed', 'danger');
            }
        } catch (error) {
            showToast('Preview failed: ' + error.message, 'danger');
        } finally {
            this._setCapturing(sensorId, false);
        }
    },
};
