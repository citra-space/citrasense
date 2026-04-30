import { loadAdapterSchema, loadSensorSchema, loadFilterConfig, buildAdapterFields } from '../config.js';
import { showToast } from '../toast.js';
import * as api from '../api.js';

export const configMethods = {
    async handleAdapterChange(adapter) {
        if (adapter) {
            const sensorCfg = this.config.sensors?.[this.configSensorIndex] || {};
            const currentSettings = sensorCfg.adapter_settings || {};
            await loadAdapterSchema(adapter, currentSettings);
            await loadFilterConfig();
        } else {
            this.adapterFields = [];
            this.filterConfigVisible = false;
        }
    },

    async reloadAdapterSchema() {
        // Reload the active sensor's schema using the current form
        // values — this is what fires when the user picks a new
        // ``camera_type`` (or mount/filter-wheel/focuser type) so that
        // device's own settings appear or disappear in the form.
        //
        // Routes on ``sensor.type``:
        //   - telescope → adapter-driven schema
        //     (``/api/hardware-adapters/{adapter}/schema``)
        //   - everything else → class-level schema
        //     (``/api/sensor-types/{type}/schema``)
        // Both paths thread current values through ``current_settings``
        // so conditional schemas can react.
        const sensorCfg = this.config.sensors?.[this.configSensorIndex];
        if (!sensorCfg) return;

        const currentSettings = {};
        (this.adapterFields || []).forEach(field => {
            currentSettings[field.name] = field.value;
        });

        const sensorType = sensorCfg.type || 'telescope';
        if (sensorType === 'telescope') {
            const adapter = sensorCfg.adapter;
            if (!adapter) return;
            await loadAdapterSchema(adapter, currentSettings);
        } else {
            // ``loadSensorSchema`` reads ``adapter_settings`` off the
            // sensor object; pass a synthetic copy carrying the live
            // form values so the backend gets the freshly-picked
            // camera_type, not whatever was last persisted.
            await loadSensorSchema({ ...sensorCfg, adapter_settings: currentSettings });
        }
    },

    async scanHardware() {
        const adapter = this.config.sensors?.[this.configSensorIndex]?.adapter;
        if (!adapter) return;

        this.isScanning = true;
        try {
            const currentSettings = {};
            (this.adapterFields || []).forEach(field => {
                currentSettings[field.name] = field.value;
            });

            const result = await api.scanHardware(adapter, currentSettings);

            if (!result.ok) {
                showToast(result.error || 'Hardware scan failed', 'danger');
                return;
            }

            this.adapterFields = buildAdapterFields(result.data?.schema || [], currentSettings);
        } catch (error) {
            console.error('Hardware scan failed:', error);
            showToast('Hardware scan failed — check connection', 'danger');
        } finally {
            this.isScanning = false;
        }
    },
};
