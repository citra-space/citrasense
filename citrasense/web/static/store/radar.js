// Store methods: radar detection ring buffer, rate calculation, hydration

import * as api from '../api.js';

export const radarMethods = {
    appendRadarDetection(sensorId, det) {
        if (!sensorId || !det) return;
        const existing = this.radarDetections[sensorId] || [];
        const next = existing.slice();
        next.push(det);
        if (next.length > this._radarDetectionsMax) {
            next.splice(0, next.length - this._radarDetectionsMax);
        }
        this.radarDetections = { ...this.radarDetections, [sensorId]: next };
    },

    radarDetectionRate(sensorId, windowSeconds = 60) {
        const arr = this.radarDetections[sensorId];
        if (!arr || !arr.length) return 0;
        const nowUnix = Date.now() / 1000;
        const cutoff = nowUnix - windowSeconds;
        let count = 0;
        for (let i = arr.length - 1; i >= 0; i--) {
            const ts = Number(arr[i]?.ts_unix || 0);
            if (ts < cutoff) break;
            count++;
        }
        return count / windowSeconds;
    },

    radarTracksBySat(sensorId, maxPerSat = 20) {
        const arr = this.radarDetections[sensorId];
        const out = {};
        if (!arr || !arr.length) return out;
        for (let i = 0; i < arr.length; i++) {
            const d = arr[i];
            const key = d?.sat_uuid || d?.sat_name;
            if (!key) continue;
            if (!out[key]) out[key] = [];
            out[key].push(d);
        }
        for (const k of Object.keys(out)) {
            const list = out[k];
            if (list.length > maxPerSat) {
                out[k] = list.slice(list.length - maxPerSat);
            }
        }
        return out;
    },

    async hydrateRadarDetections(sensorId, secondsBack = 300, options = {}) {
        if (!sensorId) return;
        const force = !!options.force;
        if (force) {
            const next = { ...this._radarDetectionsHydrated };
            delete next[sensorId];
            this._radarDetectionsHydrated = next;
        }
        if (this._radarDetectionsHydrated[sensorId]) return;
        this._radarDetectionsHydrated = {
            ...this._radarDetectionsHydrated,
            [sensorId]: true,
        };
        try {
            const result = await api.getRadarDetections(sensorId, -Math.abs(secondsBack));
            if (!result.ok) return;
            const rows = Array.isArray(result.data?.detections) ? result.data.detections : [];
            if (!rows.length) return;
            const live = this.radarDetections[sensorId] || [];
            const combined = rows.concat(live).sort(
                (a, b) => Number(a?.ts_unix || 0) - Number(b?.ts_unix || 0)
            );
            const trimmed = combined.length > this._radarDetectionsMax
                ? combined.slice(combined.length - this._radarDetectionsMax)
                : combined;
            this.radarDetections = { ...this.radarDetections, [sensorId]: trimmed };
        } catch (err) {
            console.debug('hydrateRadarDetections failed:', err);
        }
    },
};
