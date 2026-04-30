// Alpine.data component: radarControl
// Extracted from _sensor_detail_radar.html inline x-data

import * as api from '../api.js';
import { showToast } from '../toast.js';

export function radarControl(sensorId) {
    return {
        radar: null,
        radarLoading: false,
        lastError: null,
        pingInFlight: false,
        lastPingAt: null,
        lastPingRttMs: null,
        lastPingReply: null,
        _timer: null,

        stateBadgeClass(state) {
            if (state === 'running') return 'bg-success';
            if (state === 'idle') return 'bg-secondary';
            if (state === 'error') return 'bg-danger';
            if (state === 'offline') return 'bg-warning text-dark';
            return 'bg-secondary';
        },

        formatUptime(sec) {
            const n = Number(sec);
            if (!Number.isFinite(n) || n < 0) return '—';
            if (n < 60) return n.toFixed(0) + 's';
            if (n < 3600) return Math.floor(n / 60) + 'm ' + Math.floor(n % 60) + 's';
            const h = Math.floor(n / 3600);
            const m = Math.floor((n % 3600) / 60);
            return h + 'h ' + m + 'm';
        },

        async refresh() {
            this.radarLoading = true;
            try {
                const result = await api.fetchJSON(this.$store.citrasense.sensorApiBaseFor(sensorId) + '/radar/status');
                if (result.ok) { this.radar = result.data; }
            } catch (e) { this.lastError = String(e); }
            finally { this.radarLoading = false; }
        },

        _commandLabel(action, body) {
            if (action === 'start') return (body && body.mock) ? 'Start mock' : 'Start';
            if (action === 'stop') return 'Stop';
            if (action === 'ping') return 'Ping';
            if (action === 'config') return 'Push config';
            return action;
        },

        async sendCommand(action, body) {
            this.lastError = null;
            if (action === 'ping') this.pingInFlight = true;
            const sentAt = performance.now();
            const label = this._commandLabel(action, body);
            const toastId = 'radar-' + sensorId + '-' + action;
            try {
                const url = this.$store.citrasense.sensorApiBaseFor(sensorId) + '/radar/' + action;
                const result = await api.fetchJSON(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
                const data = result.data;
                if (!result.ok || data?.error) {
                    this.lastError = data?.error || ('HTTP ' + result.status);
                    showToast(label + ' failed: ' + this.lastError, 'danger', { id: toastId });
                } else if (data?.ok === false) {
                    this.lastError = data.error || 'pr_sensor rejected command';
                    showToast(label + ' rejected by pr_sensor: ' + this.lastError, 'danger', { id: toastId });
                } else if (action === 'ping') {
                    this.lastPingAt = Date.now();
                    this.lastPingRttMs = performance.now() - sentAt;
                    this.lastPingReply = data;
                    const rtt = this.lastPingRttMs.toFixed(0);
                    const st = data?.state ? ' · pr_sensor ' + data.state : '';
                    showToast('Pong · ' + rtt + ' ms' + st, 'success', { id: toastId });
                } else {
                    showToast(label + ' acknowledged', 'success', { id: toastId });
                }
            } catch (e) {
                this.lastError = String(e);
                showToast(label + ' failed: ' + this.lastError, 'danger', { id: toastId });
            } finally {
                if (action === 'ping') this.pingInFlight = false;
                this.refresh();
            }
        },

        formatPingAge(ts) {
            if (!ts) return '';
            const age = (Date.now() - ts) / 1000;
            if (age < 2) return 'just now';
            if (age < 60) return age.toFixed(0) + 's ago';
            return Math.floor(age / 60) + 'm ago';
        },

        init() {
            this.refresh();
            this._timer = setInterval(() => this.refresh(), 5000);
        },

        destroy() { if (this._timer) clearInterval(this._timer); },
    };
}
