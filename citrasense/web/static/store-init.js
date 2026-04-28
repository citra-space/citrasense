/**
 * CitraSense Alpine store - must register BEFORE Alpine starts.
 * Load this script before Alpine.js so the alpine:init listener is attached in time.
 */
import * as formatters from './formatters.js';
import * as components from './components.js';
import { FILTER_COLORS } from './filters.js';

function compareVersions(v1, v2) {
    v1 = (v1 || '').replace(/^v/, '');
    v2 = (v2 || '').replace(/^v/, '');
    const parts1 = v1.split('.').map(n => parseInt(n) || 0);
    const parts2 = v2.split('.').map(n => parseInt(n) || 0);
    const maxLen = Math.max(parts1.length, parts2.length);
    for (let i = 0; i < maxLen; i++) {
        const num1 = parts1[i] || 0;
        const num2 = parts2[i] || 0;
        if (num1 > num2) return 1;
        if (num1 < num2) return -1;
    }
    return 0;
}

(() => {
    document.addEventListener('alpine:init', () => {
        // Register Alpine components FIRST (before Alpine starts processing the DOM)
        window.Alpine.data('adapterField', components.adapterField);
        window.Alpine.data('taskRow', components.taskRow);
        window.Alpine.data('filterRow', components.filterRow);
        window.Alpine.data('logEntry', components.logEntry);

        // Register store
        window.Alpine.store('citrasense', {
            status: {},
            tasks: [],
            logs: [],
            latestLog: null,
            wsConnected: false,
            wsReconnecting: false,
            wsReconnectAt: 0,
            wsLastMessage: 0,
            // Per-sensor "currently executing" map keyed by sensor_id.
            // No scalar currentTaskId — in a multi-sensor deployment
            // "current" is per sensor; use activeTaskIdSet for any
            // "is this task in-flight anywhere" check.
            currentTaskIds: {},
            activeTaskIdSet: new Set(),
            // Wall-clock heartbeat bumped once per second by the ticker in
            // app.js.  Alpine getters that depend on "now" (e.g. taskRow's
            // countdownText) read this so they re-render reactively without
            // each component running its own setInterval.
            now: Date.now(),

            // Multi-sensor support
            sensors: [],
            // ``currentSensorId`` is set by the client-side path router in
            // app.js when the URL is ``/sensors/<id>``.  ``currentSensor``
            // resolves it against the live enriched sensor list so the
            // detail template's bindings are simple ``sensor.*`` reads.
            currentSensorId: null,
            get currentSensor() {
                const id = this.currentSensorId;
                if (!id) return null;
                return (this.sensors || []).find(s => s.id === id) || null;
            },

            // Log panel per-sensor filter. ``null`` = show all, ``"__site__"``
            // = only entries without a sensor_id, otherwise a specific
            // sensor_id.
            logSensorFilter: null,
            get filteredLogs() {
                const f = this.logSensorFilter;
                if (f === null) return this.logs;
                if (f === '__site__') return this.logs.filter(l => !l.sensor_id);
                return this.logs.filter(l => l.sensor_id === f);
            },

            sensorApiBaseFor(sensorId) {
                if (!sensorId) {
                    throw new Error('sensorApiBaseFor called without sensorId');
                }
                return `/api/sensors/${sensorId}`;
            },

            // Config editing state.  The selected sensor id is persisted
            // to ``localStorage`` so reopening the Config tab lands back
            // on the sensor the operator was last editing (important on
            // multi-sensor sites where silently defaulting to sensors[0]
            // has caused operators to edit the wrong rig).
            _configSensorId: null,
            _configSensorStorageKey: 'citrasense.configSensorId',
            get configSensorId() {
                return this._configSensorId;
            },
            set configSensorId(value) {
                if (this._configSensorId === value) return;
                this._configSensorId = value;
                try {
                    if (value) {
                        localStorage.setItem(this._configSensorStorageKey, value);
                    } else {
                        localStorage.removeItem(this._configSensorStorageKey);
                    }
                } catch {
                    // localStorage may be unavailable in private browsing /
                    // iframes; silently fall back to in-memory state.
                }
                // Reload adapter schema+values, filter config, and processors
                // for the newly selected sensor so the hardware / calibration /
                // pipeline tabs reflect that sensor's settings instead of
                // bleeding the previous sensor's adapterFields into saves.
                if (typeof window !== 'undefined' && typeof window._configOnSensorSwitch === 'function') {
                    Promise.resolve().then(() => {
                        try {
                            window._configOnSensorSwitch(value);
                        } catch (err) {
                            console.error('configSensorId switch handler failed:', err);
                        }
                    });
                }
            },
            get configSensorIndex() {
                if (!this.config?.sensors?.length) return -1;
                // When no sensor is explicitly selected yet (e.g. the
                // config hasn't finished loading), fall back to 0 so the
                // many ``x-model="config.sensors[configSensorIndex]..."``
                // bindings don't explode on first paint.  But when a
                // non-null ``configSensorId`` doesn't match any sensor,
                // return -1 so the form doesn't silently let the
                // operator edit the wrong rig.
                if (!this.configSensorId) return 0;
                return this.config.sensors.findIndex(s => s.id === this.configSensorId);
            },
            loadPersistedConfigSensorId() {
                try {
                    return localStorage.getItem(this._configSensorStorageKey);
                } catch {
                    return null;
                }
            },
            get configSensorStatus() {
                const sid = this.configSensorId;
                return sid ? (this.status?.sensors?.[sid] || {}) : {};
            },

            config: {},
            apiEndpoint: 'production',
            hardwareAdapters: [], // [{value, label}]
            filters: {},
            savedAdapters: {},
            enabledFilters: [],
            filterConfigVisible: false,
            filterNamesEditable: false,
            filterNameOptions: [],
            filterColors: FILTER_COLORS,
            filterAdapterChangeMessageVisible: false,
            currentSection: 'monitoring',
            version: '',
            versionData: null,
            updateIndicator: '',
            versionCheckState: 'idle',
            versionCheckResult: null,

            // Autofocus target presets (loaded from API)
            autofocusPresets: [],

            // Loading states for async operations.  ``isSavingConfig``,
            // ``isScanning``, and ``isReconnecting`` are intrinsically
            // site-wide.  Per-sensor "is X happening" state lives in the
            // maps below so one sensor's Snap/Save/Loop/Autofocus doesn't
            // flip the same button on a different sensor's card.
            isSavingConfig: false,
            isScanning: false,
            isReconnecting: false,
            captureResult: null,

            // Which sensor's focus loop is currently running (only one at a
            // time since it occupies a single RAF loop on the client).
            loopingSensorId: null,
            previewSource: '',
            // Per-sensor preview data URLs.  There is no singular
            // ``previewDataUrl`` slot — the fullscreen modal reads
            // ``previewDataUrls[activePreviewSensorId]`` from the card
            // that opened it.
            previewDataUrls: {},
            // Which sensor owns the fullscreen preview modal right now.
            activePreviewSensorId: null,
            loopCount: 0,
            // Per-sensor transient flags and settings.  Accessed through
            // helper methods below so call sites never have to know the
            // underlying map shape.
            capturingSensors: {},
            savingSensors: {},
            autofocusingSensors: {},
            previewExposures: {},
            previewFlipHFlags: {},
            // Per-sensor "last seen task image URL" tracking used by
            // updateStoreFromStatus in app.js to decide when to refresh
            // previewDataUrls[sensor_id].  Initialized lazily there.
            _lastTaskImageUrlBySensor: {},

            isLoopingFor(sensorId) {
                return !!sensorId && this.loopingSensorId === sensorId;
            },
            get isAnyLooping() {
                return this.loopingSensorId !== null;
            },
            isCapturingFor(sensorId) {
                return !!sensorId && this.capturingSensors[sensorId] === true;
            },
            isSavingFor(sensorId) {
                return !!sensorId && this.savingSensors[sensorId] === true;
            },
            isAutofocusingFor(sensorId) {
                return !!sensorId && this.autofocusingSensors[sensorId] === true;
            },
            previewExposureFor(sensorId) {
                if (!sensorId) return 2.0;
                const v = this.previewExposures[sensorId];
                return v == null ? 2.0 : v;
            },
            setPreviewExposure(sensorId, value) {
                if (!sensorId) return;
                this.previewExposures = { ...this.previewExposures, [sensorId]: value };
            },
            previewFlipHFor(sensorId) {
                return !!sensorId && this.previewFlipHFlags[sensorId] === true;
            },
            togglePreviewFlipH(sensorId) {
                if (!sensorId) return;
                this.previewFlipHFlags = {
                    ...this.previewFlipHFlags,
                    [sensorId]: !this.previewFlipHFlags[sensorId],
                };
            },
            _setCapturing(sensorId, value) {
                if (!sensorId) return;
                this.capturingSensors = { ...this.capturingSensors, [sensorId]: !!value };
            },
            _setSaving(sensorId, value) {
                if (!sensorId) return;
                this.savingSensors = { ...this.savingSensors, [sensorId]: !!value };
            },
            _setAutofocusing(sensorId, value) {
                if (!sensorId) return;
                this.autofocusingSensors = { ...this.autofocusingSensors, [sensorId]: !!value };
            },

            // Spread all formatter functions from shared module
            ...formatters,

            // Unified adapter fields (schema + values merged)
            adapterFields: [],

            // Computed property: Group adapter fields by their group property
            get groupedAdapterFields() {
                const grouped = {};
                this.adapterFields.forEach(f => {
                    const g = f.group || 'General';
                    if (!grouped[g]) grouped[g] = [];
                    grouped[g].push(f);
                });
                return Object.entries(grouped);
            },

            telescopeTooltip(sensorStatus) {
                const s = sensorStatus || {};
                if (!s.telescope_connected) return 'Telescope disconnected';
                let tip = 'Telescope connected';
                const pm = s.pointing_model;
                if (pm && pm.state !== 'untrained') {
                    const live = pm.live_accuracy;
                    if (live?.count > 0) {
                        tip += '\nPointing: ' + live.median_deg?.toFixed(4) + '° live (' + live.count + ' solves)';
                    } else {
                        tip += '\nPointing: ~' + pm.pointing_accuracy_deg?.toFixed(4) + '° (model fit)';
                    }
                } else if (pm) {
                    tip += '\nPointing: untrained';
                }
                return tip;
            },

            async captureImage(sensorId) {
                const duration = this.previewExposureFor(sensorId);
                if (Number.isNaN(duration) || duration <= 0) {
                    const { showToast } = await import('./config.js');
                    showToast('Invalid exposure duration', 'danger');
                    return;
                }

                this._setSaving(sensorId, true);
                try {
                    const response = await fetch(`${this.sensorApiBaseFor(sensorId)}/camera/capture`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ duration })
                    });
                    const data = await response.json();

                    if (response.ok && data.success) {
                        this.captureResult = data;
                        const { showToast } = await import('./config.js');
                        showToast('Image captured successfully', 'success');
                    } else {
                        const { showToast } = await import('./config.js');
                        showToast(data.error || 'Failed to capture image', 'danger');
                    }
                } catch (error) {
                    console.error('Capture error:', error);
                    const { showToast } = await import('./config.js');
                    showToast('Failed to capture image: ' + error.message, 'danger');
                } finally {
                    this._setSaving(sensorId, false);
                }
            },

            async toggleProcessing(enabled, sensorId) {
                if (!sensorId) {
                    console.error('toggleProcessing called without sensorId');
                    alert('Cannot toggle task processing: missing sensor_id');
                    return;
                }
                const action = enabled ? 'resume' : 'pause';
                const endpoint = `/api/sensors/${encodeURIComponent(sensorId)}/tasks/${action}`;
                try {
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    const result = await response.json();
                    if (!response.ok) {
                        alert(result.error || 'Failed to toggle task processing');
                        if (sensorId && this.status?.sensors?.[sensorId]) {
                            this.status.sensors[sensorId].task_processing_paused = enabled;
                        }
                    }
                } catch (error) {
                    console.error('Error toggling processing:', error);
                    alert('Error toggling task processing');
                }
            },

            async toggleObservingSession(enabled, sensorId) {
                if (!sensorId) {
                    console.error('toggleObservingSession called without sensorId');
                    alert('Cannot toggle observing session: missing sensor_id');
                    return;
                }
                const body = { enabled };
                try {
                    const response = await fetch(`/api/sensors/${encodeURIComponent(sensorId)}/observing-session`, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body)
                    });
                    const result = await response.json();
                    if (!response.ok) {
                        alert(result.error || 'Failed to toggle observing session');
                        if (sensorId && this.status?.sensors?.[sensorId]) {
                            this.status.sensors[sensorId].observing_session_enabled = !enabled;
                        }
                    }
                } catch (error) {
                    console.error('Error toggling observing session:', error);
                    alert('Error toggling observing session');
                }
            },

            async toggleSelfTasking(enabled, sensorId) {
                if (!sensorId) {
                    console.error('toggleSelfTasking called without sensorId');
                    alert('Cannot toggle self-tasking: missing sensor_id');
                    return;
                }
                const body = { enabled };
                try {
                    const response = await fetch(`/api/sensors/${encodeURIComponent(sensorId)}/self-tasking`, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body)
                    });
                    const result = await response.json();
                    if (!response.ok) {
                        alert(result.error || 'Failed to toggle self-tasking');
                        if (sensorId && this.status?.sensors?.[sensorId]) {
                            this.status.sensors[sensorId].self_tasking_enabled = !enabled;
                        }
                    }
                } catch (error) {
                    console.error('Error toggling self-tasking:', error);
                    alert('Error toggling self-tasking');
                }
            },

            async toggleAutomatedScheduling(enabled, sensorId) {
                if (!sensorId) {
                    console.error('toggleAutomatedScheduling called without sensorId');
                    alert('Cannot toggle automated scheduling: missing sensor_id');
                    return;
                }
                const body = { enabled };
                try {
                    const response = await fetch(`/api/sensors/${encodeURIComponent(sensorId)}/scheduling`, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body)
                    });
                    const result = await response.json();
                    if (!response.ok) {
                        alert(result.error || 'Failed to toggle automated scheduling');
                        if (sensorId && this.status?.sensors?.[sensorId]) {
                            this.status.sensors[sensorId].automated_scheduling = !enabled;
                        }
                    }
                } catch (error) {
                    console.error('Error toggling automated scheduling:', error);
                    alert('Error toggling automated scheduling');
                }
            },

            async reconnectHardware() {
                if (this.isReconnecting) return;
                this.isReconnecting = true;
                try {
                    const { reconnectHardware } = await import('./api.js');
                    const result = await reconnectHardware();
                    const { showToast } = await import('./config.js');
                    if (result.ok) {
                        showToast('Hardware reconnected successfully', 'success');
                    } else {
                        showToast(result.data?.error || 'Reconnect failed', 'danger');
                    }
                } catch (error) {
                    console.error('Reconnect error:', error);
                    const { showToast } = await import('./config.js');
                    showToast('Reconnect failed: ' + error.message, 'danger');
                } finally {
                    this.isReconnecting = false;
                }
            },

            get isSystemBusy() {
                return this.status?.system_busy === true;
            },
            get systemBusyReason() {
                return this.status?.system_busy_reason || '';
            },
            /**
             * True when the given sensor is running an imaging task or its
             * processing queue is active.  Always takes a sensorId — there
             * is no "site-wide" imaging state in a multi-sensor deployment.
             */
            isImagingTaskActive(sensorId) {
                if (!sensorId) {
                    console.error('isImagingTaskActive called without sensorId');
                    return false;
                }
                const sd = this.status?.sensors?.[sensorId];
                if (!sd) return false;
                return sd.system_busy === true || sd.processing_active === true;
            },

            async capturePreview(sensorId) {
                if (this.isImagingTaskActive(sensorId)) {
                    if (this.isLoopingFor(sensorId)) this.loopingSensorId = null;
                    return;
                }
                try {
                    const response = await fetch(`${this.sensorApiBaseFor(sensorId)}/camera/preview`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            duration: this.previewExposureFor(sensorId),
                            flip_horizontal: this.previewFlipHFor(sensorId),
                        })
                    });
                    if (response.status === 409) {
                        if (this.isLoopingFor(sensorId)) {
                            setTimeout(() => this.capturePreview(sensorId), 250);
                        }
                        return;
                    }
                    const data = await response.json();
                    if (response.ok && data.image_data) {
                        this.previewDataUrls = { ...this.previewDataUrls, [sensorId]: data.image_data };
                        this.loopCount++;
                    } else {
                        const { showToast } = await import('./config.js');
                        showToast(data.error || 'Preview failed', 'danger');
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
                    const response = await fetch(`${this.sensorApiBaseFor(sensorId)}/camera/preview`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            duration: this.previewExposureFor(sensorId),
                            flip_horizontal: this.previewFlipHFor(sensorId),
                        })
                    });
                    const data = await response.json();
                    if (response.ok && data.image_data) {
                        this.previewDataUrls = { ...this.previewDataUrls, [sensorId]: data.image_data };
                        this.loopCount++;
                    } else {
                        const { showToast } = await import('./config.js');
                        showToast(data.error || 'Preview failed', 'danger');
                    }
                } catch (error) {
                    const { showToast } = await import('./config.js');
                    showToast('Preview failed: ' + error.message, 'danger');
                } finally {
                    this._setCapturing(sensorId, false);
                }
            },

            async checkForUpdates() {
                try {
                    const versionResponse = await fetch('/api/version');
                    const versionData = await versionResponse.json();
                    this.versionData = versionData;
                    const currentVersion = versionData.version;
                    const installType = versionData.install_type || 'pypi';
                    const gitHash = versionData.git_hash;
                    const gitBranch = versionData.git_branch;
                    const gitDirty = versionData.git_dirty || false;
                    const base = { currentVersion, installType, gitHash, gitBranch, gitDirty };

                    if (currentVersion === 'development' || currentVersion === 'unknown') {
                        this.updateIndicator = '';
                        return { status: 'up-to-date', ...base };
                    }

                    if (installType !== 'pypi' && gitHash) {
                        const compareResponse = await fetch(
                            `https://api.github.com/repos/citra-space/citrasense/compare/${gitHash}...main`
                        );
                        if (!compareResponse.ok) {
                            return { status: 'error', ...base };
                        }
                        const compareData = await compareResponse.json();
                        const behindBy = compareData.ahead_by || 0;
                        if (behindBy > 0) {
                            this.updateIndicator = `${behindBy} commit${behindBy !== 1 ? 's' : ''} behind`;
                            return { status: 'update-available', ...base, behindBy };
                        }
                        this.updateIndicator = '';
                        return { status: 'up-to-date', ...base };
                    }

                    const githubResponse = await fetch('https://api.github.com/repos/citra-space/citrasense/releases/latest');
                    if (!githubResponse.ok) {
                        return { status: 'error', ...base };
                    }

                    const releaseData = await githubResponse.json();
                    const latestVersion = releaseData.tag_name.replace(/^v/, '');
                    const releaseUrl = releaseData.html_url;

                    if (compareVersions(latestVersion, currentVersion) > 0) {
                        this.updateIndicator = `${latestVersion} Available!`;
                        return { status: 'update-available', ...base, latestVersion, releaseUrl };
                    } else {
                        this.updateIndicator = '';
                        return { status: 'up-to-date', ...base };
                    }
                } catch (error) {
                    console.debug('Update check failed:', error);
                    this.updateIndicator = '';
                    return { status: 'error', currentVersion: 'unknown' };
                }
            },

            async showVersionModal() {
                this.versionCheckState = 'loading';
                this.versionCheckResult = null;

                const modal = new bootstrap.Modal(document.getElementById('versionModal'));
                modal.show();

                const result = await this.checkForUpdates();
                this.versionCheckResult = result;
                this.versionCheckState = result.status === 'update-available' ? 'update-available'
                    : result.status === 'error' ? 'error'
                    : 'up-to-date';
            },

            async copyPath(path, event) {
                const { showToast } = await import('./config.js');
                try {
                    await navigator.clipboard.writeText(path);
                    showToast('Copied to clipboard!', 'success', { delay: 1500 });
                    const btn = event?.currentTarget;
                    if (btn) {
                        const icon = btn.querySelector('i');
                        const originalClasses = icon?.className;
                        btn.classList.add('btn-success');
                        btn.classList.remove('btn-outline-secondary');
                        if (icon) icon.className = 'bi bi-check-lg';
                        setTimeout(() => {
                            btn.classList.remove('btn-success');
                            btn.classList.add('btn-outline-secondary');
                            if (icon && originalClasses) icon.className = originalClasses;
                        }, 1200);
                    }
                } catch {
                    showToast('Copy failed', 'danger');
                }
            },

            /**
             * Client-side path navigation.  Pushes the new URL into the
             * browser history and re-runs the route resolver in app.js
             * (exposed as ``window.__spaApplyRoute``).  Use this instead
             * of directly setting ``currentSection`` or mutating
             * ``window.location`` — the browser bar, back/forward
             * buttons, and deep links all stay in sync as a side effect.
             */
            navigateTo(path) {
                if (!path || typeof path !== 'string') return;
                if (window.location.pathname !== path) {
                    history.pushState({}, '', path);
                }
                if (typeof window.__spaApplyRoute === 'function') {
                    window.__spaApplyRoute();
                }
            },
            /**
             * Click handler for in-app <a> links that should route via
             * pushState on plain click but still honor
             * Cmd/Ctrl/Shift/middle-click so the browser can open the
             * link's real ``href`` in a new tab/window.  ``@click.prevent``
             * suppresses those modifier clicks too, which is why the
             * nav links + "Open" buttons use this instead.
             */
            navigateOnClick(event, path) {
                if (!event) return;
                if (event.defaultPrevented) return;
                if (event.button !== undefined && event.button !== 0) return;
                if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
                event.preventDefault();
                this.navigateTo(path);
            },
            showMonitoring() {
                this.navigateTo('/monitoring');
            },
            showAnalysisSection() {
                this.navigateTo('/analysis');
            },
            showSensorSection(sensorId) {
                if (!sensorId) return;
                this.navigateTo('/sensors/' + encodeURIComponent(sensorId));
            },
            showConfigSection() {
                // Close setup wizard modal if it was what opened us.
                const wizardModal = bootstrap.Modal.getInstance(document.getElementById('setupWizard'));
                if (wizardModal) {
                    wizardModal.hide();
                }
                this.navigateTo('/config');
            }
        });

        // Upgrade the browser tab title for sensor routes once the
        // WebSocket-driven ``sensors`` list populates and ``currentSensor``
        // resolves to a real enriched object.  ``applyRoute`` in app.js
        // seeds ``document.title`` with the raw sensor id on route entry;
        // this effect swaps it for the human-readable name when available.
        // Re-fires automatically whenever the route changes (sensor id or
        // back to a static section) or the sensor list refreshes.
        window.Alpine.effect(() => {
            const store = window.Alpine.store('citrasense');
            if (!store) return;
            if (store.currentSection !== 'sensor') return;
            const sensor = store.currentSensor;
            if (sensor?.name) {
                document.title = `${sensor.name} | CitraSense`;
            }
        });
    });
})();
