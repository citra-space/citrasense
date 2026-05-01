/**
 * CitraSense Alpine store - must register BEFORE Alpine starts.
 * Load this script before Alpine.js so the alpine:init listener is attached in time.
 */
import * as formatters from './formatters.js';
import * as components from './components/shared.js';
import { FILTER_COLORS } from './filters.js';
import { toggleMethods } from './store/toggles.js';
import { cameraMethods } from './store/camera.js';
import { versionCheckMethods } from './store/version-check.js';
import { clipboardMethods } from './store/clipboard.js';
import { radarMethods } from './store/radar.js';
import { hardwareControlMethods } from './store/hardware-control.js';
import { configMethods } from './store/config-methods.js';
import { calibrationPane } from './components/calibration.js';
import { radarControl } from './components/radar-control.js';
import { analysisTab } from './analysis.js';

(() => {
    document.addEventListener('alpine:init', () => {
        // Register Alpine components FIRST (before Alpine starts processing the DOM)
        window.Alpine.data('adapterField', components.adapterField);
        window.Alpine.data('taskRow', components.taskRow);
        window.Alpine.data('filterRow', components.filterRow);
        window.Alpine.data('logEntry', components.logEntry);
        window.Alpine.data('calibrationPane', calibrationPane);
        window.Alpine.data('radarControl', radarControl);
        window.Alpine.data('analysisTab', analysisTab);

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

            // Per-sensor slim-dict radar detections, capped to match the
            // server-side ring buffer (500 entries / ~50s at 10 Hz).
            // Populated by the ``radar_detection`` WebSocket message
            // and hydrated on page entry via ``hydrateRadarDetections``.
            //
            // Shape: ``{ [sensor_id]: Array<{
            //   ts, ts_unix, range_km, range_rate_km_s, doppler_hz,
            //   snr_db, sat_uuid, sat_name, sensor_id
            // }> }``.
            radarDetections: {},
            _radarDetectionsMax: 500,
            _radarDetectionsHydrated: {},

            ...radarMethods,

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
            sensorTypes: [], // [{value, label, description}] — populated from /api/sensor-types
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

            // Loading states for async operations.  ``isSavingConfig``
            // and ``isScanning`` are intrinsically site-wide.  Per-sensor
            // "is X happening" state lives in the maps below so one
            // sensor's Snap/Save/Loop/Autofocus doesn't flip the same
            // button on a different sensor's card.  Per-sensor reconnect
            // state lives on ``sensor.init_state`` (status feed), not a
            // store flag.
            isSavingConfig: false,
            isScanning: false,
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

            ...cameraMethods,

            ...toggleMethods,

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


            ...versionCheckMethods,
            ...clipboardMethods,
            ...hardwareControlMethods,
            ...configMethods,

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
