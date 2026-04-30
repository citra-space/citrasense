// Configuration management for CitraSense

import * as api from './api.js';
import { getFilterColor } from './filters.js';
import { showToast, showModal } from './toast.js';

const { getConfig, saveConfig, getConfigStatus, getHardwareAdapters, getProcessors } = api;

function updateStoreEnabledFilters(filters) {
    if (typeof Alpine !== 'undefined' && Alpine.store) {
        const enabled = Object.values(filters || {})
            .filter(f => f.enabled !== false)
            .map(f => ({ name: f.name, color: getFilterColor(f.name) }));
        Alpine.store('citrasense').enabledFilters = enabled;
    }
}

// API Host constants - must match backend constants in app.py
const PROD_API_HOST = 'api.citra.space';
const DEV_API_HOST = 'dev.api.citra.space';
const DEFAULT_API_PORT = 443;

/**
 * Handle adapter selection change (called from Alpine store)
 */
async function handleAdapterChange(adapter) {
    const store = Alpine.store('citrasense');
    if (adapter) {
        const sensorCfg = store.config.sensors?.[store.configSensorIndex] || {};
        const currentSettings = sensorCfg.adapter_settings || {};

        console.log(`Switching to adapter: ${adapter}`);
        console.log('Settings for this adapter:', currentSettings);

        await loadAdapterSchema(adapter, currentSettings);
        await loadFilterConfig();
    } else {
        store.adapterFields = [];
        store.filterConfigVisible = false;
    }
}

export async function initConfig() {
    // Attach config methods to Alpine store
    const store = Alpine.store('citrasense');
    store.handleAdapterChange = handleAdapterChange;
    store.reloadAdapterSchema = reloadAdapterSchema;
    store.scanHardware = scanHardware;

    // Populate hardware adapter + sensor type dropdowns used by the
    // Add Sensor form before the config finishes loading.
    await loadAdapterOptions();
    await loadSensorTypes();

    // Config form submission handled by Alpine @submit.prevent in template
    // Expose saveConfiguration for template access
    window.saveConfiguration = saveConfiguration;
    window.handleInvalidConfigField = handleInvalidConfigField;

    // Called by the Alpine store's configSensorId setter whenever the user
    // picks a different sensor in the config nav. Keeps adapterFields,
    // filters, and processor toggles in sync with the selected sensor so
    // saves don't accidentally merge the previous sensor's hardware
    // settings into the new sensor's adapter_settings.
    window._configOnSensorSwitch = onConfigSensorSwitch;

    // Load initial config
    await loadConfiguration();

    // Load processor list (after config so we can sync enabled states)
    await loadProcessors();

    checkConfigStatus();
}

async function onConfigSensorSwitch(sensorId) {
    const store = (typeof Alpine !== 'undefined' && Alpine.store)
        ? Alpine.store('citrasense')
        : null;
    if (!store || !sensorId) return;

    const sensor = store.config?.sensors?.find(s => s.id === sensorId);
    if (!sensor) return;

    await loadSensorSchema(sensor);
    await loadFilterConfig();
    await loadProcessors();
}

/**
 * Populate the Add Sensor type picker from the backend sensor-type
 * registry. Falls back to a telescope-only list if the endpoint
 * fails so existing sites keep working.
 */
async function loadSensorTypes() {
    const fallbackSensorTypes = [
        { value: 'telescope', label: 'Telescope', description: '' },
    ];
    try {
        const data = await api.getSensorTypes();
        if (!Array.isArray(data?.types)) {
            throw new Error('sensor-types response missing types[] array');
        }
        if (typeof Alpine !== 'undefined' && Alpine.store) {
            Alpine.store('citrasense').sensorTypes = data.types;
        }
    } catch (error) {
        console.error('Failed to load sensor types:', error);
        if (typeof Alpine !== 'undefined' && Alpine.store) {
            Alpine.store('citrasense').sensorTypes = fallbackSensorTypes;
        }
    }
}

/**
 * Fetch the adapter-settings schema for a sensor, routing on
 * ``sensor.type``:
 *   - ``telescope``: per-adapter classmethod via
 *     ``/api/hardware-adapters/{adapter}/schema``
 *   - anything else: class-level schema via
 *     ``/api/sensor-types/{type}/schema``
 *
 * Both paths feed the same ``buildAdapterFields`` →
 * ``$store.citrasense.adapterFields`` pipeline so the existing Hardware
 * tab renderer doesn't need to care which modality it's showing.
 */
export async function loadSensorSchema(sensor) {
    const store = Alpine.store('citrasense');
    if (!sensor) {
        store.adapterFields = [];
        return;
    }
    const currentSettings = sensor.adapter_settings || {};
    const sensorType = sensor.type || 'telescope';
    if (sensorType === 'telescope') {
        if (sensor.adapter) {
            await loadAdapterSchema(sensor.adapter, currentSettings);
        } else {
            store.adapterFields = [];
        }
        return;
    }
    try {
        const data = await api.getSensorTypeSchema(sensorType);
        store.adapterFields = buildAdapterFields(data?.schema || [], currentSettings);
    } catch (error) {
        console.error(`Failed to load ${sensorType} schema:`, error);
        showConfigError(`Failed to load settings for ${sensorType}`);
        store.adapterFields = [];
    }
}

/**
 * Check if configuration is needed and show setup wizard if not configured
 */
async function checkConfigStatus() {
    try {
        const status = await getConfigStatus();

        if (!status.configured) {
            showModal('setupWizard');
        } else if (status.error) {
            // Only show error toast if configured but there's an error (e.g., connection issue)
            // Don't show toast for "not configured" since modal already handles that
            showConfigError(status.error);
        }
    } catch (error) {
        console.error('Failed to check config status:', error);
    }
}

/**
 * Load available hardware adapters and populate dropdown
 */
async function loadAdapterOptions() {
    try {
        const data = await getHardwareAdapters();

        if (data.adapters && typeof Alpine !== 'undefined' && Alpine.store) {
            const options = data.adapters.map(adapterName => ({
                value: adapterName,
                label: data.descriptions[adapterName] || adapterName
            }));
            Alpine.store('citrasense').hardwareAdapters = options;
        }
    } catch (error) {
        console.error('Failed to load hardware adapters:', error);
    }
}

/**
 * Load available processors from API and sync with config
 */
async function loadProcessors() {
    try {
        const store = (typeof Alpine !== 'undefined' && Alpine.store)
            ? Alpine.store('citrasense')
            : null;
        const sensorId = store?.configSensorId || null;
        const processors = await getProcessors(sensorId);

        if (store) {
            // Per-sensor processor toggles live on sensor_config.enabled_processors
            const sensor = store.config?.sensors?.find(s => s.id === sensorId);
            const enabledProcessors = sensor?.enabled_processors || {};
            processors.forEach(proc => {
                if (proc.name in enabledProcessors) {
                    proc.enabled = enabledProcessors[proc.name];
                }
            });

            store.processors = processors;
        }
    } catch (error) {
        console.error('Failed to load processors:', error);
    }
}

/**
 * Load configuration from API and populate form
 */
async function loadConfiguration() {
    try {
        const config = await getConfig();

        // Sync to Alpine store - x-model handles form population
        if (typeof Alpine !== 'undefined' && Alpine.store) {
            const store = Alpine.store('citrasense');

            // Prefer the last-selected sensor (persisted in localStorage)
            // so reopening the Config tab lands back on the rig the
            // operator was editing.  Fall back to the first configured
            // sensor only when the persisted id is no longer valid or
            // nothing has been persisted yet.
            const persistedId = store.loadPersistedConfigSensorId?.();
            const persistedSensor = persistedId
                ? (config.sensors || []).find(s => s.id === persistedId)
                : null;
            const initialSensor = persistedSensor || config.sensors?.[0] || null;
            store.configSensorId = initialSensor?.id || null;

            // Load autofocus presets before setting config so the select renders with options
            try {
                const presetsResult = await api.getAutofocusPresets(store.configSensorId);
                store.autofocusPresets = presetsResult.ok ? (presetsResult.data?.presets || []) : [];
            } catch (e) {
                console.warn('Failed to load autofocus presets:', e);
                store.autofocusPresets = [];
            }

            store.config = config;
            for (const sc of (config.sensors || [])) {
                if (store.previewExposures[sc.id] == null) {
                    store.setPreviewExposure(sc.id, sc.exposure_seconds || 2.0);
                }
            }
            // Track per-sensor saved adapter for change detection
            for (const s of (config.sensors || [])) {
                store.savedAdapters[s.id] = s.adapter;
            }
            store.apiEndpoint =
                config.host === PROD_API_HOST ? 'production' :
                config.host === DEV_API_HOST ? 'development' : 'custom';

            // Load adapter/sensor-type settings for the initially selected sensor
            if (initialSensor) {
                await loadSensorSchema(initialSensor);
            }
        }
    } catch (error) {
        console.error('Failed to load config:', error);
    }
}

/**
 * Get default value for a field type
 */
function getDefaultForType(type) {
    if (type === 'bool') return false;
    if (type === 'int' || type === 'float') return 0;
    return '';
}

/**
 * Build reactive Alpine field objects from a raw schema + current values.
 */
function buildAdapterFields(schema, currentSettings = {}) {
    return schema
        .filter(field => !field.readonly)
        .map(field => Alpine.reactive({
            ...field,
            value: currentSettings[field.name] ?? field.default ?? getDefaultForType(field.type),
        }));
}

/**
 * Load adapter schema and merge with current values
 */
export async function loadAdapterSchema(adapterName, currentSettings = {}) {
    try {
        const settingsParam = Object.keys(currentSettings).length > 0
            ? `?current_settings=${encodeURIComponent(JSON.stringify(currentSettings))}`
            : '';

        console.log(`Loading schema for ${adapterName} with settings:`, currentSettings);

        const data = await api.getAdapterSchema(adapterName, settingsParam);

        console.log(`Schema API response:`, data);

        const enrichedFields = buildAdapterFields(data?.schema || [], currentSettings);
        console.log('Loaded adapter fields:', enrichedFields.map(f => `${f.name}=${f.value} (${f.type})`));

        if (typeof Alpine !== 'undefined' && Alpine.store) {
            Alpine.store('citrasense').adapterFields = enrichedFields;
        }
    } catch (error) {
        console.error('Failed to load adapter schema:', error);
        showConfigError(`Failed to load settings for ${adapterName}`);
    }
}

/**
 * Save configuration form handler
 */
async function saveConfiguration(event) {
    event.preventDefault();

    // Hide previous messages
    hideConfigMessages();

    // Get config from store (x-model keeps it up to date)
    const store = Alpine.store('citrasense');
    store.isSavingConfig = true;
    const formConfig = store.config;

    // Determine API host settings based on endpoint selection
    let host, port, use_ssl;
    if (store.apiEndpoint === 'production') {
        host = PROD_API_HOST;
        port = DEFAULT_API_PORT;
        use_ssl = true;
    } else if (store.apiEndpoint === 'development') {
        host = DEV_API_HOST;
        port = DEFAULT_API_PORT;
        use_ssl = true;
    } else {
        host = formConfig.host || '';
        port = formConfig.port || DEFAULT_API_PORT;
        use_ssl = formConfig.use_ssl !== undefined ? formConfig.use_ssl : true;
    }

    // Convert adapterFields back to flat settings for the selected sensor
    const adapterSettings = {};
    (store.adapterFields || []).forEach(field => {
        if (field.value !== undefined && field.value !== null) {
            adapterSettings[field.name] = field.value;
        }
    });

    // Inject adapter_settings into the selected sensor config
    const sensorIndex = store.configSensorIndex;
    if (store.config.sensors?.[sensorIndex]) {
        store.config.sensors[sensorIndex].adapter_settings = {
            ...store.config.sensors[sensorIndex].adapter_settings,
            ...adapterSettings,
        };
    }

    // Server-computed fields that should not be sent back on save
    const COMPUTED_FIELDS = [
        'app_url', 'config_file_path', 'log_file_path',
        'images_dir_path', 'processing_dir_path'
    ];

    // Coerce per-sensor numeric fields before save. These fields only
    // apply to the telescope imaging/plate-solve/autofocus pipeline;
    // streaming sensors (passive_radar) don't have them and would end
    // up with spurious zeros persisted to SensorConfig.
    const sc = store.config.sensors?.[sensorIndex];
    if (sc && (sc.type || 'telescope') === 'telescope') {
        sc.plate_solve_timeout = parseInt(sc.plate_solve_timeout || 60, 10);
        sc.sextractor_detect_thresh = parseFloat(sc.sextractor_detect_thresh || 5.0);
        sc.sextractor_detect_minarea = parseInt(sc.sextractor_detect_minarea || 3, 10);
        sc.autofocus_interval_minutes = parseInt(sc.autofocus_interval_minutes || 60, 10);
        sc.autofocus_after_sunset_offset_minutes = parseInt(sc.autofocus_after_sunset_offset_minutes ?? 60, 10);
    }

    const config = {
        ...store.config,
        host, port, use_ssl,
        time_check_interval_minutes: parseInt(store.config.time_check_interval_minutes || 5, 10),
        time_offset_pause_ms: parseFloat(store.config.time_offset_pause_ms || 500),
        gps_update_interval_minutes: parseInt(store.config.gps_update_interval_minutes || 5, 10),
    };

    COMPUTED_FIELDS.forEach(f => delete config[f]);
    // adapter_settings is now embedded in sensors[].adapter_settings
    delete config.adapter_settings;

    try {
        // Validate filters BEFORE saving main config
        const filters = store.filters || {};
        const filterCount = Object.keys(filters).length;
        if (filterCount > 0) {
            const enabledCount = Object.values(filters).filter(f => f.enabled).length;
            if (enabledCount === 0) {
                showConfigMessage('At least one filter must be enabled', 'danger');
                return;
            }
        }

        const result = await saveConfig(config);

        if (result.ok) {
            // Capture previous adapter before updating so we can detect switches
            const sensorId = store.configSensorId;
            const currentAdapter = store.config.sensors?.[sensorIndex]?.adapter;
            const previousAdapter = store.savedAdapters[sensorId];

            // Update saved adapter for this sensor
            if (sensorId && currentAdapter) {
                store.savedAdapters[sensorId] = currentAdapter;
            }

            // Only push filter changes when staying on the same adapter.
            const adapterChanged = currentAdapter !== previousAdapter;
            const filterResults = adapterChanged
                ? { success: 0, failed: 0 }
                : await saveModifiedFilters();

            // Build success message based on results
            let message = result.data.message || 'Configuration saved and applied successfully!';
            if (filterResults.success > 0) {
                message += ` Updated ${filterResults.success} filter focus position${filterResults.success > 1 ? 's' : ''}.`;
            }
            if (filterResults.failed > 0) {
                message += ` Warning: ${filterResults.failed} filter update${filterResults.failed > 1 ? 's' : ''} failed.`;
            }

            showConfigSuccess(message);

            // Reload filters to re-enable editing for the new adapter
            await loadFilterConfig();
        } else {
            // Check for specific error codes
            const errorMsg = result.data.error || result.data.message || 'Failed to save configuration';
            showConfigError(errorMsg);
        }
    } catch (error) {
        showConfigError('Failed to save configuration: ' + error.message);
    } finally {
        // Reset button state
        store.isSavingConfig = false;
    }
}

const CONFIG_TAB_LABELS = {
    api: 'API',
    hardware: 'Hardware',
    autofocus: 'Autofocus',
    calibration: 'Calibration',
    observation: 'Observation',
    processing: 'Processing',
    pipeline: 'Pipeline',
    timelocation: 'Time & Location',
    selftasking: 'Robotic Operations',
    advanced: 'Advanced',
};

let _invalidHandlerFiredThisPass = false;

/**
 * Handle an HTML5 `invalid` event from any input inside the config form.
 *
 * Browsers run implicit form validation before firing `submit`, so a bad
 * value on a tab the user isn't looking at silently blocks save with no
 * feedback beyond a Chrome devtools warning. This handler catches that
 * case, switches to the tab containing the invalid input, scrolls/focuses
 * it, and shows a toast naming the field.
 *
 * `invalid` doesn't bubble, so this must be registered with capture.
 * Many invalid events can fire in one validation pass (one per bad input);
 * we dedupe to the first.
 */
function handleInvalidConfigField(event, setTab) {
    if (_invalidHandlerFiredThisPass) return;
    _invalidHandlerFiredThisPass = true;
    requestAnimationFrame(() => { _invalidHandlerFiredThisPass = false; });

    const input = event.target;
    const tabContainer = input.closest('[x-show*="configTab ==="]');
    const tabMatch = tabContainer?.getAttribute('x-show')?.match(/configTab\s*===\s*['"]([^'"]+)['"]/);
    const tabKey = tabMatch?.[1];
    const tabLabel = tabKey ? (CONFIG_TAB_LABELS[tabKey] || tabKey) : null;
    const fieldLabel = input.labels?.[0]?.textContent?.trim()
        || input.getAttribute('aria-label')
        || input.name
        || input.placeholder
        || 'a required field';

    if (tabKey && typeof setTab === 'function') {
        setTab(tabKey);
        requestAnimationFrame(() => {
            try { input.scrollIntoView({ block: 'center', behavior: 'smooth' }); } catch (_) {}
            try { input.focus({ preventScroll: true }); } catch (_) {}
        });
    }

    const message = tabLabel
        ? `Please fix "${fieldLabel}" on the ${tabLabel} tab before saving.`
        : `Please fix "${fieldLabel}" before saving.`;
    showToast(message, 'warning');
}


function showConfigError(message) {
    showToast(message, 'danger');
}

function showConfigMessage(message, type = 'danger') {
    showToast(message, type === 'danger' ? 'danger' : 'success');
}

function showConfigSuccess(message) {
    showToast(message, 'success');
}

/**
 * Hide all configuration messages (no-op for toast compatibility)
 */
function hideConfigMessages() {
    // No-op - toasts handle their own hiding
}

/**
 * Show configuration section (called from setup wizard).  Delegates to
 * the Alpine store's ``showConfigSection`` so both entry points push the
 * same ``/config`` URL into history instead of faking a nav click.
 */
export function showConfigSection() {
    const store = window.Alpine?.store?.('citrasense');
    if (store && typeof store.showConfigSection === 'function') {
        store.showConfigSection();
        return;
    }
    const wizardModal = bootstrap.Modal.getInstance(document.getElementById('setupWizard'));
    if (wizardModal) wizardModal.hide();
    history.pushState({}, '', '/config');
    if (typeof window.__spaApplyRoute === 'function') {
        window.__spaApplyRoute();
    }
}

/**
 * Load and display filter configuration
 */
export async function loadFilterConfig() {
    const store = Alpine.store('citrasense');

    // Check if selected adapter matches saved adapter for this sensor
    const sensorId = store.configSensorId;
    const selectedAdapter = store.config.sensors?.[store.configSensorIndex]?.adapter;
    const savedAdapter = store.savedAdapters[sensorId];

    if (selectedAdapter && savedAdapter && selectedAdapter !== savedAdapter) {
        store.filterConfigVisible = true;
        store.filterAdapterChangeMessageVisible = true;
        return;
    }

    store.filterAdapterChangeMessageVisible = false;

    try {
        const result = await api.getFilters(sensorId);

        if (result.status === 404 || result.status === 503) {
            store.filterConfigVisible = false;
            return;
        }

        if (result.ok && result.data?.filters) {
            store.filterConfigVisible = true;
            store.filterNamesEditable = result.data.names_editable || false;
            store.filterNameOptions = result.data.filter_name_options || [];

            updateStoreEnabledFilters(result.data.filters);

            const filtersWithColor = {};
            Object.entries(result.data.filters).forEach(([id, filter]) => {
                filtersWithColor[id] = {
                    ...filter,
                    color: getFilterColor(filter.name),
                    enabled: filter.enabled !== undefined ? filter.enabled : true
                };
            });
            store.filters = filtersWithColor;
        } else {
            store.filterConfigVisible = false;
        }
    } catch (error) {
        console.error('Error loading filter config:', error);
        store.filterConfigVisible = false;
    }
}

/**
 * Save all filter focus positions and enabled states (called during main config save)
 * Returns: Object with { success: number, failed: number }
 */
async function saveModifiedFilters() {
    const store = Alpine.store('citrasense');
    const filters = store.filters || {};
    const filterIds = Object.keys(filters);

    if (filterIds.length === 0) return { success: 0, failed: 0 };

    // Validate at least one filter is enabled
    const enabledCount = Object.values(filters).filter(f => f.enabled).length;
    if (enabledCount === 0) {
        showConfigMessage('At least one filter must be enabled', 'danger');
        return { success: 0, failed: filterIds.length };
    }

    // Collect all filter updates from store
    const filterUpdates = [];
    for (const [filterId, filter] of Object.entries(filters)) {
        const rawFocus = parseInt(filter.focus_position);
        const focusPosition = (Number.isNaN(rawFocus) || rawFocus < 0) ? null : rawFocus;

        const update = {
            filter_id: filterId,
            focus_position: focusPosition,
            enabled: filter.enabled !== undefined ? filter.enabled : true
        };
        if (store.filterNamesEditable && filter.name) {
            update.name = filter.name;
        }
        filterUpdates.push(update);
    }

    if (filterUpdates.length === 0) return { success: 0, failed: 0 };

    try {
        const result = await api.updateFiltersBatch(store.configSensorId, filterUpdates);

        if (result.ok) {
            const successCount = result.data?.updated_count || 0;

            try {
                const syncResult = await api.syncFilters(store.configSensorId);
                if (!syncResult.ok) {
                    console.error('Failed to sync filters to backend');
                }
            } catch (error) {
                console.error('Error syncing filters to backend:', error);
            }

            return { success: successCount, failed: 0 };
        } else {
            const errorMsg = result.error || 'Unknown error';
            console.error(`Failed to save filters: ${errorMsg}`);

            if (result.status === 400 && errorMsg.includes('last enabled filter')) {
                showConfigMessage(errorMsg, 'danger');
            }

            return { success: 0, failed: filterUpdates.length };
        }
    } catch (error) {
        console.error('Error saving filters:', error);
        return { success: 0, failed: filterUpdates.length };
    }
}

/**
 * Initialize filter configuration on page load
 */
export async function initFilterConfig() {
    await loadFilterConfig();
}

/**
 * Reload adapter schema (called from Alpine components when device type changes)
 */
async function reloadAdapterSchema() {
    const store = Alpine.store('citrasense');
    const adapter = store.config.sensors?.[store.configSensorIndex]?.adapter;
    if (!adapter) return;

    // Convert current adapterFields back to flat settings object
    const currentSettings = {};
    (store.adapterFields || []).forEach(field => {
        currentSettings[field.name] = field.value;
    });

    await loadAdapterSchema(adapter, currentSettings);
}

/**
 * Clear hardware probe caches and re-enumerate devices.
 * Called from the "Scan Hardware" button.
 */
async function scanHardware() {
    const store = Alpine.store('citrasense');
    const adapter = store.config.sensors?.[store.configSensorIndex]?.adapter;
    if (!adapter) return;

    store.isScanning = true;
    try {
        const currentSettings = {};
        (store.adapterFields || []).forEach(field => {
            currentSettings[field.name] = field.value;
        });

        const result = await api.scanHardware(adapter, currentSettings);

        if (!result.ok) {
            showConfigError(result.error || 'Hardware scan failed');
            return;
        }

        store.adapterFields = buildAdapterFields(result.data?.schema || [], currentSettings);
    } catch (error) {
        console.error('Hardware scan failed:', error);
        showConfigError('Hardware scan failed — check connection');
    } finally {
        store.isScanning = false;
    }
}
