// Configuration management for CitraSense

import { getConfig, saveConfig, getConfigStatus, getHardwareAdapters, getProcessors } from './api.js';
import { getFilterColor } from './filters.js';

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

    // Populate hardware adapter dropdown
    await loadAdapterOptions();

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

    if (sensor.adapter) {
        await loadAdapterSchema(sensor.adapter, sensor.adapter_settings || {});
    } else {
        store.adapterFields = [];
    }
    await loadFilterConfig();
    await loadProcessors();
}

/**
 * Check if configuration is needed and show setup wizard if not configured
 */
async function checkConfigStatus() {
    try {
        const status = await getConfigStatus();

        if (!status.configured) {
            // Show setup wizard if not configured
            const wizardModal = new bootstrap.Modal(document.getElementById('setupWizard'));
            wizardModal.show();
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

            // Normalize boolean fields that may come as null from backend
            if (config.file_logging_enabled === null || config.file_logging_enabled === undefined) {
                config.file_logging_enabled = true; // Default to true
            }
            if (config.keep_images === null || config.keep_images === undefined) {
                config.keep_images = false; // Default to false
            }
            if (config.processing_output_retention_hours === null || config.processing_output_retention_hours === undefined) {
                config.processing_output_retention_hours = 0;
            }
            if (config.use_dummy_api === null || config.use_dummy_api === undefined) {
                config.use_dummy_api = false; // Default to false
            }

            // Set configSensorId to the first sensor
            const firstSensor = config.sensors?.[0];
            store.configSensorId = firstSensor?.id || null;

            // Load autofocus presets before setting config so the select renders with options
            try {
                const presetsResp = await fetch(`${store.sensorApiBaseFor(store.configSensorId)}/autofocus/presets`);
                const presetsData = await presetsResp.json();
                store.autofocusPresets = presetsData.presets || [];
            } catch (e) {
                console.warn('Failed to load autofocus presets:', e);
                store.autofocusPresets = [];
            }

            store.config = config;
            if (store.previewExposure === null) {
                const firstSc = config.sensors?.[0] || {};
                store.previewExposure = firstSc.exposure_seconds || 2.0;
            }
            // Track per-sensor saved adapter for change detection
            for (const s of (config.sensors || [])) {
                store.savedAdapters[s.id] = s.adapter;
            }
            store.apiEndpoint =
                config.host === PROD_API_HOST ? 'production' :
                config.host === DEV_API_HOST ? 'development' : 'custom';

            // Load adapter-specific settings for the first sensor
            if (firstSensor?.adapter) {
                const currentAdapterSettings = firstSensor.adapter_settings || {};
                await loadAdapterSchema(firstSensor.adapter, currentAdapterSettings);
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

        const response = await fetch(`/api/hardware-adapters/${adapterName}/schema${settingsParam}`);
        const data = await response.json();

        console.log(`Schema API response:`, data);

        const enrichedFields = buildAdapterFields(data.schema || [], currentSettings);
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

    // Coerce per-sensor numeric fields before save
    const sc = store.config.sensors?.[sensorIndex];
    if (sc) {
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

const TOAST_ICONS = {
    success: 'bi-check-circle-fill',
    danger:  'bi-x-circle-fill',
    warning: 'bi-exclamation-triangle-fill',
    info:    'bi-info-circle-fill',
};

const TOAST_MAX_VISIBLE = 5;

/**
 * Show a toast notification.
 *
 * @param {string}  message          - Text to display.
 * @param {'success'|'danger'|'warning'|'info'} type - Bootstrap colour key.
 * @param {object}  [options]
 * @param {boolean} [options.autohide] - Override auto-hide (defaults: true for success/info, false for danger/warning).
 * @param {number}  [options.delay=5000] - Auto-hide delay in ms.
 * @param {string}  [options.id]       - Dedup key; skips if a toast with this id is already visible.
 */
export function showToast(message, type = 'info', { autohide, delay = 5000, id } = {}) {
    const toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        console.log(`Toast (${type}): ${message}`);
        return;
    }

    if (id && toastContainer.querySelector(`[data-toast-id="${id}"]`)) return;

    const shouldAutohide = autohide !== undefined ? autohide : (type === 'success' || type === 'info');
    const icon = TOAST_ICONS[type] || TOAST_ICONS.info;
    const toastElId = `toast-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    const countdownBar = shouldAutohide
        ? `<div class="toast-countdown" style="animation-duration:${delay}ms"></div>`
        : '';

    const html = `
        <div id="${toastElId}" class="toast align-items-center text-bg-${type} border-0"
             role="alert" aria-live="assertive" aria-atomic="true"
             ${id ? `data-toast-id="${id}"` : ''}>
            <div class="d-flex">
                <div class="toast-body d-flex align-items-center gap-2">
                    <i class="bi ${icon}"></i>
                    <span>${message}</span>
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto"
                        data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            ${countdownBar}
        </div>
    `;

    toastContainer.insertAdjacentHTML('beforeend', html);

    const toastElement = document.getElementById(toastElId);
    const toast = new bootstrap.Toast(toastElement, {
        autohide: shouldAutohide,
        delay,
    });

    toastElement.addEventListener('hidden.bs.toast', () => toastElement.remove());
    toast.show();

    // Enforce max-visible cap: dismiss oldest when over limit
    const visible = toastContainer.querySelectorAll('.toast.show');
    if (visible.length > TOAST_MAX_VISIBLE) {
        const oldest = visible[0];
        const oldToast = bootstrap.Toast.getInstance(oldest);
        if (oldToast) oldToast.hide();
    }
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
 * Show configuration section (called from setup wizard)
 */
export function showConfigSection() {
    // Close setup wizard modal
    const wizardModal = bootstrap.Modal.getInstance(document.getElementById('setupWizard'));
    if (wizardModal) {
        wizardModal.hide();
    }

    // Show config section
    const configLink = document.querySelector('a[data-section="config"]');
    if (configLink) {
        configLink.click();
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
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/filters`);

        if (response.status === 404 || response.status === 503) {
            store.filterConfigVisible = false;
            return;
        }

        const data = await response.json();

        if (response.ok && data.filters) {
            store.filterConfigVisible = true;
            store.filterNamesEditable = data.names_editable || false;
            store.filterNameOptions = data.filter_name_options || [];

            // Update enabled filters display on dashboard (Alpine store)
            updateStoreEnabledFilters(data.filters);

            // Add color to each filter and update store
            const filtersWithColor = {};
            Object.entries(data.filters).forEach(([id, filter]) => {
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

    // Send single batch update
    try {
        const response = await fetch(`${store.sensorApiBaseFor(store.configSensorId)}/filters/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(filterUpdates)
        });

        if (response.ok) {
            const data = await response.json();
            const successCount = data.updated_count || 0;

            // After batch update, sync to backend
            try {
                const syncResponse = await fetch(`${store.sensorApiBaseFor(store.configSensorId)}/filters/sync`, {
                    method: 'POST'
                });
                if (!syncResponse.ok) {
                    console.error('Failed to sync filters to backend');
                }
            } catch (error) {
                console.error('Error syncing filters to backend:', error);
            }

            return { success: successCount, failed: 0 };
        } else {
            const data = await response.json();
            const errorMsg = data.error || 'Unknown error';
            console.error(`Failed to save filters: ${errorMsg}`);

            // Show error to user
            if (response.status === 400 && errorMsg.includes('last enabled filter')) {
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
 * Trigger or cancel autofocus routine
 */
async function triggerAutofocus(sensorId) {
    const store = Alpine.store('citrasense');
    const ss = store.status?.sensors?.[sensorId] || {};
    const shouldCancel = ss?.autofocus_requested || ss?.autofocus_running;

    if (shouldCancel) {
        try {
            const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/autofocus/cancel`, {
                method: 'POST'
            });
            const data = await response.json();

            if (response.ok && data.success) {
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

    // Request autofocus
    store.isAutofocusing = true;

    try {
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/autofocus`, {
            method: 'POST'
        });

        const data = await response.json();

        if (response.ok) {
            showToast('Autofocus queued', 'success');
        } else {
            showToast(data.error || 'Autofocus request failed', 'danger');
        }
    } catch (error) {
        console.error('Error triggering autofocus:', error);
        showToast('Failed to trigger autofocus', 'danger');
    } finally {
        store.isAutofocusing = false;
    }
}

// Old private showToast removed — all callers now use the exported showToast above.

/**
 * Initialize filter configuration on page load
 */
export async function initFilterConfig() {
    // Load filter config when config section is visible
    await loadFilterConfig();
}

async function triggerAlignment(sensorId) {
    const store = Alpine.store('citrasense');
    const ss = store.status?.sensors?.[sensorId] || {};
    const isCancel = ss?.alignment_requested;

    if (isCancel) {
        try {
            const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/alignment/cancel`, { method: 'POST' });
            const data = await response.json();
            if (response.ok && data.success) {
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
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/alignment`, { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            showToast('Alignment queued', 'success');
        } else {
            showToast(data.error || 'Alignment request failed', 'danger');
        }
    } catch (error) {
        console.error('Error triggering alignment:', error);
        showToast('Failed to trigger alignment', 'danger');
    }
}

async function manualSync(sensorId, ra, dec) {
    if (ra === '' || ra == null || dec === '' || dec == null) {
        showToast('Enter RA and Dec values before syncing', 'warning');
        return;
    }

    try {
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/sync`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ra: parseFloat(ra), dec: parseFloat(dec) }),
        });
        const data = await response.json();
        if (response.ok && data.success) {
            showToast(data.message || 'Mount synced', 'success');
        } else {
            showToast(data.error || 'Sync failed', 'danger');
        }
    } catch (error) {
        console.error('Error syncing mount:', error);
        showToast('Failed to sync mount', 'danger');
    }
}

/**
 * Initiate mount homing routine.
 */
async function homeMount(sensorId) {
    const store = Alpine.store('citrasense');
    try {
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/mount/home`, { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            showToast('Mount homing initiated', 'success');
        } else {
            showToast(data.error || 'Failed to home mount', 'danger');
        }
    } catch (error) {
        console.error('Error homing mount:', error);
        showToast('Failed to home mount', 'danger');
    }
}

/**
 * Trigger cable unwind to resolve cable wrap buildup.
 */
async function triggerCableUnwind(sensorId) {
    const store = Alpine.store('citrasense');
    try {
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/mount/unwind`, { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            showToast('Cable unwind started — monitor progress in the Telescope card', 'success');
        } else {
            showToast(data.error || 'Cable unwind failed', 'danger');
        }
    } catch (error) {
        console.error('Error triggering cable unwind:', error);
        showToast('Failed to trigger cable unwind', 'danger');
    }
}

/**
 * Reset cable wrap counter to zero after operator verifies cables are straight.
 */
async function resetCableWrap(sensorId) {
    const store = Alpine.store('citrasense');
    try {
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/safety/cable-wrap/reset`, { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            showToast('Cable wrap counter reset to 0°', 'success');
        } else {
            showToast(data.error || 'Reset failed', 'danger');
        }
    } catch (error) {
        console.error('Error resetting cable wrap:', error);
        showToast('Failed to reset cable wrap', 'danger');
    }
}

/**
 * Emergency stop — halt mount, pause tasks, cancel in-flight imaging.
 */
async function emergencyStop() {
    try {
        const response = await fetch('/api/emergency-stop', { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            showToast(data.message || 'Emergency stop executed', 'warning');
        } else {
            showToast(data.error || 'Emergency stop failed', 'danger');
        }
    } catch (error) {
        console.error('Emergency stop error:', error);
        showToast('Failed to execute emergency stop', 'danger');
    }
}

/**
 * Clear the operator stop — allows motion to resume.
 */
async function clearOperatorStop() {
    try {
        const response = await fetch('/api/safety/operator-stop/clear', { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            showToast(data.message || 'Operator stop cleared', 'success');
        } else {
            showToast(data.error || 'Failed to clear operator stop', 'danger');
        }
    } catch (error) {
        console.error('Clear operator stop error:', error);
        showToast('Failed to clear operator stop', 'danger');
    }
}

/**
 * Setup autofocus/alignment button event listeners (call once during init)
 */
async function changeFilterPosition(sensorId, position) {
    const store = Alpine.store('citrasense');
    const wasLooping = store.isLooping;
    if (wasLooping) store.stopFocusLoop();

    try {
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/filter/set`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ position: parseInt(position) })
        });
        const data = await response.json();
        if (response.ok) {
            showToast(`Filter changed to ${data.name}`, 'success');
        } else {
            showToast(data.error || 'Filter change failed', 'danger');
        }
    } catch (error) {
        console.error('Filter change error:', error);
        showToast('Failed to change filter', 'danger');
    }

    if (wasLooping) store.startFocusLoop();
}

async function moveFocuserRelative(sensorId, steps) {
    const store = Alpine.store('citrasense');
    try {
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/focuser/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ relative: parseInt(steps) })
        });
        const data = await response.json();
        if (!response.ok) {
            showToast(data.error || 'Focuser move failed', 'danger');
        }
    } catch (error) {
        console.error('Focuser move error:', error);
        showToast('Failed to move focuser', 'danger');
    }
}

async function moveFocuserAbsolute(sensorId, position) {
    const store = Alpine.store('citrasense');
    try {
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/focuser/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ position: parseInt(position) })
        });
        const data = await response.json();
        if (response.ok) {
            showToast(`Focuser moved to ${data.position}`, 'success');
        } else {
            showToast(data.error || 'Focuser move failed', 'danger');
        }
    } catch (error) {
        console.error('Focuser move error:', error);
        showToast('Failed to move focuser', 'danger');
    }
}

async function abortFocuser(sensorId) {
    const store = Alpine.store('citrasense');
    try {
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/focuser/abort`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        if (response.ok) {
            showToast('Focuser stopped', 'warning');
        } else {
            showToast(data.error || 'Failed to stop focuser', 'danger');
        }
    } catch (error) {
        console.error('Focuser abort error:', error);
        showToast('Failed to stop focuser', 'danger');
    }
}

async function mountMove(sensorId, action, direction) {
    const store = Alpine.store('citrasense');
    try {
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/mount/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, direction })
        });
        if (!response.ok) {
            const data = await response.json();
            console.error('Mount move error:', data.error);
        }
    } catch (error) {
        console.error('Mount move error:', error);
    }
}

async function mountGoto(sensorId, ra, dec) {
    const store = Alpine.store('citrasense');
    try {
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/mount/goto`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ra: parseFloat(ra), dec: parseFloat(dec) })
        });
        const data = await response.json();
        if (response.ok) {
            showToast(`Slewing to RA=${parseFloat(ra).toFixed(2)}°, Dec=${parseFloat(dec).toFixed(2)}°`, 'info');
        } else {
            showToast(data.error || 'Goto failed', 'danger');
        }
    } catch (error) {
        console.error('Mount goto error:', error);
        showToast('Failed to send goto command', 'danger');
    }
}

async function mountSetTracking(sensorId, enabled) {
    const store = Alpine.store('citrasense');
    try {
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/mount/tracking`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled })
        });
        const data = await response.json();
        if (response.ok) {
            const store = Alpine.store('citrasense');
            const sd = store.status?.sensors?.[sensorId];
            if (sd) sd.mount_tracking = enabled;
            showToast(enabled ? 'Sidereal tracking started' : 'Tracking stopped', 'info');
        } else {
            showToast(data.error || 'Tracking command failed', 'danger');
        }
    } catch (error) {
        console.error('Mount tracking error:', error);
        showToast('Failed to change tracking', 'danger');
    }
}

async function calibratePointingModel(sensorId) {
    const store = Alpine.store('citrasense');
    try {
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/pointing-model/calibrate`, { method: 'POST' });
        const data = await response.json();
        if (response.ok && data.success) {
            showToast('Pointing calibration started', 'success');
        } else {
            showToast(data.error || 'Failed to start calibration', 'danger');
        }
    } catch (error) {
        console.error('Error starting pointing calibration:', error);
        showToast('Failed to start pointing calibration', 'danger');
    }
}

async function resetPointingModel(sensorId) {
    const store = Alpine.store('citrasense');
    try {
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/pointing-model/reset`, { method: 'POST' });
        const data = await response.json();
        if (response.ok && data.success) {
            showToast('Pointing model reset', 'success');
        } else {
            showToast(data.error || 'Failed to reset pointing model', 'danger');
        }
    } catch (error) {
        console.error('Error resetting pointing model:', error);
        showToast('Failed to reset pointing model', 'danger');
    }
}

async function cancelPointingCalibration(sensorId) {
    const store = Alpine.store('citrasense');
    try {
        const response = await fetch(`${store.sensorApiBaseFor(sensorId)}/pointing-model/calibrate/cancel`, { method: 'POST' });
        const data = await response.json();
        if (response.ok && data.success) {
            showToast('Pointing calibration cancelled', 'success');
        } else {
            showToast(data.error || 'Failed to cancel calibration', 'danger');
        }
    } catch (error) {
        console.error('Error cancelling pointing calibration:', error);
        showToast('Failed to cancel pointing calibration', 'danger');
    }
}

async function removeSensor(sid) {
    if (!confirm("Remove sensor '" + sid + "'? You will need to Save & Reload to finalize.")) return;
    try {
        const resp = await fetch('/api/config/sensors/' + encodeURIComponent(sid), { method: 'DELETE' });
        const data = await resp.json();
        if (!resp.ok) { alert(data.error || 'Failed to remove sensor'); return; }
        const store = Alpine.store('citrasense');
        const sensors = store.config.sensors;
        const idx = sensors.findIndex(s => s.id === sid);
        if (idx >= 0) sensors.splice(idx, 1);
        delete store.savedAdapters[sid];
        if (store.configSensorId === sid) {
            store.configSensorId = sensors[0]?.id || null;
        }
    } catch (e) {
        alert(e.message);
    }
}

export function setupAutofocusButton() {
    window.triggerAutofocus = triggerAutofocus;
    window.triggerAlignment = triggerAlignment;
    window.manualSync = manualSync;
    window.homeMount = homeMount;
    window.triggerCableUnwind = triggerCableUnwind;
    window.resetCableWrap = resetCableWrap;
    window.calibratePointingModel = calibratePointingModel;
    window.resetPointingModel = resetPointingModel;
    window.cancelPointingCalibration = cancelPointingCalibration;
    window.emergencyStop = emergencyStop;
    window.clearOperatorStop = clearOperatorStop;
    window._configRemoveSensor = removeSensor;
    window.changeFilterPosition = changeFilterPosition;
    window.moveFocuserRelative = moveFocuserRelative;
    window.moveFocuserAbsolute = moveFocuserAbsolute;
    window.abortFocuser = abortFocuser;
    window.mountMove = mountMove;
    window.mountGoto = mountGoto;
    window.mountSetTracking = mountSetTracking;
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

        const response = await fetch('/api/hardware/scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ adapter_name: adapter, current_settings: currentSettings }),
        });
        const data = await response.json();

        if (!response.ok) {
            showConfigError(data.error || 'Hardware scan failed');
            return;
        }

        store.adapterFields = buildAdapterFields(data.schema || [], currentSettings);
    } catch (error) {
        console.error('Hardware scan failed:', error);
        showConfigError('Hardware scan failed — check connection');
    } finally {
        store.isScanning = false;
    }
}
