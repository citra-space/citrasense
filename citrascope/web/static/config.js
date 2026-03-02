// Configuration management for CitraScope

import { getConfig, saveConfig, getConfigStatus, getHardwareAdapters, getProcessors } from './api.js';
import { getFilterColor } from './filters.js';

function updateStoreEnabledFilters(filters) {
    if (typeof Alpine !== 'undefined' && Alpine.store) {
        const enabled = Object.values(filters || {})
            .filter(f => f.enabled !== false)
            .map(f => ({ name: f.name, color: getFilterColor(f.name) }));
        Alpine.store('citrascope').enabledFilters = enabled;
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
    const store = Alpine.store('citrascope');
    if (adapter) {
        const allAdapterSettings = store.config.adapter_settings || {};
        const newAdapterSettings = allAdapterSettings[adapter] || {};

        console.log(`Switching to adapter: ${adapter}`);
        console.log('All adapter settings:', allAdapterSettings);
        console.log('Settings for this adapter:', newAdapterSettings);

        await loadAdapterSchema(adapter, newAdapterSettings);
        await loadFilterConfig();
    } else {
        store.adapterFields = [];
        store.filterConfigVisible = false;
    }
}

export async function initConfig() {
    // Attach config methods to Alpine store
    const store = Alpine.store('citrascope');
    store.handleAdapterChange = handleAdapterChange;
    store.reloadAdapterSchema = reloadAdapterSchema;

    // Populate hardware adapter dropdown
    await loadAdapterOptions();

    // Config form submission handled by Alpine @submit.prevent in template
    // Expose saveConfiguration for template access
    window.saveConfiguration = saveConfiguration;

    // Load initial config
    await loadConfiguration();

    // Load processor list (after config so we can sync enabled states)
    await loadProcessors();

    checkConfigStatus();
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
            Alpine.store('citrascope').hardwareAdapters = options;
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
        const processors = await getProcessors();

        if (typeof Alpine !== 'undefined' && Alpine.store) {
            const store = Alpine.store('citrascope');

            // Update processor enabled states from config
            const enabledProcessors = store.config?.enabled_processors || {};
            processors.forEach(proc => {
                // If config has an explicit setting, use it; otherwise use the default from API
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
            const store = Alpine.store('citrascope');

            // Normalize boolean fields that may come as null from backend
            if (config.file_logging_enabled === null || config.file_logging_enabled === undefined) {
                config.file_logging_enabled = true; // Default to true
            }
            if (config.keep_images === null || config.keep_images === undefined) {
                config.keep_images = false; // Default to false
            }
            if (config.scheduled_autofocus_enabled === null || config.scheduled_autofocus_enabled === undefined) {
                config.scheduled_autofocus_enabled = false; // Default to false
            }
            if (config.processors_enabled === null || config.processors_enabled === undefined) {
                config.processors_enabled = true; // Default to true
            }
            if (config.use_dummy_api === null || config.use_dummy_api === undefined) {
                config.use_dummy_api = false; // Default to false
            }
            if (!config.enabled_processors) {
                config.enabled_processors = {}; // Default to empty object
            }

            // Default autofocus target settings
            if (!config.autofocus_target_preset) {
                config.autofocus_target_preset = 'mirach';
            }

            // Load autofocus presets before setting config so the select renders with options
            try {
                const presetsResp = await fetch('/api/adapter/autofocus/presets');
                const presetsData = await presetsResp.json();
                store.autofocusPresets = presetsData.presets || [];
            } catch (e) {
                console.warn('Failed to load autofocus presets:', e);
                store.autofocusPresets = [];
            }

            store.config = config;
            store.savedAdapter = config.hardware_adapter; // Sync savedAdapter to store
            store.apiEndpoint =
                config.host === PROD_API_HOST ? 'production' :
                config.host === DEV_API_HOST ? 'development' : 'custom';

            // Load adapter-specific settings if adapter is selected
            if (config.hardware_adapter) {
                const allAdapterSettings = config.adapter_settings || {};
                const currentAdapterSettings = allAdapterSettings[config.hardware_adapter] || {};
                await loadAdapterSchema(config.hardware_adapter, currentAdapterSettings);
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
 * Load adapter schema and merge with current values
 */
async function loadAdapterSchema(adapterName, currentSettings = {}) {
    try {
        // Pass current adapter settings for dynamic schema generation
        const settingsParam = Object.keys(currentSettings).length > 0
            ? `?current_settings=${encodeURIComponent(JSON.stringify(currentSettings))}`
            : '';

        console.log(`Loading schema for ${adapterName} with settings:`, currentSettings);

        const response = await fetch(`/api/hardware-adapters/${adapterName}/schema${settingsParam}`);
        const data = await response.json();

        console.log(`Schema API response:`, data);

        const schema = data.schema || [];

        // Merge schema with values into enriched field objects
        // Use Alpine.reactive to ensure nested properties are reactive
        const enrichedFields = schema
            .filter(field => !field.readonly) // Skip readonly fields
            .map(field => Alpine.reactive({
                ...field,  // All schema properties (name, type, options, etc.)
                value: currentSettings[field.name] ?? field.default ?? getDefaultForType(field.type)
            }));

        console.log('Loaded adapter fields:', enrichedFields.map(f => `${f.name}=${f.value} (${f.type})`));

        // Update Alpine store with unified field objects
        if (typeof Alpine !== 'undefined' && Alpine.store) {
            const store = Alpine.store('citrascope');
            store.adapterFields = enrichedFields;
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
    const store = Alpine.store('citrascope');
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

    // Convert adapterFields back to flat settings object (for current adapter only)
    const adapterSettings = {};
    (store.adapterFields || []).forEach(field => {
        // Include all fields with defined values (including 0, false, empty string)
        if (field.value !== undefined && field.value !== null) {
            adapterSettings[field.name] = field.value;
        }
    });

    const config = {
        personal_access_token: formConfig.personal_access_token || '',
        telescope_id: formConfig.telescope_id || '',
        use_dummy_api: formConfig.use_dummy_api || false,
        hardware_adapter: formConfig.hardware_adapter || '',
        adapter_settings: adapterSettings, // Send flat settings for current adapter
        log_level: formConfig.log_level || 'INFO',
        keep_images: formConfig.keep_images || false,
        file_logging_enabled: formConfig.file_logging_enabled !== undefined ? formConfig.file_logging_enabled : true,
        scheduled_autofocus_enabled: formConfig.scheduled_autofocus_enabled || false,
        autofocus_interval_minutes: parseInt(formConfig.autofocus_interval_minutes || 60, 10),
        time_check_interval_minutes: parseInt(formConfig.time_check_interval_minutes || 5, 10),
        time_offset_pause_ms: parseFloat(formConfig.time_offset_pause_ms || 500),
        gps_location_updates_enabled: formConfig.gps_location_updates_enabled !== undefined ? formConfig.gps_location_updates_enabled : true,
        gps_update_interval_minutes: parseInt(formConfig.gps_update_interval_minutes || 5, 10),
        task_processing_paused: formConfig.task_processing_paused !== undefined ? formConfig.task_processing_paused : false,
        processors_enabled: formConfig.processors_enabled !== undefined ? formConfig.processors_enabled : true,
        enabled_processors: formConfig.enabled_processors || {},
        host,
        port,
        use_ssl,
        // Preserve other settings from Alpine store (the single source of truth)
        max_task_retries: store.config.max_task_retries || 3,
        initial_retry_delay_seconds: store.config.initial_retry_delay_seconds || 30,
        max_retry_delay_seconds: store.config.max_retry_delay_seconds || 300,
        log_retention_days: store.config.log_retention_days || 30,
        last_autofocus_timestamp: store.config.last_autofocus_timestamp,
        autofocus_target_preset: store.config.autofocus_target_preset || 'mirach',
        autofocus_target_custom_ra: store.config.autofocus_target_custom_ra,
        autofocus_target_custom_dec: store.config.autofocus_target_custom_dec,
        alignment_exposure_seconds: store.config.alignment_exposure_seconds || 2.0,
        align_on_startup: store.config.align_on_startup || false,
    };

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
            // Update saved adapter to match newly saved config
            if (typeof Alpine !== 'undefined' && Alpine.store) {
                Alpine.store('citrascope').savedAdapter = config.hardware_adapter;
            }

            // After config saved successfully, save any modified filter focus positions
            const filterResults = await saveModifiedFilters();

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

/**
 * Create and show a Bootstrap toast notification
 * @param {string} message - The message to display
 * @param {string} type - 'danger' for errors, 'success' for success messages
 * @param {boolean} autohide - Whether to auto-hide the toast
 */
export function createToast(message, type = 'danger', autohide = false) {
    const toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        console.error('Toast container not found');
        return;
    }

    // Create toast element
    const toastId = `toast-${Date.now()}`;
    const toastHTML = `
        <div id="${toastId}" class="toast text-bg-${type}" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header text-bg-${type}">
                <strong class="me-auto">CitraScope</strong>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;

    // Insert toast into container
    toastContainer.insertAdjacentHTML('beforeend', toastHTML);

    // Get the toast element and initialize Bootstrap toast
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, {
        autohide: autohide,
        delay: 5000
    });

    // Remove toast element from DOM after it's hidden
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });

    // Show the toast
    toast.show();
}

/**
 * Show configuration error message
 */
function showConfigError(message) {
    createToast(message, 'danger', false);
}

/**
 * Show configuration message (can be error or success)
 */
function showConfigMessage(message, type = 'danger') {
    if (type === 'danger') {
        showConfigError(message);
    } else {
        showConfigSuccess(message);
    }
}

/**
 * Show configuration success message
 */
function showConfigSuccess(message) {
    createToast(message, 'success', true);
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
async function loadFilterConfig() {
    const store = Alpine.store('citrascope');

    // Check if selected adapter matches saved adapter
    const selectedAdapter = store.config.hardware_adapter;

    if (selectedAdapter && store.savedAdapter && selectedAdapter !== store.savedAdapter) {
        // Adapter has changed but not saved yet - show message and hide table
        store.filterConfigVisible = true;
        store.filterAdapterChangeMessageVisible = true;
        return;
    }

    // Hide message and show table when adapters match
    store.filterAdapterChangeMessageVisible = false;

    try {
        const response = await fetch('/api/adapter/filters');

        if (response.status === 404 || response.status === 503) {
            store.filterConfigVisible = false;
            return;
        }

        const data = await response.json();

        if (response.ok && data.filters) {
            store.filterConfigVisible = true;

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
    const store = Alpine.store('citrascope');
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
        const focusPosition = parseInt(filter.focus_position);
        if (Number.isNaN(focusPosition) || focusPosition < 0) continue;

        filterUpdates.push({
            filter_id: filterId,
            focus_position: focusPosition,
            enabled: filter.enabled !== undefined ? filter.enabled : true
        });
    }

    if (filterUpdates.length === 0) return { success: 0, failed: 0 };

    // Send single batch update
    try {
        const response = await fetch('/api/adapter/filters/batch', {
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
                const syncResponse = await fetch('/api/adapter/filters/sync', {
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
async function triggerAutofocus() {
    const store = Alpine.store('citrascope');
    const isCancel = store.status?.autofocus_requested;

    if (isCancel) {
        // Cancel autofocus
        try {
            const response = await fetch('/api/adapter/autofocus/cancel', {
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
            showToast('Failed to cancel autofocus', 'error');
        }
        return;
    }

    // Request autofocus
    store.isAutofocusing = true;

    try {
        const response = await fetch('/api/adapter/autofocus', {
            method: 'POST'
        });

        const data = await response.json();

        if (response.ok) {
            showToast('Autofocus queued', 'success');
        } else {
            showToast(data.error || 'Autofocus request failed', 'error');
        }
    } catch (error) {
        console.error('Error triggering autofocus:', error);
        showToast('Failed to trigger autofocus', 'error');
    } finally {
        store.isAutofocusing = false;
    }
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    // Use Bootstrap toast if available, otherwise fallback to alert
    const toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        console.log(`Toast (${type}): ${message}`);
        return;
    }

    const toastId = `toast-${Date.now()}`;
    const bgClass = type === 'success' ? 'bg-success' :
                    type === 'error' ? 'bg-danger' :
                    type === 'warning' ? 'bg-warning' : 'bg-info';

    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center text-white ${bgClass} border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;

    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { delay: 3000 });
    toast.show();

    // Remove from DOM after hidden
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

/**
 * Initialize filter configuration on page load
 */
export async function initFilterConfig() {
    // Load filter config when config section is visible
    await loadFilterConfig();
}

async function triggerAlignment() {
    const store = Alpine.store('citrascope');
    const isCancel = store.status?.alignment_requested;

    if (isCancel) {
        try {
            const response = await fetch('/api/adapter/alignment/cancel', { method: 'POST' });
            const data = await response.json();
            if (response.ok && data.success) {
                showToast('Alignment cancelled', 'info');
            } else {
                showToast('Nothing to cancel', 'warning');
            }
        } catch (error) {
            console.error('Error cancelling alignment:', error);
            showToast('Failed to cancel alignment', 'error');
        }
        return;
    }

    try {
        const response = await fetch('/api/adapter/alignment', { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            showToast('Alignment queued', 'success');
        } else {
            showToast(data.error || 'Alignment request failed', 'error');
        }
    } catch (error) {
        console.error('Error triggering alignment:', error);
        showToast('Failed to trigger alignment', 'error');
    }
}

async function manualSync(ra, dec) {
    if (ra === '' || ra == null || dec === '' || dec == null) {
        showToast('Enter RA and Dec values before syncing', 'warning');
        return;
    }

    try {
        const response = await fetch('/api/adapter/sync', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ra: parseFloat(ra), dec: parseFloat(dec) }),
        });
        const data = await response.json();
        if (response.ok && data.success) {
            showToast(data.message || 'Mount synced', 'success');
        } else {
            showToast(data.error || 'Sync failed', 'error');
        }
    } catch (error) {
        console.error('Error syncing mount:', error);
        showToast('Failed to sync mount', 'error');
    }
}

/**
 * Initiate mount homing routine.
 */
async function homeMount() {
    try {
        const response = await fetch('/api/mount/home', { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            createToast('Mount homing initiated', 'success');
        } else {
            createToast(data.error || 'Failed to home mount', 'danger', false);
        }
    } catch (error) {
        console.error('Error homing mount:', error);
        createToast('Failed to home mount', 'danger', false);
    }
}

/**
 * Trigger cable unwind to resolve cable wrap buildup.
 */
async function triggerCableUnwind() {
    try {
        const response = await fetch('/api/mount/unwind', { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            createToast('Cable unwind started — monitor progress in the Telescope card', 'success');
        } else {
            createToast(data.error || 'Cable unwind failed', 'danger', false);
        }
    } catch (error) {
        console.error('Error triggering cable unwind:', error);
        createToast('Failed to trigger cable unwind', 'danger', false);
    }
}

/**
 * Reset cable wrap counter to zero after operator verifies cables are straight.
 */
async function resetCableWrap() {
    try {
        const response = await fetch('/api/safety/cable-wrap/reset', { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            createToast('Cable wrap counter reset to 0°', 'success');
        } else {
            createToast(data.error || 'Reset failed', 'danger', false);
        }
    } catch (error) {
        console.error('Error resetting cable wrap:', error);
        createToast('Failed to reset cable wrap', 'danger', false);
    }
}

/**
 * Emergency stop — halt mount, pause tasks, drain imaging queue.
 */
async function emergencyStop() {
    try {
        const response = await fetch('/api/emergency-stop', { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            createToast(data.message || 'Emergency stop executed', 'warning', false);
        } else {
            createToast(data.error || 'Emergency stop failed', 'danger', false);
        }
    } catch (error) {
        console.error('Emergency stop error:', error);
        createToast('Failed to execute emergency stop', 'danger', false);
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
            createToast(data.message || 'Operator stop cleared', 'success', true);
        } else {
            createToast(data.error || 'Failed to clear operator stop', 'danger', false);
        }
    } catch (error) {
        console.error('Clear operator stop error:', error);
        createToast('Failed to clear operator stop', 'danger', false);
    }
}

/**
 * Setup autofocus/alignment button event listeners (call once during init)
 */
export function setupAutofocusButton() {
    window.triggerAutofocus = triggerAutofocus;
    window.triggerAlignment = triggerAlignment;
    window.manualSync = manualSync;
    window.homeMount = homeMount;
    window.triggerCableUnwind = triggerCableUnwind;
    window.resetCableWrap = resetCableWrap;
    window.emergencyStop = emergencyStop;
    window.clearOperatorStop = clearOperatorStop;
}

/**
 * Reload adapter schema (called from Alpine components when device type changes)
 */
async function reloadAdapterSchema() {
    const store = Alpine.store('citrascope');
    const adapter = store.config.hardware_adapter;
    if (!adapter) return;

    // Convert current adapterFields back to flat settings object
    const currentSettings = {};
    (store.adapterFields || []).forEach(field => {
        currentSettings[field.name] = field.value;
    });

    await loadAdapterSchema(adapter, currentSettings);
}
