/**
 * CitraScope Alpine store - must register BEFORE Alpine starts.
 * Load this script before Alpine.js so the alpine:init listener is attached in time.
 */
import * as formatters from './formatters.js';
import * as components from './components.js';
import { FILTER_COLORS } from './filters.js';

(() => {
    document.addEventListener('alpine:init', () => {
        // Register Alpine components FIRST (before Alpine starts processing the DOM)
        window.Alpine.data('adapterField', components.adapterField);
        window.Alpine.data('taskRow', components.taskRow);
        window.Alpine.data('filterRow', components.filterRow);
        window.Alpine.data('logEntry', components.logEntry);

        // Register store
        window.Alpine.store('citrascope', {
            status: {},
            tasks: [],
            logs: [],
            latestLog: null,
            wsConnected: false,
            wsReconnecting: false,
            currentTaskId: null,
            isTaskActive: false,
            nextTaskStartTime: null,
            countdown: '',
            config: {},
            apiEndpoint: 'production',
            hardwareAdapters: [], // [{value, label}]
            filters: {},
            savedAdapter: null,
            enabledFilters: [],
            filterConfigVisible: false,
            filterNamesEditable: false,
            filterNameOptions: [],
            filterColors: FILTER_COLORS,
            filterAdapterChangeMessageVisible: false,
            currentSection: 'monitoring',
            version: '',
            updateIndicator: '',
            versionCheckState: 'idle',
            versionCheckResult: null,

            // Autofocus target presets (loaded from API)
            autofocusPresets: [],

            // Loading states for async operations
            isSavingConfig: false,
            isCapturing: false,
            isAutofocusing: false,
            captureResult: null,
            // Focus loop state
            isLooping: false,
            previewDataUrl: null,
            loopCount: 0,
            previewExposure: 0.01,

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

            // Store methods
            previewFlipH: false,

            async captureImage() {
                const duration = this.previewExposure;
                if (Number.isNaN(duration) || duration <= 0) {
                    const { createToast } = await import('./config.js');
                    createToast('Invalid exposure duration', 'danger', false);
                    return;
                }

                this.isCapturing = true;
                try {
                    const response = await fetch('/api/camera/capture', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ duration })
                    });
                    const data = await response.json();

                    if (response.ok && data.success) {
                        this.captureResult = data;
                        const { createToast } = await import('./config.js');
                        createToast('Image captured successfully', 'success', true);
                    } else {
                        const { createToast } = await import('./config.js');
                        createToast(data.error || 'Failed to capture image', 'danger', false);
                    }
                } catch (error) {
                    console.error('Capture error:', error);
                    const { createToast } = await import('./config.js');
                    createToast('Failed to capture image: ' + error.message, 'danger', false);
                } finally {
                    this.isCapturing = false;
                }
            },

            async toggleProcessing(enabled) {
                const endpoint = enabled ? '/api/tasks/resume' : '/api/tasks/pause';
                try {
                    const response = await fetch(endpoint, { method: 'POST' });
                    const result = await response.json();
                    if (!response.ok) {
                        alert(result.error || 'Failed to toggle task processing');
                        // Revert on error
                        this.status.processing_active = !enabled;
                    }
                } catch (error) {
                    console.error('Error toggling processing:', error);
                    alert('Error toggling task processing');
                    this.status.processing_active = !enabled;
                }
            },

            async toggleAutomatedScheduling(enabled) {
                try {
                    const response = await fetch('/api/telescope/automated-scheduling', {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ enabled: enabled })
                    });
                    const result = await response.json();
                    if (!response.ok) {
                        alert(result.error || 'Failed to toggle automated scheduling');
                        // Revert on error
                        this.status.automated_scheduling = !enabled;
                    }
                } catch (error) {
                    console.error('Error toggling automated scheduling:', error);
                    alert('Error toggling automated scheduling');
                    this.status.automated_scheduling = !enabled;
                }
            },

            get isImagingTaskActive() {
                return this.status?.processing_active === true;
            },

            async capturePreview() {
                if (this.isImagingTaskActive) {
                    this.isLooping = false;
                    return;
                }
                try {
                    const response = await fetch('/api/camera/preview', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ duration: this.previewExposure, flip_horizontal: this.previewFlipH })
                    });
                    if (response.status === 409) {
                        // Camera busy with previous capture — wait and retry
                        if (this.isLooping) {
                            setTimeout(() => this.capturePreview(), 250);
                        }
                        return;
                    }
                    const data = await response.json();
                    if (response.ok && data.image_data) {
                        this.previewDataUrl = data.image_data;
                        this.loopCount++;
                    } else {
                        const { createToast } = await import('./config.js');
                        createToast(data.error || 'Preview failed', 'danger', false);
                        this.isLooping = false;
                        return;
                    }
                } catch (error) {
                    console.error('Preview error:', error);
                    this.isLooping = false;
                    return;
                }

                if (this.isLooping) {
                    requestAnimationFrame(() => this.capturePreview());
                }
            },

            startFocusLoop() {
                if (this.isLooping || this.isImagingTaskActive) return;
                this.isLooping = true;
                this.loopCount = 0;
                this.capturePreview();
            },

            stopFocusLoop() {
                this.isLooping = false;
            },

            async singlePreview() {
                if (this.isLooping) return;
                this.isCapturing = true;
                try {
                    const response = await fetch('/api/camera/preview', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ duration: this.previewExposure, flip_horizontal: this.previewFlipH })
                    });
                    const data = await response.json();
                    if (response.ok && data.image_data) {
                        this.previewDataUrl = data.image_data;
                        this.loopCount++;
                    } else {
                        const { createToast } = await import('./config.js');
                        createToast(data.error || 'Preview failed', 'danger', false);
                    }
                } catch (error) {
                    const { createToast } = await import('./config.js');
                    createToast('Preview failed: ' + error.message, 'danger', false);
                } finally {
                    this.isCapturing = false;
                }
            },

            async showVersionModal() {
                this.versionCheckState = 'loading';
                this.versionCheckResult = null;

                const modal = new bootstrap.Modal(document.getElementById('versionModal'));
                modal.show();

                // Check for updates (inline implementation)
                try {
                    const versionResponse = await fetch('/api/version');
                    const versionData = await versionResponse.json();
                    const currentVersion = versionData.version;

                    const githubResponse = await fetch('https://api.github.com/repos/citra-space/citrascope/releases/latest');
                    if (!githubResponse.ok) {
                        this.versionCheckState = 'error';
                        this.versionCheckResult = { status: 'error', currentVersion };
                        return;
                    }

                    const releaseData = await githubResponse.json();
                    const latestVersion = releaseData.tag_name.replace(/^v/, '');
                    const releaseUrl = releaseData.html_url;

                    if (currentVersion === 'development' || currentVersion === 'unknown') {
                        this.updateIndicator = '';
                        this.versionCheckState = 'up-to-date';
                        this.versionCheckResult = { status: 'up-to-date', currentVersion };
                        return;
                    }

                    // Compare versions
                    const v1 = latestVersion.split('.').map(n => parseInt(n) || 0);
                    const v2 = currentVersion.split('.').map(n => parseInt(n) || 0);
                    const maxLen = Math.max(v1.length, v2.length);
                    let comparison = 0;
                    for (let i = 0; i < maxLen; i++) {
                        const num1 = v1[i] || 0;
                        const num2 = v2[i] || 0;
                        if (num1 > num2) { comparison = 1; break; }
                        if (num1 < num2) { comparison = -1; break; }
                    }

                    if (comparison > 0) {
                        this.updateIndicator = `${latestVersion} Available!`;
                        this.versionCheckState = 'update-available';
                        this.versionCheckResult = { status: 'update-available', currentVersion, latestVersion, releaseUrl };
                    } else {
                        this.updateIndicator = '';
                        this.versionCheckState = 'up-to-date';
                        this.versionCheckResult = { status: 'up-to-date', currentVersion };
                    }
                } catch (error) {
                    console.debug('Update check failed:', error);
                    this.versionCheckState = 'error';
                    this.versionCheckResult = { status: 'error', currentVersion: 'unknown' };
                }
            },

            showConfigSection() {
                // Close setup wizard modal
                const wizardModal = bootstrap.Modal.getInstance(document.getElementById('setupWizard'));
                if (wizardModal) {
                    wizardModal.hide();
                }

                // Navigate to config section
                this.currentSection = 'config';
                window.location.hash = 'config';
            }
        });
    });
})();
