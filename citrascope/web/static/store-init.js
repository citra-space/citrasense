/**
 * CitraScope Alpine store - must register BEFORE Alpine starts.
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
        window.Alpine.store('citrascope', {
            status: {},
            tasks: [],
            logs: [],
            latestLog: null,
            wsConnected: false,
            wsReconnecting: false,
            wsReconnectAt: 0,
            wsLastMessage: 0,
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
            versionData: null,
            updateIndicator: '',
            versionCheckState: 'idle',
            versionCheckResult: null,

            // Autofocus target presets (loaded from API)
            autofocusPresets: [],

            // Loading states for async operations
            isSavingConfig: false,
            isScanning: false,
            isReconnecting: false,
            isCapturing: false,
            isSaving: false,
            isAutofocusing: false,
            captureResult: null,
            // Focus loop state
            isLooping: false,
            previewDataUrl: null,
            previewSource: '',
            loopCount: 0,
            previewExposure: 0.01,
            _lastTaskImageUrl: null,

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

            telescopeTooltip() {
                const s = this.status;
                if (!s?.telescope_connected) return 'Telescope disconnected';
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

            // Store methods
            previewFlipH: false,

            async captureImage() {
                const duration = this.previewExposure;
                if (Number.isNaN(duration) || duration <= 0) {
                    const { showToast } = await import('./config.js');
                    showToast('Invalid exposure duration', 'danger');
                    return;
                }

                this.isSaving = true;
                try {
                    const response = await fetch('/api/camera/capture', {
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
                    this.isSaving = false;
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

            async toggleObservingSession(enabled) {
                try {
                    const response = await fetch('/api/observing-session', {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ enabled: enabled })
                    });
                    const result = await response.json();
                    if (!response.ok) {
                        alert(result.error || 'Failed to toggle observing session');
                        this.status.observing_session_enabled = !enabled;
                    }
                } catch (error) {
                    console.error('Error toggling observing session:', error);
                    alert('Error toggling observing session');
                    this.status.observing_session_enabled = !enabled;
                }
            },

            async toggleSelfTasking(enabled) {
                try {
                    const response = await fetch('/api/self-tasking', {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ enabled: enabled })
                    });
                    const result = await response.json();
                    if (!response.ok) {
                        alert(result.error || 'Failed to toggle self-tasking');
                        this.status.self_tasking_enabled = !enabled;
                    }
                } catch (error) {
                    console.error('Error toggling self-tasking:', error);
                    alert('Error toggling self-tasking');
                    this.status.self_tasking_enabled = !enabled;
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
            get isImagingTaskActive() {
                return this.isSystemBusy || this.status?.processing_active === true;
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
                        const { showToast } = await import('./config.js');
                        showToast(data.error || 'Preview failed', 'danger');
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
                if (this.isLooping || this.isSystemBusy) return;
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
                        const { showToast } = await import('./config.js');
                        showToast(data.error || 'Preview failed', 'danger');
                    }
                } catch (error) {
                    const { showToast } = await import('./config.js');
                    showToast('Preview failed: ' + error.message, 'danger');
                } finally {
                    this.isCapturing = false;
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
                            `https://api.github.com/repos/citra-space/citrascope/compare/${gitHash}...main`
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

                    const githubResponse = await fetch('https://api.github.com/repos/citra-space/citrascope/releases/latest');
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
