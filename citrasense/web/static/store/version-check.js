// Store methods: version check and update indicator

import { showModal } from '../toast.js';
import * as api from '../api.js';

export const versionCheckMethods = {
    async checkForUpdates() {
        try {
            const result = await api.checkForUpdates();
            if (!result.ok) {
                this.updateIndicator = '';
                return { status: 'error', currentVersion: 'unknown' };
            }
            const data = result.data;
            this.versionData = data;

            if (data.status === 'update-available') {
                if (data.behind_by) {
                    this.updateIndicator = `${data.behind_by} commit${data.behind_by !== 1 ? 's' : ''} behind`;
                } else if (data.latest_version) {
                    this.updateIndicator = `${data.latest_version} Available!`;
                }
            } else {
                this.updateIndicator = '';
            }
            return data;
        } catch (error) {
            console.debug('Update check failed:', error);
            this.updateIndicator = '';
            return { status: 'error', currentVersion: 'unknown' };
        }
    },

    async showVersionModal() {
        this.versionCheckState = 'loading';
        this.versionCheckResult = null;

        showModal('versionModal');

        const result = await this.checkForUpdates();
        this.versionCheckResult = result;
        this.versionCheckState = result.status === 'update-available' ? 'update-available'
            : result.status === 'error' ? 'error'
            : 'up-to-date';
    },
};
