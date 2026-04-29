// Store methods: clipboard copy with button animation

import { showToast } from '../toast.js';

export const clipboardMethods = {
    async copyPath(path, event) {
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
};
