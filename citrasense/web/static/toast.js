// Toast notification module for CitraSense

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

/**
 * Show or get-or-create a Bootstrap modal by element ID.
 */
export function showModal(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const modal = bootstrap.Modal.getOrCreateInstance(el);
    modal.show();
}
