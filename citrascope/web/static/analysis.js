/**
 * Analysis tab — fetches pipeline metrics and renders the task table / stats.
 */

const PROCESSOR_COLORS = {
    calibration_s:        '#00b894',
    plate_solve_s:        '#fdcb6e',
    source_extractor_s:   '#e17055',
    photometry_s:         '#74b9ff',
    satellite_matcher_s:  '#a29bfe',
    annotated_image_s:    '#636e72',
};

const PROCESSOR_LABELS = {
    calibration_s:        'Cal',
    plate_solve_s:        'Solve',
    source_extractor_s:   'SExtract',
    photometry_s:         'Phot',
    satellite_matcher_s:  'Match',
    annotated_image_s:    'Annotate',
};

document.addEventListener('alpine:init', () => {
    Alpine.data('analysisTab', () => ({
        stats: {},
        tasks: [],
        total: 0,
        offset: 0,
        pageSize: 50,
        sortCol: 'completed_at',
        sortOrder: 'desc',
        filterTarget: '',
        filterDateFrom: '',
        filterDateTo: '',
        filterFilterName: '',
        filterSolved: '',
        filterMatchDetail: '',
        filterWindow: '',
        filterUploadStatus: '',
        loading: false,
        loaded: false,
        expanded: null,
        detail: null,
        _extractedData: {},
        _satObs: [],
        lightboxSrc: null,
        _refreshTimer: null,

        // Reprocess (single) — form values and original-run values
        reprocessThresh: 5.0,
        reprocessMinarea: 3,
        reprocessFilter: 'default',
        origThresh: null,
        origMinarea: null,
        origFilter: null,
        reprocessing: false,
        reprocessResult: null,
        reprocessError: null,
        uploadingReprocessed: false,
        uploadReprocessedOk: false,
        uploadReprocessedError: null,

        // Multi-select
        selectedTasks: {},
        batchThresh: 5.0,
        batchMinarea: 3,
        batchFilter: 'default',
        batchJobId: null,
        batchProgress: 0,
        batchTotal: 0,
        batchResult: null,
        _batchPollTimer: null,

        // Auto-tune
        autotuneOpen: false,
        autotuneJobId: null,
        autotuneProgress: 0,
        autotuneTotal: 0,
        autotuneResult: null,
        autotuneRunning: false,
        _autotunePollTimer: null,

        init() {
            this._refreshTimer = setInterval(() => {
                const store = Alpine.store('citrascope');
                if (store && store.currentSection === 'analysis' && this.loaded && !this.loading) {
                    this.loadAll();
                }
            }, 15_000);
        },

        async loadAll() {
            if (this.loading) return;
            this.loading = true;
            try {
                await Promise.all([this.loadStats(), this.loadTasks()]);
                this.loaded = true;
            } finally {
                this.loading = false;
            }
        },

        async loadStats() {
            try {
                const resp = await fetch('/api/analysis/stats?hours=24');
                this.stats = await resp.json();
            } catch (e) {
                console.error('Failed to load analysis stats', e);
            }
        },

        async loadTasks() {
            this.selectedTasks = {};
            try {
                const params = new URLSearchParams({
                    limit: this.pageSize,
                    offset: this.offset,
                    sort: this.sortCol,
                    order: this.sortOrder,
                });
                if (this.filterTarget) params.set('target_name', this.filterTarget);
                if (this.filterDateFrom) params.set('date_from', this.filterDateFrom + 'T00:00:00+00:00');
                if (this.filterDateTo) params.set('date_to', this.filterDateTo + 'T23:59:59+00:00');
                if (this.filterFilterName) params.set('filter_name', this.filterFilterName);
                if (this.filterSolved) params.set('plate_solved', this.filterSolved);
                if (this.filterMatchDetail) params.set('match_detail', this.filterMatchDetail);
                if (this.filterWindow) params.set('missed_window', this.filterWindow);
                if (this.filterUploadStatus) params.set('upload_status', this.filterUploadStatus);

                const resp = await fetch('/api/analysis/tasks?' + params.toString());
                const data = await resp.json();
                this.tasks = data.tasks || [];
                this.total = data.total || 0;
            } catch (e) {
                console.error('Failed to load analysis tasks', e);
            }
        },

        toggleSort(col) {
            if (this.sortCol === col) {
                this.sortOrder = this.sortOrder === 'desc' ? 'asc' : 'desc';
            } else {
                this.sortCol = col;
                this.sortOrder = 'desc';
            }
            this.offset = 0;
            this.loadTasks();
        },

        sortIcon(col) {
            if (this.sortCol !== col) return '';
            return this.sortOrder === 'desc' ? '▾' : '▴';
        },

        _resetReprocessState() {
            this.reprocessResult = null;
            this.reprocessError = null;
            this.origThresh = null;
            this.origMinarea = null;
            this.origFilter = null;
            this.reprocessThresh = 5.0;
            this.reprocessMinarea = 3;
            this.reprocessFilter = 'default';
            this.uploadingReprocessed = false;
            this.uploadReprocessedOk = false;
            this.uploadReprocessedError = null;
        },

        async toggleDetail(taskId) {
            if (this.expanded === taskId) {
                this.expanded = null;
                this.detail = null;
                this._extractedData = {};
                this._satObs = [];
                this._resetReprocessState();
                return;
            }

            // Immediately clear stale detail so the old result isn't
            // shown under the newly-clicked row while the fetch runs.
            this.expanded = null;
            this.detail = null;
            this._resetReprocessState();

            try {
                const resp = await fetch('/api/analysis/tasks/' + taskId);
                this.detail = await resp.json();
                try {
                    this._extractedData = JSON.parse(this.detail.extracted_data_json || '{}');
                } catch { this._extractedData = {}; }
                try {
                    this._satObs = JSON.parse(this.detail.satellite_observations_json || '[]');
                } catch { this._satObs = []; }

                // Pre-populate reprocess form from the original run's settings
                const origT = this._extractedData['source_extractor.detect_thresh'];
                const origM = this._extractedData['source_extractor.detect_minarea'];
                const origF = this._extractedData['source_extractor.filter_name'];
                this.origThresh = origT ?? null;
                this.origMinarea = origM ?? null;
                this.origFilter = origF ?? null;
                this.reprocessThresh = origT ?? 5.0;
                this.reprocessMinarea = origM ?? 3;
                this.reprocessFilter = origF ?? 'default';

                if (this.detail.reprocessed_result) {
                    const rr = this.detail.reprocessed_result;
                    const ed = rr.extracted_data || {};
                    rr._usedThresh = ed['source_extractor.detect_thresh'] ?? null;
                    rr._usedMinarea = ed['source_extractor.detect_minarea'] ?? null;
                    rr._usedFilter = ed['source_extractor.filter_name'] ?? null;
                    this.reprocessResult = rr;
                }

                this.expanded = taskId;
            } catch (e) {
                console.error('Failed to load task detail', e);
            }
        },

        formatTime(iso) {
            if (!iso) return '—';
            try {
                const d = new Date(iso);
                return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                    + ' ' + d.toLocaleDateString([], { month: 'short', day: 'numeric' });
            } catch { return iso; }
        },

        pointingClass(deg) {
            return Alpine.store('citrascope').pointingAccuracyClass(deg, Alpine.store('citrascope').status?.fov_short_deg);
        },

        // Window utilization mini-bar: delay | slew | imaging | margin
        _windowSpan(t) {
            if (!t.window_start || !t.window_stop) return 0;
            return (new Date(t.window_stop) - new Date(t.window_start)) / 1000;
        },
        _windowSegments(t) {
            const span = this._windowSpan(t);
            if (span <= 0 || !t.window_start) return { delay: 0, slew: 0, imaging: 0, margin: 0, overran: false };
            const ws = new Date(t.window_start).getTime();
            const we = new Date(t.window_stop).getTime();
            const slewStart = t.slew_started_at ? new Date(t.slew_started_at).getTime() : null;
            const imgStart = t.imaging_started_at ? new Date(t.imaging_started_at).getTime() : null;
            const imgEnd = t.imaging_finished_at ? new Date(t.imaging_finished_at).getTime() : null;

            const delay = slewStart ? Math.max(0, slewStart - ws) / 1000 : 0;
            const slew = (slewStart && imgStart) ? Math.max(0, imgStart - slewStart) / 1000 : (t.total_slew_time_s || 0);
            const imaging = (imgStart && imgEnd) ? Math.max(0, imgEnd - imgStart) / 1000 : 0;
            const usedEnd = imgEnd || (imgStart ? imgStart : (slewStart ? slewStart : ws));
            const margin = Math.max(0, we - usedEnd) / 1000;
            const overran = imgEnd ? imgEnd > we : false;

            return { delay, slew, imaging, margin, overran };
        },
        windowBarPct(task, segment) {
            const span = this._windowSpan(task);
            if (span <= 0) return 0;
            const segs = this._windowSegments(task);
            const val = segs[segment] || 0;
            return Math.max(val > 0 ? 1 : 0, Math.round((val / span) * 100));
        },
        windowBarOverran(task) {
            return this._windowSegments(task).overran;
        },
        windowBarTitle(task) {
            const s = this._windowSegments(task);
            const parts = [];
            if (s.delay) parts.push('Delay: ' + s.delay.toFixed(1) + 's');
            if (s.slew) parts.push('Slew: ' + s.slew.toFixed(1) + 's');
            if (s.imaging) parts.push('Imaging: ' + s.imaging.toFixed(1) + 's');
            if (s.margin) parts.push('Margin: ' + s.margin.toFixed(1) + 's');
            if (s.overran) parts.push('OVERRAN WINDOW');
            return parts.join(' | ') || 'No window data';
        },

        totalElapsed(d) {
            const start = d.slew_started_at ? new Date(d.slew_started_at).getTime() : null;
            const end = d.processing_finished_at ? new Date(d.processing_finished_at).getTime()
                      : d.imaging_finished_at ? new Date(d.imaging_finished_at).getTime()
                      : null;
            if (!start || !end) return 0;
            return Math.max(0, (end - start) / 1000);
        },

        // Processor segments for the summary bar
        get processorSegments() {
            const pp = this.stats.per_processor_timing;
            if (!pp) return [];
            const keys = Object.keys(PROCESSOR_COLORS);
            const vals = keys.map(k => pp[k] || 0);
            const total = vals.reduce((a, b) => a + b, 0);
            if (total <= 0) return [];
            return keys.map((k, i) => ({
                name: PROCESSOR_LABELS[k],
                val: vals[i].toFixed(2),
                pct: Math.round((vals[i] / total) * 100),
                color: PROCESSOR_COLORS[k],
            }));
        },

        imagingDuration(d) {
            if (!d || !d.imaging_started_at || !d.imaging_finished_at) return 0;
            return Math.max(0, (new Date(d.imaging_finished_at) - new Date(d.imaging_started_at)) / 1000);
        },

        // Phase bar: slew, imaging, queue wait, processing
        phasePct(d, phase) {
            if (!d) return 0;
            const vals = {
                slew: d.total_slew_time_s || 0,
                imaging: this.imagingDuration(d),
                queue: d.processing_queue_wait_s || 0,
                processing: d.total_processing_time_s || 0,
            };
            const total = Object.values(vals).reduce((a, b) => a + b, 0);
            if (total <= 0) return 0;
            return Math.max(vals[phase] > 0 ? 1 : 0, Math.round((vals[phase] / total) * 100));
        },

        // Processing breakdown bar: only per-processor times
        procPct(d, field) {
            if (!d) return 0;
            const fields = [
                'calibration_time_s', 'plate_solve_time_s', 'source_extractor_time_s',
                'photometry_time_s', 'matcher_time_s', 'annotated_image_time_s'
            ];
            const total = fields.reduce((s, f) => s + (d[f] || 0), 0);
            if (total <= 0) return 0;
            const val = d[field] || 0;
            return Math.max(val > 0 ? 1 : 0, Math.round((val / total) * 100));
        },

        // Window timeline proportional positioning
        windowPct(detail, part) {
            if (!detail.window_start || !detail.window_stop || !detail.imaging_started_at) return 0;
            try {
                const ws = new Date(detail.window_start).getTime();
                const we = new Date(detail.window_stop).getTime();
                const is = new Date(detail.imaging_started_at).getTime();
                const ie = detail.imaging_finished_at ? new Date(detail.imaging_finished_at).getTime() : is;
                const span = we - ws;
                if (span <= 0) return 0;
                if (part === 'start') return Math.max(0, Math.min(100, ((is - ws) / span) * 100));
                if (part === 'width') return Math.max(1, Math.min(100, ((ie - is) / span) * 100));
            } catch { return 0; }
            return 0;
        },

        ed(key) { return this._extractedData[key]; },

        get predictionsInField() {
            return this._extractedData['satellite_matcher.predictions_in_field'] || [];
        },

        get solveQuality() {
            return this._extractedData['plate_solver.solve_quality'] || null;
        },

        get hfr() {
            const v = this._extractedData['plate_solver.hfr_median'];
            return v != null ? v.toFixed(2) : null;
        },

        formatUploadType(d) {
            if (!d) return '—';
            if (d.should_upload === 0) return d.skip_reason || 'Skipped';
            const types = { fits_image: 'FITS image', observation_record: 'Observation only', csv_observations: 'CSV' };
            return types[d.upload_type] || d.upload_type || '—';
        },

        formatDeg(v, digits = 4) {
            return v != null ? v.toFixed(digits) + '°' : '—';
        },

        formatRA(deg) {
            if (deg == null) return '—';
            const h = deg / 15;
            const hh = Math.floor(h);
            const mm = Math.floor((h - hh) * 60);
            const ss = ((h - hh) * 60 - mm) * 60;
            return `${hh}h ${mm}m ${ss.toFixed(1)}s (${deg.toFixed(4)}°)`;
        },

        formatDec(deg) {
            if (deg == null) return '—';
            const sign = deg < 0 ? '-' : '+';
            const abs = Math.abs(deg);
            const dd = Math.floor(abs);
            const mm = Math.floor((abs - dd) * 60);
            const ss = ((abs - dd) * 60 - mm) * 60;
            return `${sign}${dd}° ${mm}′ ${ss.toFixed(1)}″ (${deg.toFixed(4)}°)`;
        },

        // ── Reprocess (single) ──────────────────────────────────────

        async doReprocess(taskId) {
            this.reprocessing = true;
            this.reprocessResult = null;
            this.reprocessError = null;
            try {
                const resp = await fetch('/api/analysis/tasks/' + taskId + '/reprocess', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        settings_overrides: {
                            sextractor_detect_thresh: this.reprocessThresh,
                            sextractor_detect_minarea: this.reprocessMinarea,
                            sextractor_filter_name: this.reprocessFilter,
                        },
                    }),
                });
                if (!resp.ok) {
                    const err = await resp.json();
                    this.reprocessError = err.error || 'Reprocess failed';
                    return;
                }
                const result = await resp.json();
                result._usedThresh = this.reprocessThresh;
                result._usedMinarea = this.reprocessMinarea;
                result._usedFilter = this.reprocessFilter;
                this.reprocessResult = result;
            } catch (e) {
                this.reprocessError = 'Request failed: ' + e.message;
            } finally {
                this.reprocessing = false;
            }
        },

        async doUploadReprocessed(taskId) {
            this.uploadingReprocessed = true;
            this.uploadReprocessedOk = false;
            this.uploadReprocessedError = null;
            try {
                const resp = await fetch('/api/analysis/tasks/' + taskId + '/reprocess/upload', {
                    method: 'POST',
                });
                if (!resp.ok) {
                    const err = await resp.json();
                    this.uploadReprocessedError = err.error || 'Upload failed';
                    return;
                }
                this.uploadReprocessedOk = true;
            } catch (e) {
                this.uploadReprocessedError = 'Request failed: ' + e.message;
            } finally {
                this.uploadingReprocessed = false;
            }
        },

        // ── Multi-select ────────────────────────────────────────────

        get selectedCount() {
            return Object.values(this.selectedTasks).filter(Boolean).length;
        },

        toggleSelect(taskId, checked) {
            this.selectedTasks[taskId] = checked;
        },

        toggleSelectAll(checked) {
            for (const task of this.tasks) {
                this.selectedTasks[task.task_id] = checked;
            }
        },

        async doBatchReprocess() {
            const ids = Object.entries(this.selectedTasks).filter(([, v]) => v).map(([k]) => k);
            if (!ids.length) return;
            this.batchResult = null;
            try {
                const resp = await fetch('/api/analysis/reprocess-batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        task_ids: ids,
                        settings_overrides: {
                            sextractor_detect_thresh: this.batchThresh,
                            sextractor_detect_minarea: this.batchMinarea,
                            sextractor_filter_name: this.batchFilter,
                        },
                    }),
                });
                const data = await resp.json();
                if (!resp.ok || !data.job_id) {
                    const { showToast } = await import('./config.js');
                    showToast(data.error || 'Batch reprocess failed', 'danger');
                    return;
                }
                this.batchJobId = data.job_id;
                this.batchProgress = 0;
                this.batchTotal = ids.length;
                this._pollBatchJob();
            } catch (e) {
                console.error('Batch reprocess failed', e);
            }
        },

        _pollBatchJob() {
            if (this._batchPollTimer) clearTimeout(this._batchPollTimer);
            this._batchPollTimer = setTimeout(async () => {
                if (!this.batchJobId) return;
                try {
                    const resp = await fetch('/api/jobs/' + this.batchJobId);
                    const job = await resp.json();
                    this.batchProgress = job.progress || 0;
                    this.batchTotal = job.total || this.batchTotal;
                    if (job.state === 'completed' || job.state === 'failed') {
                        this.batchResult = job.result || { succeeded: 0, failed: 0 };
                        this.batchJobId = null;
                        this.loadTasks();
                        return;
                    }
                    this._pollBatchJob();
                } catch (e) {
                    console.error('Poll batch job failed', e);
                    this._pollBatchJob();
                }
            }, 1000);
        },

        // ── Auto-tune ───────────────────────────────────────────────

        async doAutotune() {
            const ids = Object.entries(this.selectedTasks).filter(([, v]) => v).map(([k]) => k);
            this.autotuneResult = null;
            this.autotuneRunning = true;
            try {
                const resp = await fetch('/api/analysis/autotune', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ task_ids: ids.length ? ids : undefined }),
                });
                const data = await resp.json();
                if (data.error) {
                    this.autotuneResult = { error: data.error };
                    this.autotuneRunning = false;
                    return;
                }
                this.autotuneJobId = data.job_id;
                this.autotuneProgress = 0;
                this.autotuneTotal = data.total || 0;
                this._pollAutotuneJob();
            } catch (e) {
                console.error('Auto-tune failed', e);
                this.autotuneRunning = false;
            }
        },

        async cancelAutotune() {
            if (!this.autotuneJobId) return;
            try {
                await fetch('/api/jobs/' + this.autotuneJobId + '/cancel', { method: 'POST' });
            } catch (e) {
                console.error('Cancel autotune failed', e);
            }
        },

        _pollAutotuneJob() {
            if (this._autotunePollTimer) clearTimeout(this._autotunePollTimer);
            this._autotunePollTimer = setTimeout(async () => {
                if (!this.autotuneJobId) return;
                try {
                    const resp = await fetch('/api/jobs/' + this.autotuneJobId);
                    const job = await resp.json();
                    this.autotuneProgress = job.progress || 0;
                    this.autotuneTotal = job.total || this.autotuneTotal;
                    if (job.state === 'completed' || job.state === 'failed' || job.state === 'cancelled') {
                        this.autotuneResult = job.result;
                        this.autotuneJobId = null;
                        this.autotuneRunning = false;
                        this.autotuneOpen = true;
                        const { showToast } = await import('./config.js');
                        if (job.state === 'completed') {
                            const n = (job.result && job.result.configs) ? job.result.configs.length : 0;
                            showToast(`Auto-tune complete — ${n} configurations ranked`, 'success');
                        } else if (job.state === 'cancelled') {
                            showToast('Auto-tune cancelled — showing partial results', 'warning');
                        } else {
                            showToast('Auto-tune failed: ' + (job.error || 'unknown error'), 'danger');
                        }
                        return;
                    }
                    this._pollAutotuneJob();
                } catch (e) {
                    console.error('Poll autotune job failed', e);
                    this._pollAutotuneJob();
                }
            }, 2000);
        },

        async applyAutotuneSettings(config) {
            const store = Alpine.store('citrascope');
            if (!store || !store.config) return;

            store.config.sextractor_detect_thresh = config.detect_thresh;
            store.config.sextractor_detect_minarea = config.detect_minarea;
            store.config.sextractor_filter_name = config.filter_name;

            await window.saveConfiguration({ preventDefault() {} });
        },
    }));
});

window.loadAnalysisData = function () {
    const el = document.querySelector('#analysisSection');
    if (!el || !window.Alpine) return;
    const data = Alpine.$data(el);
    if (data && typeof data.loadAll === 'function') {
        data.loadAll();
    }
};
