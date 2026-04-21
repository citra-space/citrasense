"""Tests for SlewRateTracker — rolling-mean observed slew rate."""

from __future__ import annotations

import math

import pytest

from citrasense.hardware.abstract_astro_hardware_adapter import SlewRateTracker


class TestEmptyState:
    def test_no_samples_returns_none(self):
        t = SlewRateTracker()
        assert t.mean is None
        assert t.count == 0

    def test_reset_clears_samples(self):
        t = SlewRateTracker()
        t.record(3.0)
        t.record(4.0)
        assert t.count == 2
        t.reset()
        assert t.count == 0
        assert t.mean is None


class TestMean:
    def test_single_sample_is_the_mean(self):
        t = SlewRateTracker()
        t.record(4.2)
        assert t.mean == pytest.approx(4.2)
        assert t.count == 1

    def test_arithmetic_mean(self):
        t = SlewRateTracker()
        for v in (2.0, 4.0, 6.0):
            t.record(v)
        assert t.mean == pytest.approx(4.0)
        assert t.count == 3


class TestWindow:
    def test_window_evicts_oldest(self):
        t = SlewRateTracker(window=3)
        for v in (1.0, 2.0, 3.0, 4.0):
            t.record(v)
        # Oldest (1.0) should be gone; mean over [2, 3, 4].
        assert t.count == 3
        assert t.mean == pytest.approx(3.0)

    def test_window_must_be_positive(self):
        with pytest.raises(ValueError, match="window must be >= 1"):
            SlewRateTracker(window=0)


class TestBoundsValidation:
    def test_lo_must_be_finite(self):
        with pytest.raises(ValueError, match="lo/hi must be finite"):
            SlewRateTracker(lo=-math.inf, hi=50.0)

    def test_hi_must_be_finite(self):
        with pytest.raises(ValueError, match="lo/hi must be finite"):
            SlewRateTracker(lo=0.1, hi=math.inf)

    def test_nan_bounds_rejected(self):
        with pytest.raises(ValueError, match="lo/hi must be finite"):
            SlewRateTracker(lo=math.nan, hi=50.0)

    def test_lo_greater_than_hi_rejected(self):
        with pytest.raises(ValueError, match="lo <= hi"):
            SlewRateTracker(lo=10.0, hi=1.0)

    def test_lo_equal_hi_is_allowed(self):
        # Degenerate but well-defined: every sample pins to the same value.
        t = SlewRateTracker(lo=3.0, hi=3.0)
        t.record(0.0)
        t.record(100.0)
        assert t.mean == pytest.approx(3.0)


class TestClamping:
    def test_high_samples_are_clamped(self):
        t = SlewRateTracker(lo=0.1, hi=50.0)
        t.record(999.0)
        assert t.mean == pytest.approx(50.0)

    def test_low_samples_are_clamped(self):
        t = SlewRateTracker(lo=0.1, hi=50.0)
        t.record(0.0001)
        assert t.mean == pytest.approx(0.1)

    def test_in_range_samples_untouched(self):
        t = SlewRateTracker(lo=0.1, hi=50.0)
        t.record(5.0)
        t.record(10.0)
        assert t.mean == pytest.approx(7.5)

    def test_custom_bounds(self):
        t = SlewRateTracker(lo=1.0, hi=2.0)
        t.record(0.5)
        t.record(5.0)
        # Both clamp: 1.0, 2.0 → mean 1.5
        assert t.mean == pytest.approx(1.5)
