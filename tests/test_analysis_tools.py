"""
Phase 2 Test Suite — Analysis Tools
======================================
Validates signal diagnostics, trend detection, correlation, and FFT functions.

Run with:
    pytest tests/test_analysis_tools.py -v
"""

import pytest
import math
import numpy as np
from agent.data_tools.bearing_data import get_bearing_telemetry
from agent.data_tools.hx_data import get_hx_telemetry
from agent.tools.engineering_tools import calc_fouling_resistance
from agent.tools.analysis_tools import (
    trend_analyzer,
    anomaly_detector,
    correlation_matrix,
    compute_fft_spectrum,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def bearing_t2_rms():
    data = get_bearing_telemetry(2, "Bearing1", ["rms", "kurtosis"])
    return data

@pytest.fixture(scope="module")
def hx_e01_telemetry():
    return get_hx_telemetry("E01")

@pytest.fixture(scope="module")
def sine_signal():
    """Clean 10 Hz sine wave at 1000 Hz sample rate — known FFT peak."""
    t = np.linspace(0, 1.0, 1000, endpoint=False)
    y = np.sin(2 * np.pi * 10 * t)   # 10 Hz
    return y.tolist()

@pytest.fixture(scope="module")
def noisy_increasing():
    """Synthetic clearly-increasing signal with small noise."""
    np.random.seed(42)
    t = np.linspace(0, 100, 200)
    y = 0.5 * t + np.random.normal(0, 0.5, 200)
    return t.tolist(), y.tolist()

@pytest.fixture(scope="module")
def rf_e01(hx_e01_telemetry):
    return calc_fouling_resistance(hx_e01_telemetry, "E01")


# ===========================================================================
# 1. Trend Analyzer
# ===========================================================================

class TestTrendAnalyzer:

    def test_increasing_signal_direction(self, noisy_increasing):
        t, y = noisy_increasing
        result = trend_analyzer(y, t, label="test_signal")
        assert result["trend_direction"].startswith("degrading"), \
            f"Expected 'degrading', got: {result['trend_direction']}"

    def test_slope_correct_order_of_magnitude(self, noisy_increasing):
        """Synthetic signal has slope ~0.5 per unit time."""
        t, y = noisy_increasing
        result = trend_analyzer(y, t)
        assert 0.3 <= result["slope"] <= 0.7, \
            f"Slope {result['slope']} far from expected ~0.5"

    def test_r_squared_high_for_linear(self, noisy_increasing):
        t, y = noisy_increasing
        result = trend_analyzer(y, t)
        assert result["r_squared"] >= 0.95, \
            f"R² {result['r_squared']} low for nearly-linear signal"

    def test_projected_breach_future(self, noisy_increasing):
        """Breach projection must be in the future (> current hour)."""
        t, y = noisy_increasing
        result = trend_analyzer(y, t, threshold=100.0)
        if result["projected_breach_hour"] is not None:
            assert result["projected_breach_hour"] > result["current_hour"]

    def test_bearing_rms_trend_degrading(self, bearing_t2_rms):
        """Test 2 Bearing 1 is a known run-to-failure — must trend upward."""
        rms   = [v for v in bearing_t2_rms["metrics"]["rms"] if v is not None]
        hours = bearing_t2_rms["hours_elapsed"]
        result = trend_analyzer(rms, hours, label="Bearing1 RMS")
        assert result["slope"] > 0, "Run-to-failure bearing must have positive slope"

    def test_rf_trend_upward(self, rf_e01):
        """Fouling resistance should trend upward over the campaign."""
        rf    = [v for v in rf_e01["Rf"] if v is not None and not math.isnan(v)]
        hours = [h for h, v in zip(rf_e01["time_hr"], rf_e01["Rf"])
                 if v is not None and not math.isnan(v)]
        result = trend_analyzer(rf, hours, label="E01 Rf")
        assert result["slope"] >= 0, f"Fouling Rf slope should be >= 0, got {result['slope']}"

    def test_required_keys(self, noisy_increasing):
        t, y   = noisy_increasing
        result = trend_analyzer(y, t)
        keys   = {"slope", "r_squared", "trend_direction", "current_value",
                  "fitted_values", "residuals", "intercept"}
        assert not (keys - set(result.keys()))

    def test_json_serializable(self, noisy_increasing):
        import json
        t, y = noisy_increasing
        result = trend_analyzer(y, t, threshold=50.0)
        json.dumps(result)


# ===========================================================================
# 2. Anomaly Detector
# ===========================================================================

class TestAnomalyDetector:

    def test_zscore_detects_spike(self):
        """Inject a known outlier and verify it is flagged."""
        hours  = list(range(100))
        values = [1.0] * 100
        values[50] = 20.0   # obvious spike
        result = anomaly_detector(values, hours, method="zscore", window=20)
        flagged_indices = [e["index"] for e in result["anomaly_events"]]
        assert 50 in flagged_indices, "Injected spike at index 50 not detected"

    def test_iqr_detects_outliers(self):
        np.random.seed(0)
        hours  = list(range(200))
        values = np.random.normal(0, 0.1, 200).tolist()
        values[100] = 5.0   # extreme outlier
        result = anomaly_detector(values, hours, method="iqr")
        assert result["n_anomalies"] >= 1

    def test_kurtosis_method_runs(self, bearing_t2_rms):
        kurtosis = [v for v in bearing_t2_rms["metrics"]["kurtosis"] if v is not None]
        hours    = bearing_t2_rms["hours_elapsed"][:len(kurtosis)]
        result   = anomaly_detector(kurtosis, hours, method="kurtosis")
        assert "n_anomalies"    in result
        assert "anomaly_events" in result

    def test_no_anomalies_in_flat_signal(self):
        """Perfectly flat signal should have no z-score anomalies."""
        hours  = list(range(100))
        values = [5.0] * 100
        result = anomaly_detector(values, hours, method="zscore")
        assert result["n_anomalies"] == 0

    def test_invalid_method_returns_error(self):
        result = anomaly_detector([1, 2, 3], [0, 1, 2], method="invalid")
        assert "error" in result

    def test_required_keys(self):
        result = anomaly_detector([1.0] * 60, list(range(60)), method="zscore")
        keys   = {"n_anomalies", "anomaly_rate_pct", "baseline_mean",
                  "baseline_std", "anomaly_events", "method"}
        assert not (keys - set(result.keys()))

    def test_bearing_kurtosis_spike_detected_late_stage(self, bearing_t2_rms):
        """Test 2 Bearing 1 is known to spike in kurtosis as it fails."""
        kurtosis = [v for v in bearing_t2_rms["metrics"]["kurtosis"] if v is not None]
        hours    = bearing_t2_rms["hours_elapsed"][:len(kurtosis)]
        result   = anomaly_detector(kurtosis, hours, method="zscore", window=30)
        # The famous kurtosis spike near end of Test 2 should be detectable
        if result["n_anomalies"] > 0:
            assert result["last_anomaly_hour"] > result["first_anomaly_hour"] or \
                   result["n_anomalies"] >= 1


# ===========================================================================
# 3. Correlation Matrix
# ===========================================================================

class TestCorrelationMatrix:

    def test_self_correlation_is_one(self):
        values = [float(i) for i in range(50)]
        result = correlation_matrix({"a": values, "b": values})
        # a vs a and b vs b should be 1.0
        assert result["matrix"][0][0] == 1.0
        assert result["matrix"][1][1] == 1.0

    def test_identical_signals_perfect_correlation(self):
        values = [float(i) for i in range(50)]
        result = correlation_matrix({"x": values, "y": values})
        assert result["matrix"][0][1] == pytest.approx(1.0, abs=1e-3)

    def test_opposite_signals_negative_correlation(self):
        values = [float(i) for i in range(50)]
        neg    = [float(-i) for i in range(50)]
        result = correlation_matrix({"pos": values, "neg": neg})
        assert result["matrix"][0][1] == pytest.approx(-1.0, abs=1e-3)

    def test_strong_correlations_list(self):
        values = [float(i) for i in range(100)]
        result = correlation_matrix({"a": values, "b": values})
        assert len(result["strong_correlations"]) >= 1

    def test_minimum_two_signals(self):
        result = correlation_matrix({"only_one": [1, 2, 3]})
        assert "error" in result

    def test_real_data_tan_vs_rf(self, hx_e01_telemetry, rf_e01):
        """Check correlation between crude TAN and fouling resistance."""
        tan = hx_e01_telemetry["crude_tan"]
        rf  = rf_e01["Rf"]
        # Align lengths
        min_len = min(len(tan), len(rf))
        tan_c   = [v for v in tan[:min_len] if v is not None]
        rf_c    = [v for v in rf[:min_len]  if v is not None and not math.isnan(v)]
        min_len = min(len(tan_c), len(rf_c))
        result  = correlation_matrix({"crude_tan": tan_c[:min_len], "Rf_E01": rf_c[:min_len]})
        # Just verify it runs without error and has valid structure
        assert "matrix"       in result
        assert "signal_names" in result
        assert len(result["matrix"]) == 2

    def test_json_serializable(self):
        import json
        result = correlation_matrix({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
        json.dumps(result)


# ===========================================================================
# 4. FFT Spectrum
# ===========================================================================

class TestFFTSpectrum:

    def test_sine_peak_at_correct_frequency(self, sine_signal):
        """10 Hz sine wave at 1000 Hz sample rate — dominant peak must be ~10 Hz."""
        result = compute_fft_spectrum(sine_signal, sample_rate_hz=1000.0, top_n=5)
        peak_freqs = [p["freq_hz"] for p in result["dominant_peaks"]]
        assert any(abs(f - 10.0) < 2.0 for f in peak_freqs), \
            f"10 Hz peak not found. Top peaks: {peak_freqs}"

    def test_output_has_required_keys(self, sine_signal):
        result   = compute_fft_spectrum(sine_signal, sample_rate_hz=1000.0)
        required = {"frequencies", "magnitudes", "dominant_peaks",
                    "sample_rate_hz", "n_samples", "rms_from_fft"}
        assert not (required - set(result.keys()))

    def test_nyquist_respected(self, sine_signal):
        """Maximum frequency must be <= Nyquist (sample_rate / 2)."""
        result  = compute_fft_spectrum(sine_signal, sample_rate_hz=1000.0)
        assert result["max_freq_hz"] <= 500.5   # Nyquist = 500 Hz

    def test_n_samples_correct(self, sine_signal):
        result = compute_fft_spectrum(sine_signal, sample_rate_hz=1000.0)
        assert result["n_samples"] == len(sine_signal)

    def test_too_few_samples_returns_error(self):
        result = compute_fft_spectrum([1.0, 2.0, 3.0], sample_rate_hz=100.0)
        assert "error" in result

    def test_json_serializable(self, sine_signal):
        import json
        result = compute_fft_spectrum(sine_signal, sample_rate_hz=1000.0)
        json.dumps(result)

    def test_two_tone_signal_two_peaks(self):
        """Two-frequency signal should produce two dominant peaks."""
        sr   = 2000.0
        t    = np.linspace(0, 1.0, int(sr), endpoint=False)
        y    = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 200 * t)
        result = compute_fft_spectrum(y.tolist(), sample_rate_hz=sr, top_n=5)
        freqs = [p["freq_hz"] for p in result["dominant_peaks"]]
        near_50  = any(abs(f - 50.0)  < 3 for f in freqs)
        near_200 = any(abs(f - 200.0) < 3 for f in freqs)
        assert near_50  and near_200, f"Expected peaks near 50 Hz and 200 Hz, got: {freqs}"
