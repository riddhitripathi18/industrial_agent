"""
Analysis Tools — Signal Diagnostics & Statistical Trend Detection
==================================================================
Pure signal-processing and statistical functions for industrial equipment
condition monitoring.

All functions are LLM-tool-ready (type hints, docstrings, JSON outputs).

Functions:
    trend_analyzer()        Linear regression on any degradation signal
    anomaly_detector()      Z-score / IQR / kurtosis spike outlier detection
    correlation_matrix()    Pearson correlations between multiple signals
    compute_fft_spectrum()  FFT magnitude spectrum for raw vibration data
"""

from __future__ import annotations

import math
import numpy as np
from scipy import stats, signal as sp_signal
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Trend Analyzer
# ---------------------------------------------------------------------------

def trend_analyzer(
    values: list[float],
    hours: list[float],
    threshold: Optional[float] = None,
    label: str = "signal",
) -> dict:
    """
    Fit a linear trend to a time-series signal and project when it breaches a threshold.

    Uses Ordinary Least Squares linear regression: y = slope * t + intercept.

    Args:
        values:    List of signal values (e.g. Rf, RMS, delta-T).
        hours:     Corresponding timestamps in hours elapsed.
        threshold: Optional breach threshold. If provided, projects the hour
                   when the trend line will cross it.
        label:     Human-readable name for the signal (used in output).

    Returns:
        dict with keys:
            label, slope (units/hr), slope_per_day, intercept,
            r_squared, trend_direction, current_value, current_hour,
            fitted_values, residuals,
            projected_breach_hour (or None if no threshold / not trending there).

    Example:
        >>> result = trend_analyzer(rms_values, hours, threshold=4.5, label="Bearing1 RMS")
        >>> result["slope"]
        0.00412
    """
    t = np.array(hours,  dtype=float)
    y = np.array(values, dtype=float)

    # Remove NaN
    mask = ~(np.isnan(t) | np.isnan(y))
    t, y = t[mask], y[mask]

    if len(t) < 3:
        return {"error": "Need at least 3 data points for trend analysis."}

    slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
    r2 = float(r_value ** 2)

    fitted     = slope * t + intercept
    residuals  = y - fitted

    # Trend direction
    if abs(slope) < std_err * 0.5:
        direction = "stable"
    elif slope > 0:
        direction = "degrading (increasing)"
    else:
        direction = "recovering (decreasing)"

    # Project breach
    projected_breach = None
    if threshold is not None and slope != 0:
        t_breach = (threshold - intercept) / slope
        if t_breach > t[-1]:   # only future breaches
            projected_breach = round(float(t_breach), 2)

    return {
        "label":                 label,
        "n_points":              int(len(t)),
        "slope":                 round(float(slope), 8),
        "slope_per_day":         round(float(slope * 24), 8),
        "intercept":             round(float(intercept), 8),
        "r_squared":             round(r2, 4),
        "p_value":               round(float(p_value), 6),
        "std_err":               round(float(std_err), 8),
        "trend_direction":       direction,
        "current_value":         round(float(y[-1]), 6),
        "current_hour":          round(float(t[-1]), 2),
        "initial_value":         round(float(y[0]),  6),
        "total_drift":           round(float(y[-1] - y[0]), 6),
        "fitted_values":         [round(float(v), 6) for v in fitted],
        "residuals":             [round(float(v), 6) for v in residuals],
        "projected_breach_hour": projected_breach,
        "threshold":             threshold,
    }


# ---------------------------------------------------------------------------
# 2. Anomaly Detector
# ---------------------------------------------------------------------------

def anomaly_detector(
    values: list[float],
    hours: list[float],
    method: str = "zscore",
    window: int = 50,
    z_threshold: float = 3.0,
    kurtosis_threshold: float = 3.5,
    label: str = "signal",
) -> dict:
    """
    Detect statistical anomalies in a time-series signal.

    Methods:
        'zscore'   — flags points where rolling z-score > z_threshold (default 3σ)
        'iqr'      — flags points outside rolling 1.5 × IQR fence
        'kurtosis' — flags rolling windows where kurtosis > kurtosis_threshold

    Args:
        values:              Signal values list.
        hours:               Timestamps in hours elapsed.
        method:              Detection method: 'zscore' | 'iqr' | 'kurtosis'.
        window:              Rolling window size in samples (default 50).
        z_threshold:         Z-score cutoff for 'zscore' method (default 3.0).
        kurtosis_threshold:  Kurtosis cutoff for 'kurtosis' method (default 3.5).
        label:               Human-readable signal name.

    Returns:
        dict with keys:
            label, method, n_anomalies, anomaly_events (list of dicts),
            baseline_mean, baseline_std, anomaly_rate_pct.

    Example:
        >>> result = anomaly_detector(kurtosis_values, hours, method="zscore")
        >>> result["n_anomalies"]
        7
    """
    t = np.array(hours,  dtype=float)
    y = np.array(values, dtype=float)

    mask = ~(np.isnan(t) | np.isnan(y))
    t, y = t[mask], y[mask]

    if len(y) < window:
        window = max(3, len(y) // 3)

    baseline_mean = float(np.mean(y))
    baseline_std  = float(np.std(y))

    anomaly_indices = []

    if method == "zscore":
        for i in range(len(y)):
            lo = max(0, i - window)
            chunk = y[lo:i + 1]
            mu, sigma = np.mean(chunk), np.std(chunk)
            if sigma > 0 and abs(y[i] - mu) / sigma > z_threshold:
                anomaly_indices.append(i)

    elif method == "iqr":
        for i in range(len(y)):
            lo    = max(0, i - window)
            chunk = y[lo:i + 1]
            q1, q3 = np.percentile(chunk, [25, 75])
            iqr    = q3 - q1
            fence_lo, fence_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            if y[i] < fence_lo or y[i] > fence_hi:
                anomaly_indices.append(i)

    elif method == "kurtosis":
        for i in range(window, len(y)):
            chunk = y[i - window:i]
            kurt  = float(stats.kurtosis(chunk, fisher=True))
            if kurt > kurtosis_threshold:
                anomaly_indices.append(i)

    else:
        return {"error": f"Unknown method '{method}'. Use 'zscore', 'iqr', or 'kurtosis'."}

    events = []
    for idx in anomaly_indices:
        events.append({
            "hour":     round(float(t[idx]), 2),
            "value":    round(float(y[idx]), 6),
            "index":    int(idx),
            "severity": round(abs(float(y[idx]) - baseline_mean) / (baseline_std + 1e-9), 2),
        })

    return {
        "label":             label,
        "method":            method,
        "n_points":          int(len(y)),
        "n_anomalies":       len(events),
        "anomaly_rate_pct":  round(len(events) / max(len(y), 1) * 100, 2),
        "baseline_mean":     round(baseline_mean, 6),
        "baseline_std":      round(baseline_std,  6),
        "window":            window,
        "anomaly_events":    events,
        "first_anomaly_hour": events[0]["hour"] if events else None,
        "last_anomaly_hour":  events[-1]["hour"] if events else None,
    }


# ---------------------------------------------------------------------------
# 3. Correlation Matrix
# ---------------------------------------------------------------------------

def correlation_matrix(signals: dict[str, list[float]]) -> dict:
    """
    Compute the Pearson correlation matrix between multiple signals.

    Useful for identifying root causes: e.g. does crude TAN correlate
    with fouling rate? Does RMS Bearing1 correlate with Bearing3?

    Args:
        signals: Dict mapping signal name → list of values.
                 All lists must be the same length.
                 Example: {"crude_tan": [...], "Rf_E01": [...], "Rf_E02": [...]}

    Returns:
        dict with keys:
            signal_names, matrix (2D list of Pearson r values, row=col=signal_names),
            strong_correlations (|r| > 0.7), n_points.

    Example:
        >>> result = correlation_matrix({"tan": tan_vals, "Rf": rf_vals})
        >>> result["matrix"][0][1]
        0.87
    """
    names = list(signals.keys())
    if len(names) < 2:
        return {"error": "Need at least 2 signals for correlation."}

    # Align lengths to shortest
    min_len = min(len(v) for v in signals.values())
    arrays  = {k: np.array(v[:min_len], dtype=float) for k, v in signals.items()}

    n    = len(names)
    mat  = [[0.0] * n for _ in range(n)]
    strong = []

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                mat[i][j] = 1.0
            elif j > i:
                xi, xj = arrays[ni], arrays[nj]
                valid   = ~(np.isnan(xi) | np.isnan(xj))
                if valid.sum() < 3:
                    r = float("nan")
                else:
                    r, _ = stats.pearsonr(xi[valid], xj[valid])
                r = round(float(r), 4)
                mat[i][j] = r
                mat[j][i] = r
                if abs(r) > 0.7:
                    strong.append({
                        "signal_a": ni,
                        "signal_b": nj,
                        "r":        r,
                        "strength": "strong positive" if r > 0.7 else "strong negative",
                    })

    return {
        "signal_names":       names,
        "matrix":             mat,
        "n_points":           min_len,
        "strong_correlations": strong,
        "interpretation":     (
            "r > 0.7  = strong positive correlation\n"
            "r < -0.7 = strong negative correlation\n"
            "0.3–0.7  = moderate correlation\n"
            "|r| < 0.3 = weak or no correlation"
        ),
    }


# ---------------------------------------------------------------------------
# 4. FFT Spectrum
# ---------------------------------------------------------------------------

def compute_fft_spectrum(
    raw_signal: list[float],
    sample_rate_hz: float,
    top_n: int = 10,
    label: str = "vibration",
) -> dict:
    """
    Compute the FFT magnitude spectrum of a raw vibration signal.

    Identifies dominant frequencies and supports bearing fault diagnosis
    using standard defect frequencies (BPFO, BPFI, BSF, FTF).

    Typical IMS bearing defect frequencies at 2000 RPM (33.3 Hz shaft):
        BPFO (Ball Pass, Outer Race): ~236 Hz
        BPFI (Ball Pass, Inner Race): ~297 Hz
        BSF  (Ball Spin Frequency):   ~139 Hz
        FTF  (Fundamental Train Freq): ~13 Hz

    Args:
        raw_signal:     Time-domain vibration samples (acceleration g or mm/s).
        sample_rate_hz: Sampling rate in Hz (IMS dataset = 20,480 Hz).
        top_n:          Number of dominant frequency peaks to return (default 10).
        label:          Signal label for output.

    Returns:
        dict with keys:
            label, sample_rate_hz, n_samples, freq_resolution_hz,
            frequencies (Hz list), magnitudes (list), dominant_peaks (top_n),
            rms_from_fft, dc_component.

    Example:
        >>> result = compute_fft_spectrum(raw_signal, sample_rate_hz=20480)
        >>> result["dominant_peaks"][0]["freq_hz"]
        236.4
    """
    y = np.array(raw_signal, dtype=float)
    n = len(y)

    if n < 8:
        return {"error": "Need at least 8 samples for FFT."}

    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(n)
    y_win  = y * window

    # One-sided FFT
    fft_vals = np.fft.rfft(y_win)
    freqs    = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    mags     = (np.abs(fft_vals) * 2.0 / n)   # amplitude normalised

    # DC component (0 Hz)
    dc = round(float(mags[0]), 8)

    # RMS from FFT (Parseval's theorem)
    rms_fft = round(float(np.sqrt(np.mean(np.abs(fft_vals / n) ** 2) * 2)), 6)

    # Top N peaks (excluding DC)
    peak_indices = np.argsort(mags[1:])[::-1][:top_n] + 1
    dominant_peaks = [
        {
            "rank":    int(rank + 1),
            "freq_hz": round(float(freqs[idx]), 3),
            "magnitude": round(float(mags[idx]), 8),
        }
        for rank, idx in enumerate(peak_indices)
    ]
    dominant_peaks.sort(key=lambda x: x["freq_hz"])

    return {
        "label":               label,
        "sample_rate_hz":      sample_rate_hz,
        "n_samples":           n,
        "freq_resolution_hz":  round(float(sample_rate_hz / n), 4),
        "max_freq_hz":         round(float(freqs[-1]), 2),
        "dc_component":        dc,
        "rms_from_fft":        rms_fft,
        "dominant_peaks":      dominant_peaks,
        "frequencies":         [round(float(f), 3) for f in freqs[:500]],  # first 500 bins
        "magnitudes":          [round(float(m), 8) for m in mags[:500]],
        "note": (
            "IMS bearing shaft speed ~2000 RPM. "
            "Known defect frequencies: BPFO ~236 Hz, BPFI ~297 Hz, "
            "BSF ~139 Hz, FTF ~13 Hz."
        ),
    }
