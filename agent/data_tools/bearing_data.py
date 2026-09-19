"""
IMS Bearing Vibration Data Access Tool
=========================================
Provides fast, LLM-friendly query functions over the processed IMS bearing
vibration feature dataset (`bearings_all_features.csv`).

All public functions return pure Python dicts/lists (JSON-serializable),
ready to be passed directly as LLM tool call responses.

Dataset structure:
    - Test 1: 2,156 snapshots, 8 channels (Bearing1_x/y to Bearing4_x/y)
    - Test 2:   984 snapshots, 4 channels (Bearing1 to Bearing4)
    - Test 3: 6,324 snapshots, 4 channels (Bearing1 to Bearing4)

Usage:
    from agent.data_tools.bearing_data import get_bearing_telemetry
    result = get_bearing_telemetry(test_id=2, bearing_id="Bearing1", metrics=["rms", "kurtosis"])
"""

from __future__ import annotations

import functools
import numpy as np
import pandas as pd
from typing import Optional

from agent.config import (
    BEARINGS_ALL_CSV,
    BEARING_TEST_META,
    BEARING_METRICS,
    ISO_10816_ZONES,
    VALID_TEST_IDS,
)


# ---------------------------------------------------------------------------
# Internal — cached DataFrame loader
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load_bearings_df() -> pd.DataFrame:
    """Load and cache the combined bearing features dataset (called once)."""
    if not BEARINGS_ALL_CSV.is_file():
        raise FileNotFoundError(
            f"Bearing dataset not found at: {BEARINGS_ALL_CSV}\n"
            "Set the INDUSTRIAL_DATA_DIR environment variable to the folder "
            "containing 'bearings_all_features.csv'."
        )
    df = pd.read_csv(BEARINGS_ALL_CSV, parse_dates=["timestamp"])
    df.sort_values(["test_id", "hours_elapsed"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _validate_test(test_id: int) -> None:
    if test_id not in VALID_TEST_IDS:
        raise ValueError(
            f"Unknown test_id '{test_id}'. Valid options: {VALID_TEST_IDS}"
        )


def _resolve_bearing_col(test_id: int, bearing_id: str, metric: str) -> str:
    """
    Resolve the correct column name for (test_id, bearing_id, metric).

    Test 1 uses dual-axis columns (Bearing1_x_rms, Bearing1_y_rms).
    Tests 2 & 3 use single-axis columns (Bearing1_rms).
    """
    meta = BEARING_TEST_META[test_id]

    if meta["axis"] == "dual":
        # bearing_id may already include axis suffix (e.g. "Bearing1_x")
        if "_x" in bearing_id.lower() or "_y" in bearing_id.lower():
            return f"{bearing_id}_{metric}"
        # Default to _x axis if not specified
        return f"{bearing_id}_x_{metric}"
    else:
        return f"{bearing_id}_{metric}"


def _to_list(series: pd.Series) -> list:
    return [round(float(v), 6) if not np.isnan(v) else None for v in series]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_bearing_tests_overview() -> list[dict]:
    """
    Return metadata summary for all 3 IMS bearing test runs.

    Returns:
        List of dicts per test run with:
            test_id, n_channels, bearings, failed_bearing, failure_type,
            n_snapshots, axis_type, duration_hr (from data).

    Example:
        >>> overview = get_bearing_tests_overview()
        >>> overview[0]["test_id"]
        1
    """
    df = _load_bearings_df()
    result = []
    for tid in VALID_TEST_IDS:
        meta   = BEARING_TEST_META[tid]
        subset = df[df["test_id"] == tid]
        result.append({
            "test_id":        tid,
            "n_channels":     meta["n_channels"],
            "bearings":       meta["bearings"],
            "failed_bearing": meta["failed_bearing"],
            "failure_type":   meta["failure_type"],
            "n_snapshots":    len(subset),
            "axis_type":      meta["axis"],
            "duration_hr":    round(float(subset["hours_elapsed"].max()), 2) if len(subset) > 0 else 0.0,
            "start_timestamp": str(subset["timestamp"].min()) if len(subset) > 0 else None,
            "end_timestamp":   str(subset["timestamp"].max()) if len(subset) > 0 else None,
        })
    return result


def get_bearing_telemetry(
    test_id: int,
    bearing_id: str = "Bearing1",
    metrics: list[str] = None,
    start_hour: float = 0.0,
    end_hour: Optional[float] = None,
    sample_step: int = 1,
) -> dict:
    """
    Retrieve vibration feature trajectories over time for a specific bearing.

    Args:
        test_id:     IMS test run number: 1, 2, or 3.
        bearing_id:  Bearing identifier.
                       Test 1: 'Bearing1_x', 'Bearing1_y', ... 'Bearing4_y'
                       Test 2/3: 'Bearing1', 'Bearing2', 'Bearing3', 'Bearing4'
        metrics:     List of metric names to return. Defaults to ['rms', 'kurtosis'].
                     Available: rms, peak, peak_to_peak, kurtosis, crest_factor,
                                std, skewness, variance, mean_abs, shape_factor,
                                impulse_factor.
        start_hour:  Start of time window (hours elapsed since test start).
        end_hour:    End of time window (None = full run).
        sample_step: Return every Nth row for downsampling (default 1 = all).

    Returns:
        dict with keys:
            test_id, bearing_id, metrics (dict of metric -> list of values),
            hours_elapsed, timestamps, failure_info, n_points.

    Example:
        >>> data = get_bearing_telemetry(2, "Bearing1", ["rms", "kurtosis"])
        >>> data["metrics"]["rms"][:3]
        [0.0032, 0.0031, 0.0033]
    """
    _validate_test(test_id)

    if metrics is None:
        metrics = ["rms", "kurtosis"]

    invalid = [m for m in metrics if m not in BEARING_METRICS]
    if invalid:
        raise ValueError(f"Unknown metric(s): {invalid}. Valid: {BEARING_METRICS}")

    df = _load_bearings_df()
    subset = df[df["test_id"] == test_id].copy()

    # Time window filter
    mask = subset["hours_elapsed"] >= start_hour
    if end_hour is not None:
        mask &= subset["hours_elapsed"] <= end_hour
    sliced = subset.loc[mask].iloc[::sample_step]

    # Resolve column names and extract
    metric_data: dict = {}
    for m in metrics:
        col = _resolve_bearing_col(test_id, bearing_id, m)
        if col not in sliced.columns:
            metric_data[m] = []
        else:
            metric_data[m] = _to_list(sliced[col])

    meta = BEARING_TEST_META[test_id]
    return {
        "test_id":       test_id,
        "bearing_id":    bearing_id,
        "axis_type":     meta["axis"],
        "metrics":       metric_data,
        "hours_elapsed": _to_list(sliced["hours_elapsed"]),
        "timestamps":    [str(t) for t in sliced["timestamp"]],
        "failure_info":  {
            "failed_bearing": meta["failed_bearing"],
            "failure_type":   meta["failure_type"],
        },
        "time_range_hr": [
            round(float(sliced["hours_elapsed"].min()), 2) if len(sliced) > 0 else 0.0,
            round(float(sliced["hours_elapsed"].max()), 2) if len(sliced) > 0 else 0.0,
        ],
        "n_points": len(sliced),
    }


def get_bearing_health_snapshot(test_id: int, hour: float) -> dict:
    """
    Return vibration status of all bearings at a specific hour within a test run.

    Finds the closest available snapshot to the requested hour.

    Args:
        test_id: IMS test run number: 1, 2, or 3.
        hour:    Target elapsed run-hour.

    Returns:
        dict keyed by bearing_id. Each bearing entry has all 11 vibration
        metrics plus ISO 10816 severity zone classification (based on RMS).

    Example:
        >>> snap = get_bearing_health_snapshot(2, hour=500.0)
        >>> snap["bearings"]["Bearing1"]["iso_zone"]
        'B'
    """
    _validate_test(test_id)

    df = _load_bearings_df()
    subset = df[df["test_id"] == test_id]

    if len(subset) == 0:
        raise ValueError(f"No data found for test_id={test_id}")

    # Closest snapshot
    idx    = (subset["hours_elapsed"] - hour).abs().idxmin()
    row    = subset.loc[idx]
    actual = float(row["hours_elapsed"])

    meta     = BEARING_TEST_META[test_id]
    bearings = meta["bearings"]
    axis     = meta["axis"]

    result: dict = {
        "test_id":        test_id,
        "requested_hour": hour,
        "actual_hour":    round(actual, 2),
        "timestamp":      str(row["timestamp"]),
        "bearings":       {},
    }

    # For dual-axis Test 1, report x-axis stats for each bearing base name
    base_bearings = (
        list({b.rsplit("_", 1)[0] for b in bearings})
        if axis == "dual"
        else bearings
    )
    base_bearings = sorted(base_bearings)

    for b in base_bearings:
        b_data: dict = {}
        for metric in BEARING_METRICS:
            col = _resolve_bearing_col(test_id, b, metric)
            b_data[metric] = round(float(row[col]), 6) if col in row.index and not np.isnan(row[col]) else None

        # ISO 10816 zone classification using RMS
        rms_val = b_data.get("rms")
        iso_zone = "Unknown"
        if rms_val is not None:
            for zone, (lo, hi) in ISO_10816_ZONES.items():
                if lo <= rms_val < hi:
                    iso_zone = zone
                    break

        b_data["iso_zone"] = iso_zone
        result["bearings"][b] = b_data

    return result


def get_bearing_critical_events(
    test_id: int,
    metric: str = "rms",
    threshold: Optional[float] = None,
) -> dict:
    """
    Detect timestamps when a vibration metric exceeds a threshold for any bearing.

    Useful for finding the precise onset of degradation or failure events.

    Args:
        test_id:   IMS test run number: 1, 2, or 3.
        metric:    Metric to monitor (default 'rms').
        threshold: Custom threshold value. If None, uses the ISO 10816 Zone B
                   boundary (2.3 for RMS) or 3× mean for other metrics.

    Returns:
        dict with:
            test_id, metric, threshold_used, events: list of dicts with
            hour, timestamp, bearing_id, value, iso_zone (for rms).

    Example:
        >>> events = get_bearing_critical_events(2, metric="kurtosis", threshold=4.0)
        >>> len(events["events"])
        12
    """
    _validate_test(test_id)

    if metric not in BEARING_METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Valid: {BEARING_METRICS}")

    df     = _load_bearings_df()
    subset = df[df["test_id"] == test_id].copy()
    meta   = BEARING_TEST_META[test_id]
    axis   = meta["axis"]

    # Determine bearing column names
    bearings = meta["bearings"]
    if axis == "dual":
        # Only x-axis for duplicate detection
        bearings = [b for b in bearings if "_x" in b]

    # Auto-threshold
    if threshold is None:
        if metric == "rms":
            threshold = ISO_10816_ZONES["B"][0]  # 2.3 mm/s
        else:
            all_cols = [_resolve_bearing_col(test_id, b, metric) for b in bearings]
            valid    = [c for c in all_cols if c in subset.columns]
            if valid:
                overall_mean = subset[valid].mean().mean()
                threshold = round(float(overall_mean * 3.0), 6)
            else:
                threshold = 0.0

    events = []
    for b in bearings:
        col = _resolve_bearing_col(test_id, b, metric)
        if col not in subset.columns:
            continue
        over = subset[subset[col] > threshold]
        for _, row in over.iterrows():
            val = float(row[col])
            iso_zone = "N/A"
            if metric == "rms":
                for zone, (lo, hi) in ISO_10816_ZONES.items():
                    if lo <= val < hi:
                        iso_zone = zone
                        break
            events.append({
                "hour":       round(float(row["hours_elapsed"]), 2),
                "timestamp":  str(row["timestamp"]),
                "bearing_id": b,
                "value":      round(val, 6),
                "iso_zone":   iso_zone,
            })

    # Sort chronologically
    events.sort(key=lambda e: e["hour"])

    return {
        "test_id":         test_id,
        "metric":          metric,
        "threshold_used":  round(float(threshold), 6),
        "n_events":        len(events),
        "failure_info":    {
            "failed_bearing": meta["failed_bearing"],
            "failure_type":   meta["failure_type"],
        },
        "events": events,
    }
