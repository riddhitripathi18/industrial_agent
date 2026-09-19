"""
Heat Exchanger Data Access Tool
================================
Provides fast, LLM-friendly query functions over the heat exchanger
fouling dataset (`heat_exchanger_fouling_clean.csv`).

All public functions return pure Python dicts/lists (JSON-serializable),
ready to be passed directly as LLM tool call responses.

Usage:
    from agent.data_tools.hx_data import get_hx_telemetry
    result = get_hx_telemetry("E02", start_hour=500, end_hour=700)
"""

from __future__ import annotations

import functools
import pandas as pd
import numpy as np
from typing import Optional

from agent.config import (
    HX_CLEAN_CSV,
    HX_EXCHANGER_META,
    HX_COLUMN_PREFIX,
    VALID_EXCHANGER_IDS,
)


# ---------------------------------------------------------------------------
# Internal — cached DataFrame loader
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load_hx_df() -> pd.DataFrame:
    """Load and cache the heat exchanger dataset (called once per session)."""
    if not HX_CLEAN_CSV.is_file():
        raise FileNotFoundError(
            f"Heat exchanger dataset not found at: {HX_CLEAN_CSV}\n"
            "Set the INDUSTRIAL_DATA_DIR environment variable to the folder "
            "containing 'heat_exchanger_fouling_clean.csv'."
        )
    df = pd.read_csv(HX_CLEAN_CSV)
    df.sort_values("Time_hr", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _validate_exchanger(exchanger_id: str) -> None:
    eid = exchanger_id.upper()
    if eid not in VALID_EXCHANGER_IDS:
        raise ValueError(
            f"Unknown exchanger '{exchanger_id}'. "
            f"Valid options: {VALID_EXCHANGER_IDS}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_available_exchangers() -> list[dict]:
    """
    Return metadata for all heat exchangers (E01–E05).

    Returns:
        List of dicts with keys: exchanger_id, shell_fluid, tube_fluid,
        shell_cp_kJ_kgK, tube_cp_kJ_kgK, description.

    Example:
        >>> exchangers = get_available_exchangers()
        >>> exchangers[0]["exchanger_id"]
        'E01'
    """
    result = []
    for eid, meta in HX_EXCHANGER_META.items():
        result.append({"exchanger_id": eid, **meta})
    return result


def get_hx_telemetry(
    exchanger_id: str,
    start_hour: float = 0.0,
    end_hour: Optional[float] = None,
    sample_step: int = 1,
) -> dict:
    """
    Retrieve time-series telemetry for a single heat exchanger.

    Args:
        exchanger_id: One of 'E01', 'E02', 'E03', 'E04', 'E05'.
        start_hour:   Start of time window in run-hours (default 0.0).
        end_hour:     End of time window in run-hours (default: full run).
        sample_step:  Return every Nth row for downsampling (default 1 = all).

    Returns:
        dict with keys:
            exchanger_id, meta, time_hr, tube_temp_in, tube_temp_out,
            tube_mass_flow_kg_s, shell_temp_in, shell_temp_out,
            shell_mass_flow_kg_s, delta_t_tube, delta_t_shell,
            crude_api, crude_chlorides, crude_tan, n_points.

    Example:
        >>> data = get_hx_telemetry("E02", start_hour=200, end_hour=400)
        >>> len(data["time_hr"])
        200
    """
    eid = exchanger_id.upper()
    _validate_exchanger(eid)

    df = _load_hx_df()
    pfx = HX_COLUMN_PREFIX[eid]
    tube_pfx  = pfx["tube"]
    shell_pfx = pfx["shell"]

    # Time slice
    mask = df["Time_hr"] >= start_hour
    if end_hour is not None:
        mask &= df["Time_hr"] <= end_hour
    sliced = df.loc[mask].iloc[::sample_step].copy()

    # Derived temperature differentials
    sliced["_delta_t_tube"]  = sliced[f"{tube_pfx}_T_Out_degC"]  - sliced[f"{tube_pfx}_T_In_degC"]
    sliced["_delta_t_shell"] = sliced[f"{shell_pfx}_T_In_degC"]  - sliced[f"{shell_pfx}_T_Out_degC"]

    def _to_list(series: pd.Series) -> list:
        return [round(float(v), 4) if not np.isnan(v) else None for v in series]

    return {
        "exchanger_id":          eid,
        "meta":                  HX_EXCHANGER_META[eid],
        "time_hr":               _to_list(sliced["Time_hr"]),
        "tube_temp_in":          _to_list(sliced[f"{tube_pfx}_T_In_degC"]),
        "tube_temp_out":         _to_list(sliced[f"{tube_pfx}_T_Out_degC"]),
        "tube_mass_flow_kg_s":   _to_list(sliced[f"{tube_pfx}_m_kg_s"]),
        "shell_temp_in":         _to_list(sliced[f"{shell_pfx}_T_In_degC"]),
        "shell_temp_out":        _to_list(sliced[f"{shell_pfx}_T_Out_degC"]),
        "shell_mass_flow_kg_s":  _to_list(sliced[f"{shell_pfx}_m_kg_s"]),
        "delta_t_tube":          _to_list(sliced["_delta_t_tube"]),
        "delta_t_shell":         _to_list(sliced["_delta_t_shell"]),
        "crude_api":             _to_list(sliced["Crude_API"]),
        "crude_chlorides":       _to_list(sliced["Crude_Chlorides"]),
        "crude_tan":             _to_list(sliced["Crude_TAN"]),
        "n_points":              len(sliced),
        "time_range_hr":         [float(sliced["Time_hr"].iloc[0]), float(sliced["Time_hr"].iloc[-1])] if len(sliced) > 0 else [],
    }


def get_hx_snapshot(hour: float) -> dict:
    """
    Return a cross-sectional snapshot of all 5 exchangers at a specific run-hour.

    Finds the closest available data point to the requested hour.

    Args:
        hour: Target run-hour (e.g. 500.0).

    Returns:
        dict keyed by exchanger_id. Each entry has tube_temp_in/out,
        shell_temp_in/out, mass flows, delta-T tube/shell, and crude properties.

    Example:
        >>> snap = get_hx_snapshot(hour=500.0)
        >>> snap["E01"]["delta_t_tube"]
        12.34
    """
    df = _load_hx_df()

    # Find closest row to requested hour
    idx = (df["Time_hr"] - hour).abs().idxmin()
    row = df.loc[idx]
    actual_hour = float(row["Time_hr"])

    result: dict = {
        "requested_hour": hour,
        "actual_hour":    round(actual_hour, 2),
        "crude_api":      round(float(row["Crude_API"]), 4),
        "crude_chlorides": round(float(row["Crude_Chlorides"]), 4),
        "crude_tan":      round(float(row["Crude_TAN"]), 4),
        "exchangers":     {},
    }

    for eid in VALID_EXCHANGER_IDS:
        pfx = HX_COLUMN_PREFIX[eid]
        t_in  = float(row[f"{pfx['tube']}_T_In_degC"])
        t_out = float(row[f"{pfx['tube']}_T_Out_degC"])
        s_in  = float(row[f"{pfx['shell']}_T_In_degC"])
        s_out = float(row[f"{pfx['shell']}_T_Out_degC"])
        result["exchangers"][eid] = {
            "tube_temp_in":          round(t_in,  2),
            "tube_temp_out":         round(t_out, 2),
            "tube_mass_flow_kg_s":   round(float(row[f"{pfx['tube']}_m_kg_s"]),  4),
            "shell_temp_in":         round(s_in,  2),
            "shell_temp_out":        round(s_out, 2),
            "shell_mass_flow_kg_s":  round(float(row[f"{pfx['shell']}_m_kg_s"]), 4),
            "delta_t_tube":          round(t_out - t_in,  2),
            "delta_t_shell":         round(s_in  - s_out, 2),
        }

    return result


def get_hx_operating_summary(exchanger_id: str) -> dict:
    """
    Return statistical baseline summary for a heat exchanger over the full run.

    Useful for establishing 'clean' baseline values before degradation.

    Args:
        exchanger_id: One of 'E01', 'E02', 'E03', 'E04', 'E05'.

    Returns:
        dict with initial (first 5%), final (last 5%), mean, min, and max values
        for all key telemetry signals, plus total run duration in hours.

    Example:
        >>> summary = get_hx_operating_summary("E01")
        >>> summary["total_run_hr"]
        1000.0
    """
    eid = exchanger_id.upper()
    _validate_exchanger(eid)

    df    = _load_hx_df()
    pfx   = HX_COLUMN_PREFIX[eid]
    t_pfx = pfx["tube"]
    s_pfx = pfx["shell"]

    n       = len(df)
    n5      = max(1, int(n * 0.05))
    initial = df.iloc[:n5]
    final   = df.iloc[-n5:]

    def _stats(col: str) -> dict:
        series = df[col].dropna()
        return {
            "mean":    round(float(series.mean()),  4),
            "min":     round(float(series.min()),   4),
            "max":     round(float(series.max()),   4),
            "initial": round(float(initial[col].mean()), 4),
            "final":   round(float(final[col].mean()),   4),
            "drift":   round(float(final[col].mean()) - float(initial[col].mean()), 4),
        }

    return {
        "exchanger_id":          eid,
        "meta":                  HX_EXCHANGER_META[eid],
        "total_run_hr":          round(float(df["Time_hr"].max()), 2),
        "n_datapoints":          n,
        "tube_temp_in":          _stats(f"{t_pfx}_T_In_degC"),
        "tube_temp_out":         _stats(f"{t_pfx}_T_Out_degC"),
        "tube_mass_flow_kg_s":   _stats(f"{t_pfx}_m_kg_s"),
        "shell_temp_in":         _stats(f"{s_pfx}_T_In_degC"),
        "shell_temp_out":        _stats(f"{s_pfx}_T_Out_degC"),
        "shell_mass_flow_kg_s":  _stats(f"{s_pfx}_m_kg_s"),
        "crude_api":             _stats("Crude_API"),
        "crude_tan":             _stats("Crude_TAN"),
    }
