"""
Milling Data Tool — Phase B (Predictive Maintenance Dataset)
=============================================================
Provides query functions for the 10,000-record CNC milling machine dataset (ai4i2020.csv).
Tracks:
  - Operating conditions: Air temp, Process temp, Speed (rpm), Torque (Nm), Tool wear (min)
  - Failure indicators: Machine failure, TWF, HDF, PWF, OSF, RNF
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "ai4i2020.csv"
if not DATA_PATH.exists():
    DATA_PATH = Path(r"c:\Users\riddh\OneDrive\Desktop\predictive_maintenance\ai4i2020.csv")

_cached_df: Optional[pd.DataFrame] = None


def _get_df() -> pd.DataFrame:
    global _cached_df
    if _cached_df is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"ai4i2020.csv not found at {DATA_PATH}")
        _cached_df = pd.read_csv(DATA_PATH)
    return _cached_df


def get_milling_fleet_overview() -> Dict[str, Any]:
    """
    Get high-level summary of the 10,000 milling machines in the fleet.
    Returns:
        dict: Total machines, failure breakdown by mode (TWF, HDF, PWF, OSF, RNF),
              distribution by quality type (L, M, H), and tool wear statistics.
    """
    df = _get_df()
    total = len(df)
    failures = int(df["Machine failure"].sum())
    
    twf = int(df["TWF"].sum())
    hdf = int(df["HDF"].sum())
    pwf = int(df["PWF"].sum())
    osf = int(df["OSF"].sum())
    rnf = int(df["RNF"].sum())
    
    type_counts = df["Type"].value_counts().to_dict()
    
    return {
        "total_machines": total,
        "total_failures": failures,
        "fleet_failure_rate_pct": round((failures / total) * 100, 2),
        "failure_mode_breakdown": {
            "TWF_tool_wear_failures": twf,
            "HDF_heat_dissipation_failures": hdf,
            "PWF_power_failures": pwf,
            "OSF_overstrain_failures": osf,
            "RNF_random_failures": rnf,
        },
        "quality_types": type_counts,
        "tool_wear_stats_min": {
            "mean": round(float(df["Tool wear [min]"].mean()), 1),
            "median": round(float(df["Tool wear [min]"].median()), 1),
            "max": int(df["Tool wear [min]"].max()),
        }
    }


def get_milling_machine_telemetry(machine_id: int) -> Dict[str, Any]:
    """
    Retrieve telemetry and failure status for a specific machine by UDI (1 to 10000).
    """
    df = _get_df()
    row = df[df["UDI"] == machine_id]
    if row.empty:
        # Try matching on Product ID string or default
        return {"error": f"Milling machine with UDI {machine_id} not found."}
    
    r = row.iloc[0]
    return {
        "udi": int(r["UDI"]),
        "product_id": str(r["Product ID"]),
        "type": str(r["Type"]),
        "air_temperature_K": float(r["Air temperature [K]"]),
        "process_temperature_K": float(r["Process temperature [K]"]),
        "temperature_difference_K": round(float(r["Process temperature [K]"]) - float(r["Air temperature [K]"]), 2),
        "rotational_speed_rpm": int(r["Rotational speed [rpm]"]),
        "torque_Nm": float(r["Torque [Nm]"]),
        "tool_wear_min": int(r["Tool wear [min]"]),
        "machine_failure": bool(r["Machine failure"] == 1),
        "failure_modes": {
            "TWF": bool(r["TWF"] == 1),
            "HDF": bool(r["HDF"] == 1),
            "PWF": bool(r["PWF"] == 1),
            "OSF": bool(r["OSF"] == 1),
            "RNF": bool(r["RNF"] == 1),
        }
    }


def filter_high_risk_milling_machines(min_tool_wear: int = 200, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Filter milling machines at high operational risk (e.g. tool wear >= min_tool_wear).
    """
    df = _get_df()
    high_risk = df[df["Tool wear [min]"] >= min_tool_wear].head(limit)
    res = []
    for _, r in high_risk.iterrows():
        res.append({
            "udi": int(r["UDI"]),
            "product_id": str(r["Product ID"]),
            "type": str(r["Type"]),
            "tool_wear_min": int(r["Tool wear [min]"]),
            "torque_Nm": float(r["Torque [Nm]"]),
            "speed_rpm": int(r["Rotational speed [rpm]"]),
            "failure_recorded": bool(r["Machine failure"] == 1)
        })
    return res
