"""
ML Prediction Tools — Unified Multi-Asset Inference
===================================================
Connects the trained predictive models in the predictive_maintenance repository
to the IPMA Agent for real-time failure prediction on:
  1. CNC Milling Machines (ai4i2020)
  2. Heat Exchangers (heat_exchanger_fouling)
  3. Rotating Bearings (NASA vibration test sets)
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add predictive_maintenance to path for UnifiedPredictor
PRED_REPO_PATH = Path(r"c:\Users\riddh\OneDrive\Desktop\predictive_maintenance")
if str(PRED_REPO_PATH) not in sys.path and PRED_REPO_PATH.exists():
    sys.path.insert(0, str(PRED_REPO_PATH))

try:
    from unified_predictor import UnifiedPredictor
    _predictor = UnifiedPredictor(str(PRED_REPO_PATH / "models"))
except Exception as e:
    _predictor = None

from agent.data_tools.milling_data import get_milling_machine_telemetry
from agent.data_tools.hx_data import get_hx_telemetry
from agent.data_tools.bearing_data import get_bearing_telemetry


def predict_milling_machine(machine_id: int = 142) -> Dict[str, Any]:
    """
    Run trained XGBoost ML failure prediction on a CNC Milling Machine.
    Args:
        machine_id: UDI number of the machine (1 to 10000).
    Returns:
        dict: Overall failure probability, specific failure mode probabilities
              (TWF, HDF, PWF, OSF, RNF), physics features, and maintenance recommendation.
    """
    if _predictor is None:
        return {"error": "UnifiedPredictor ML models not available."}
    
    tel = get_milling_machine_telemetry(machine_id)
    if "error" in tel:
        return tel

    pred = _predictor.predict_milling(
        air_temp_k=tel["air_temperature_K"],
        process_temp_k=tel["process_temperature_K"],
        rotational_speed_rpm=tel["rotational_speed_rpm"],
        torque_nm=tel["torque_Nm"],
        tool_wear_min=tel["tool_wear_min"],
        product_type=tel["type"]
    )
    pred["machine_id"] = machine_id
    pred["product_id"] = tel["product_id"]
    return pred


def predict_hx_fouling(exchanger_id: str = "E02") -> Dict[str, Any]:
    """
    Run trained XGBoost ML fouling prediction on a Heat Exchanger.
    Args:
        exchanger_id: One of 'E01', 'E02', 'E03', 'E04', 'E05'.
    Returns:
        dict: TEMA limit breach probability, predicted fouling resistance Rf,
              thermal duty, and operational action recommendation.
    """
    if _predictor is None:
        return {"error": "UnifiedPredictor ML models not available."}

    exchanger_id = exchanger_id.upper()
    try:
        tel = get_hx_telemetry(exchanger_id)
        # Grab latest data point
        idx = -1
        t_in = tel["tube_temp_in"][idx]
        t_out = tel["tube_temp_out"][idx]
        t_flow = tel["tube_mass_flow_kg_s"][idx]
        s_in = tel["shell_temp_in"][idx]
        s_out = tel["shell_temp_out"][idx]
        s_flow = tel["shell_mass_flow_kg_s"][idx]
        t_hr = tel["time_hr"][idx]
        tan = tel["crude_tan"][idx]

        pred = _predictor.predict_heat_exchanger(
            exchanger_id=exchanger_id,
            tube_t_in=t_in,
            tube_t_out=t_out,
            tube_flow=t_flow,
            shell_t_in=s_in,
            shell_t_out=s_out,
            shell_flow=s_flow,
            time_hr=t_hr,
            crude_tan=tan
        )
        return pred
    except Exception as e:
        return {"error": f"Failed to predict fouling on {exchanger_id}: {str(e)}"}


def predict_bearing_failure(test_id: int = 2, bearing_id: str = "Bearing1") -> Dict[str, Any]:
    """
    Run trained XGBoost ML degradation stage and RUL prediction on a Bearing.
    Args:
        test_id: Test campaign ID (1, 2, or 3).
        bearing_id: Bearing name (e.g. 'Bearing1', 'Bearing2').
    Returns:
        dict: Predicted degradation stage, class probabilities, estimated Remaining
              Useful Life (RUL in hours), vibration metrics, and recommendation.
    """
    if _predictor is None:
        return {"error": "UnifiedPredictor ML models not available."}

    try:
        tel = get_bearing_telemetry(test_id, bearing_id)
        metrics = tel.get("metrics", {})
        rms_vals = [v for v in metrics.get("rms", []) if v is not None]
        kurt_vals = [v for v in metrics.get("kurtosis", []) if v is not None]
        peak_vals = [v for v in metrics.get("peak", []) if v is not None]
        hrs = tel.get("hours_elapsed", [100.0])

        rms = float(rms_vals[-1]) if rms_vals else 0.15
        kurt = float(kurt_vals[-1]) if kurt_vals else 3.0
        peak = float(peak_vals[-1]) if peak_vals else 0.4
        latest_hr = float(hrs[-1]) if hrs else 100.0

        pred = _predictor.predict_bearing(
            bearing_id=bearing_id,
            rms=rms,
            kurtosis=kurt,
            peak=peak,
            hours_elapsed=latest_hr,
            test_id=test_id
        )
        return pred
    except Exception as e:
        return {"error": f"Failed to predict bearing health for {bearing_id} in Test {test_id}: {str(e)}"}
