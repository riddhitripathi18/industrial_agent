"""
unified_predictor.py
====================
Unified Predictive Maintenance Inference Engine across 3 Machine Classes:
  1. CNC Milling Machines (ai4i2020 dataset)
  2. Heat Exchangers (heat_exchanger_fouling dataset)
  3. Rotating Bearings (NASA bearing vibration dataset)
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

BASE_DIR = r"c:\Users\riddh\OneDrive\Desktop\predictive_maintenance"
MODELS_DIR = os.path.join(BASE_DIR, "models")


class UnifiedPredictor:
    """Multi-asset predictive maintenance inference engine."""

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or MODELS_DIR
        self._cache = {}

    def _load(self, filename: str) -> Any:
        if filename not in self._cache:
            path = os.path.join(self.models_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file '{filename}' not found in {self.models_dir}")
            with open(path, "rb") as f:
                self._cache[filename] = pickle.load(f)
        return self._cache[filename]

    # -----------------------------------------------------------------------
    # 1. CNC Milling Machine Prediction
    # -----------------------------------------------------------------------
    def predict_milling(
        self,
        air_temp_k: float,
        process_temp_k: float,
        rotational_speed_rpm: float,
        torque_nm: float,
        tool_wear_min: float,
        product_type: str = "M"
    ) -> Dict[str, Any]:
        """
        Predict binary failure and specific failure modes for a milling machine.
        Failure Modes: TWF (Tool Wear), HDF (Heat Dissipation), PWF (Power),
                       OSF (Overstrain), RNF (Random Failure).
        """
        scaler = self._load("scaler.pkl")
        clf = self._load("xgb_model.pkl")
        feature_cols = self._load("feature_cols.pkl")

        # Type encoding L=0, M=1, H=2
        type_map = {"L": 0, "M": 1, "H": 2}
        type_val = type_map.get(str(product_type).upper(), 1)

        # Physics-informed & heuristic features
        power_watts = (2.0 * math.pi * rotational_speed_rpm * torque_nm) / 60.0
        temp_diff_k = process_temp_k - air_temp_k
        wear_torque = tool_wear_min * torque_nm
        wear_pct = tool_wear_min / 253.0
        high_torque = 1 if torque_nm > 55 else 0
        high_wear = 1 if tool_wear_min > 200 else 0

        row_dict = {
            "Type_encoded": type_val,
            "Air temperature [K]": air_temp_k,
            "Process temperature [K]": process_temp_k,
            "Rotational speed [rpm]": rotational_speed_rpm,
            "Torque [Nm]": torque_nm,
            "Tool wear [min]": tool_wear_min,
            "power_watts": power_watts,
            "temp_diff_K": temp_diff_k,
            "wear_torque_product": wear_torque,
            "wear_pct": wear_pct,
            "high_torque": high_torque,
            "high_wear": high_wear
        }

        df_input = pd.DataFrame([row_dict])[feature_cols]
        X_scaled = scaler.transform(df_input)

        failure_prob = float(clf.predict_proba(X_scaled)[0, 1])
        is_failure = bool(failure_prob >= 0.5)

        # Individual failure modes
        modes = {}
        for mode in ["twf", "hdf", "pwf", "osf", "rnf"]:
            try:
                m_model = self._load(f"xgb_{mode}_model.pkl")
                m_prob = float(m_model.predict_proba(X_scaled)[0, 1])
                modes[mode.upper()] = {
                    "probability": round(m_prob, 4),
                    "flag": bool(m_prob >= 0.5)
                }
            except Exception:
                modes[mode.upper()] = {"probability": 0.0, "flag": False}

        # Recommendations
        rec = []
        if modes.get("TWF", {}).get("flag") or tool_wear_min >= 200:
            rec.append("Tool wear exceeds threshold — schedule immediate tool replacement.")
        if modes.get("HDF", {}).get("flag") or temp_diff_k < 8.6:
            rec.append("Heat dissipation inadequate — check spindle coolant and air circulation.")
        if modes.get("PWF", {}).get("flag") or power_watts < 3500 or power_watts > 9000:
            rec.append("Power draw abnormal — inspect motor windings and electrical drive.")
        if modes.get("OSF", {}).get("flag"):
            rec.append("Mechanical overstrain detected — reduce cutting feed rate or torque load.")
        if not rec:
            rec.append("Operating within normal machine parameters.")

        return {
            "machine_type": "CNC Milling Machine",
            "overall_failure_probability": round(failure_prob, 4),
            "predicted_failure": is_failure,
            "failure_modes": modes,
            "physics_metrics": {
                "power_watts": round(power_watts, 2),
                "temp_diff_K": round(temp_diff_k, 2),
                "wear_torque_product": round(wear_torque, 2)
            },
            "recommendation": " ".join(rec)
        }

    # -----------------------------------------------------------------------
    # 2. Heat Exchanger Fouling Prediction
    # -----------------------------------------------------------------------
    def predict_heat_exchanger(
        self,
        exchanger_id: str,
        tube_t_in: float,
        tube_t_out: float,
        tube_flow: float,
        shell_t_in: float,
        shell_t_out: float,
        shell_flow: float,
        time_hr: float = 800.0,
        crude_api: float = 32.0,
        crude_tan: float = 0.8,
        crude_chlorides: float = 15.0
    ) -> Dict[str, Any]:
        """
        Predict fouling resistance (Rf) and TEMA limit breach probability for a Heat Exchanger.
        """
        clf = self._load("hx_fouling_classifier.pkl")
        reg = self._load("hx_fouling_regressor.pkl")
        scaler = self._load("hx_scaler.pkl")
        feature_cols = self._load("hx_feature_cols.pkl")

        exchanger_id = exchanger_id.upper()
        areas = {"E01": 105.0, "E02": 110.0, "E03": 120.0, "E04": 95.0, "E05": 85.0}
        area = areas.get(exchanger_id, 100.0)

        delta_t_tube = abs(tube_t_out - tube_t_in)
        delta_t_shell = abs(shell_t_in - shell_t_out)
        q_kw = tube_flow * 2.1 * delta_t_tube

        dt1 = abs(shell_t_in - tube_t_out)
        dt2 = abs(shell_t_out - tube_t_in)
        if dt1 > 0 and dt2 > 0 and dt1 != dt2:
            lmtd = (dt1 - dt2) / math.log(dt1 / dt2)
        else:
            lmtd = max((dt1 + dt2) / 2.0, 1.0)

        row = {
            "time_hr": time_hr,
            "tube_t_in": tube_t_in,
            "tube_t_out": tube_t_out,
            "tube_flow": tube_flow,
            "shell_t_in": shell_t_in,
            "shell_t_out": shell_t_out,
            "shell_flow": shell_flow,
            "delta_t_tube": delta_t_tube,
            "delta_t_shell": delta_t_shell,
            "q_duty_kw": q_kw,
            "lmtd": lmtd,
            "crude_api": crude_api,
            "crude_tan": crude_tan,
            "crude_chlorides": crude_chlorides,
        }

        # Add one-hot columns
        for hx in ["E01", "E02", "E03", "E04", "E05"]:
            row[f"exchanger_id_{hx}"] = 1.0 if hx == exchanger_id else 0.0

        # Construct dataframe matching feature_cols
        row_clean = {c: row.get(c, 0.0) for c in feature_cols}
        df_input = pd.DataFrame([row_clean])[feature_cols]

        X_scaled = scaler.transform(df_input)
        tema_breach_prob = float(clf.predict_proba(X_scaled)[0, 1])
        predicted_rf = float(reg.predict(X_scaled)[0])

        status = "CRITICAL FOULING" if tema_breach_prob >= 0.5 else "OPERATING NORMAL"
        action = "Schedule chemical or mechanical cleaning procedure immediately." if tema_breach_prob >= 0.5 else "Continue regular condition monitoring."

        return {
            "machine_type": "Shell & Tube Heat Exchanger",
            "exchanger_id": exchanger_id,
            "tema_breach_probability": round(tema_breach_prob, 4),
            "predicted_rf_m2KW": round(predicted_rf, 7),
            "tema_limit_m2KW": 0.0005,
            "status": status,
            "thermal_metrics": {
                "duty_kW": round(q_kw, 2),
                "lmtd_C": round(lmtd, 2),
                "delta_t_tube_C": round(delta_t_tube, 2),
                "delta_t_shell_C": round(delta_t_shell, 2)
            },
            "recommendation": action
        }

    # -----------------------------------------------------------------------
    # 3. Rotating Bearing Vibration Prediction
    # -----------------------------------------------------------------------
    def predict_bearing(
        self,
        bearing_id: str = "Bearing1",
        rms: float = 0.15,
        kurtosis: float = 3.5,
        peak: float = 0.45,
        peak_to_peak: float = 0.90,
        crest_factor: float = 3.0,
        hours_elapsed: float = 100.0,
        test_id: int = 2
    ) -> Dict[str, Any]:
        """
        Predict degradation stage and remaining useful life (RUL) for a bearing.
        """
        clf = self._load("bearing_stage_classifier.pkl")
        reg = self._load("bearing_rul_regressor.pkl")
        scaler = self._load("bearing_scaler.pkl")
        feature_cols = self._load("bearing_feature_cols.pkl")

        row = {
            "test_id": test_id,
            "hours_elapsed": hours_elapsed,
            "rms": rms,
            "peak": peak,
            "peak_to_peak": peak_to_peak,
            "kurtosis": kurtosis,
            "crest_factor": crest_factor,
            "std": rms * 0.95,
            "skewness": 0.1,
            "variance": rms ** 2,
            "mean_abs": rms * 0.8,
            "shape_factor": 1.25,
            "impulse_factor": 3.8,
        }

        # One-hot bearing columns
        for b in ["Bearing1", "Bearing2", "Bearing3", "Bearing4"]:
            row[f"bearing_id_{b}"] = 1.0 if b.lower() == bearing_id.lower() else 0.0

        row_clean = {c: row.get(c, 0.0) for c in feature_cols}
        df_input = pd.DataFrame([row_clean])[feature_cols]

        X_scaled = scaler.transform(df_input)
        stage_idx = int(clf.predict(X_scaled)[0])
        stage_probs = clf.predict_proba(X_scaled)[0]
        rul_pred = float(reg.predict(X_scaled)[0])

        stages = ["Healthy (Normal)", "Warning (Degrading)", "Critical (Failure Imminent)"]
        stage_label = stages[stage_idx]

        if stage_idx == 2:
            rec = "CRITICAL: Outer/inner race failure imminent. Immediate replacement required."
        elif stage_idx == 1:
            rec = "ADVISORY: Accelerated vibration detected. Inspect lubrication and schedule maintenance."
        else:
            rec = "Normal bearing vibration. Safe for continuous operation."

        return {
            "machine_type": "Rotating Bearing (Accelerometry)",
            "bearing_id": bearing_id,
            "degradation_stage": stage_label,
            "stage_probabilities": {stages[i]: round(float(stage_probs[i]), 4) for i in range(len(stages))},
            "predicted_rul_hours": round(max(rul_pred, 0.0), 1),
            "vibration_metrics": {
                "rms": round(rms, 4),
                "kurtosis": round(kurtosis, 4),
                "crest_factor": round(crest_factor, 4)
            },
            "recommendation": rec
        }
