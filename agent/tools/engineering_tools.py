"""
Engineering Tools — First-Principles Physics Calculations
===========================================================
All functions use verified engineering equations from:
  - TEMA (Tubular Exchanger Manufacturers Association) standards
  - ISO 10816 vibration severity classification
  - Classical heat transfer textbooks (Incropera, Kern)

Every function is LLM-tool-ready:
  - Type-hinted arguments
  - Descriptive docstrings
  - JSON-serializable dict output

Functions:
    calc_overall_heat_transfer_coeff()  U-value from live telemetry
    calc_fouling_resistance()            Rf trend and TEMA breach detection
    calc_energy_penalty()                Dollar cost of heat loss due to fouling
    calc_bearing_rul()                   Remaining Useful Life via exponential fit
    iso_10816_severity()                 ISO zone classification for RMS vibration
"""

from __future__ import annotations

import math
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from typing import Optional

from agent.config import (
    HX_EXCHANGER_META,
    ISO_10816_ZONES,
    BEARING_TEST_META,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Approximate heat transfer area per exchanger shell pass (m²)
# In absence of exact geometry, use a typical refinery crude preheat train value.
# All 5 exchangers in this dataset are the same shell-and-tube type.
_HX_AREA_M2: dict[str, float] = {
    "E01": 180.0,
    "E02": 180.0,
    "E03": 180.0,
    "E04": 180.0,
    "E05": 180.0,
}

# Furnace efficiency (fraction) — energy needed to make up for lost preheat
_FURNACE_ETA: float = 0.88

# BTU/hr to kW conversion
_BTU_HR_PER_KW: float = 3412.14


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lmtd(t_hot_in: float, t_hot_out: float,
          t_cold_in: float, t_cold_out: float,
          flow_config: str = "counter") -> float:
    """
    Log Mean Temperature Difference for shell-and-tube exchangers.
    Assumes counter-current flow (most refinery HX configurations).
    Returns 0.0 if temperatures are degenerate.
    """
    if flow_config == "counter":
        dt1 = t_hot_in  - t_cold_out
        dt2 = t_hot_out - t_cold_in
    else:  # co-current / parallel
        dt1 = t_hot_in  - t_cold_in
        dt2 = t_hot_out - t_cold_out

    if dt1 <= 0 or dt2 <= 0:
        return 0.0
    if abs(dt1 - dt2) < 1e-6:
        return dt1
    return (dt1 - dt2) / math.log(dt1 / dt2)


def _to_list(arr) -> list:
    return [round(float(v), 6) if not math.isnan(float(v)) else None for v in arr]


# ---------------------------------------------------------------------------
# 1. Overall Heat Transfer Coefficient
# ---------------------------------------------------------------------------

def calc_overall_heat_transfer_coeff(
    telemetry: dict,
    exchanger_id: str,
) -> dict:
    """
    Compute the Overall Heat Transfer Coefficient U (W/m²·K) at every timestep.

    Uses the crude (tube-side) energy balance:
        Q  = m_dot * Cp * delta_T      [kW]
        U  = Q / (A * LMTD)            [W/m²·K]

    Args:
        telemetry:    Output dict from get_hx_telemetry().
        exchanger_id: One of 'E01' ... 'E05'.

    Returns:
        dict with keys:
            exchanger_id, time_hr, Q_kW, LMTD_K, U_W_m2K,
            U_clean (mean of first 5% — baseline),
            A_m2 (heat transfer area assumed), n_points.

    Note:
        U values for crude oil preheat exchangers typically range 100–500 W/m²·K.
        Values outside this range may indicate sensor noise or process upsets.

    Example:
        >>> data    = get_hx_telemetry("E02")
        >>> result  = calc_overall_heat_transfer_coeff(data, "E02")
        >>> result["U_clean"]
        342.7
    """
    eid   = exchanger_id.upper()
    meta  = HX_EXCHANGER_META[eid]
    cp_kJ = meta["tube_cp_kJ_kgK"]    # crude oil Cp  [kJ/kg·K]
    A     = _HX_AREA_M2[eid]          # shell area    [m²]

    t_in    = np.array(telemetry["tube_temp_in"],         dtype=float)
    t_out   = np.array(telemetry["tube_temp_out"],        dtype=float)
    m_dot   = np.array(telemetry["tube_mass_flow_kg_s"],  dtype=float)
    sh_in   = np.array(telemetry["shell_temp_in"],        dtype=float)
    sh_out  = np.array(telemetry["shell_temp_out"],       dtype=float)
    time_hr = np.array(telemetry["time_hr"],              dtype=float)

    n = len(t_in)
    Q_kW   = np.zeros(n)
    lmtd_v = np.zeros(n)
    U_v    = np.zeros(n)

    for i in range(n):
        # Tube-side heat duty [kW]
        q = m_dot[i] * cp_kJ * (t_out[i] - t_in[i])
        Q_kW[i] = max(q, 0.0)   # negative values = sensor noise

        # LMTD (counter-current): hot = shell, cold = tube
        lm = _lmtd(sh_in[i], sh_out[i], t_in[i], t_out[i])
        lmtd_v[i] = lm

        # U [W/m²·K]  (Q in kW → multiply by 1000)
        if lm > 0 and A > 0:
            U_v[i] = (Q_kW[i] * 1000.0) / (A * lm)
        else:
            U_v[i] = float("nan")

    # Baseline U: mean of first 5% of run (clean exchanger)
    n5        = max(1, int(n * 0.05))
    first_5pct = U_v[:n5]
    valid_u0   = first_5pct[~np.isnan(first_5pct)]
    U_clean    = float(np.mean(valid_u0)) if len(valid_u0) > 0 else float(np.nanmean(U_v) if np.any(~np.isnan(U_v)) else 0.0)

    return {
        "exchanger_id": eid,
        "A_m2":         A,
        "U_clean":      round(U_clean, 4),
        "time_hr":      _to_list(time_hr),
        "Q_kW":         _to_list(Q_kW),
        "LMTD_K":       _to_list(lmtd_v),
        "U_W_m2K":      _to_list(U_v),
        "n_points":     n,
        "note":         f"Cp={cp_kJ} kJ/kg·K, A={A} m² assumed for {eid}",
    }


# ---------------------------------------------------------------------------
# 2. Fouling Resistance
# ---------------------------------------------------------------------------

def calc_fouling_resistance(
    telemetry: dict,
    exchanger_id: str,
    tema_limit: float = 0.0005,
) -> dict:
    """
    Compute the Fouling Resistance Rf (m²·K/W) at every timestep.

    Formula:
        Rf(t) = 1/U_dirty(t)  -  1/U_clean

    TEMA standard maximum allowable Rf for crude oil = 0.0005 m²·K/W.

    Args:
        telemetry:    Output dict from get_hx_telemetry().
        exchanger_id: One of 'E01' ... 'E05'.
        tema_limit:   TEMA fouling resistance threshold (default 0.0005 m²·K/W).

    Returns:
        dict with keys:
            exchanger_id, time_hr, Rf, U_clean, tema_limit,
            tema_breach_hour (first hour Rf exceeds limit, or None),
            tema_breached (bool), n_points.

    Example:
        >>> data   = get_hx_telemetry("E01")
        >>> result = calc_fouling_resistance(data, "E01")
        >>> result["tema_breached"]
        True
    """
    u_result = calc_overall_heat_transfer_coeff(telemetry, exchanger_id)
    U_clean  = u_result["U_clean"]
    U_arr    = np.array(u_result["U_W_m2K"], dtype=float)
    time_arr = np.array(u_result["time_hr"],  dtype=float)

    Rf = np.where(
        (U_arr > 0) & (U_clean > 0),
        (1.0 / U_arr) - (1.0 / U_clean),
        float("nan"),
    )
    # Clip negatives (early run noise) to zero
    Rf = np.where(Rf < 0, 0.0, Rf)

    # TEMA breach detection
    breach_mask = Rf > tema_limit
    tema_breach_hour = None
    if breach_mask.any():
        first_idx = int(np.argmax(breach_mask))
        tema_breach_hour = round(float(time_arr[first_idx]), 2)

    valid_rf = Rf[~np.isnan(Rf)]
    rf_max   = round(float(np.nanmax(valid_rf)), 8) if len(valid_rf) > 0 else None
    rf_final = round(float(valid_rf[-1]), 8)         if len(valid_rf) > 0 else None

    return {
        "exchanger_id":    exchanger_id.upper(),
        "U_clean":         round(U_clean, 4),
        "tema_limit":      tema_limit,
        "tema_breached":   bool(breach_mask.any()),
        "tema_breach_hour": tema_breach_hour,
        "time_hr":         _to_list(time_arr),
        "Rf":              _to_list(Rf),
        "Rf_max":          rf_max,
        "Rf_final":        rf_final,
        "n_points":        len(Rf),
    }


# ---------------------------------------------------------------------------
# 3. Energy Penalty
# ---------------------------------------------------------------------------

def calc_energy_penalty(
    telemetry: dict,
    exchanger_id: str,
    fuel_price_per_mmbtu: float = 12.0,
) -> dict:
    """
    Quantify the financial cost of heat loss due to fouling ($/hour and cumulative).

    Formula:
        Q_lost(t)     = Q_clean - Q_dirty(t)             [kW]
        Cost ($/hr)   = Q_lost * 3412 / (eta * 1e6) * fuel_price
                      where 3412 = BTU/hr per kW, 1e6 = BTU/MMBtu

    Args:
        telemetry:            Output from get_hx_telemetry().
        exchanger_id:         One of 'E01' ... 'E05'.
        fuel_price_per_mmbtu: Fuel cost in USD/MMBtu (default $12/MMBtu — typical refinery).

    Returns:
        dict with keys:
            exchanger_id, fuel_price_per_mmbtu, time_hr,
            Q_kW, Q_clean_kW, Q_lost_kW,
            cost_per_hr_USD, cumulative_loss_USD,
            total_loss_USD (over full run), n_points.

    Example:
        >>> data   = get_hx_telemetry("E01")
        >>> result = calc_energy_penalty(data, "E01", fuel_price_per_mmbtu=12.0)
        >>> result["total_loss_USD"]
        48320.5
    """
    u_result  = calc_overall_heat_transfer_coeff(telemetry, exchanger_id)
    Q_arr     = np.array(u_result["Q_kW"],    dtype=float)
    time_arr  = np.array(u_result["time_hr"], dtype=float)

    # Baseline clean heat duty (first 5%)
    n5        = max(1, int(len(Q_arr) * 0.05))
    Q_clean   = float(np.nanmean(Q_arr[:n5]))

    Q_lost    = np.maximum(Q_clean - Q_arr, 0.0)   # kW lost

    # Cost per hour [USD/hr]
    # Q_lost [kW] × 3412.14 [BTU/hr per kW] / 1e6 [BTU/MMBtu] × price [$/MMBtu]
    cost_per_hr = Q_lost * (_BTU_HR_PER_KW / 1e6) * fuel_price_per_mmbtu / _FURNACE_ETA

    # Cumulative loss via trapezoidal integration over time [USD]
    dt_hr        = np.diff(time_arr, prepend=time_arr[0])
    cum_loss     = np.cumsum(cost_per_hr * dt_hr)
    total_loss   = float(cum_loss[-1]) if len(cum_loss) > 0 else 0.0

    return {
        "exchanger_id":        exchanger_id.upper(),
        "fuel_price_per_mmbtu": fuel_price_per_mmbtu,
        "furnace_efficiency":   _FURNACE_ETA,
        "Q_clean_kW":          round(Q_clean, 4),
        "time_hr":             _to_list(time_arr),
        "Q_kW":                _to_list(Q_arr),
        "Q_lost_kW":           _to_list(Q_lost),
        "cost_per_hr_USD":     _to_list(cost_per_hr),
        "cumulative_loss_USD": _to_list(cum_loss),
        "total_loss_USD":      round(total_loss, 2),
        "n_points":            len(Q_arr),
    }


# ---------------------------------------------------------------------------
# 4. Bearing Remaining Useful Life
# ---------------------------------------------------------------------------

def _exp_model(t, a, b):
    """Exponential degradation model: y = a * exp(b * t)."""
    return a * np.exp(b * t)


def calc_bearing_rul(
    telemetry: dict,
    bearing_id: str,
    metric: str = "rms",
    danger_threshold: Optional[float] = None,
) -> dict:
    """
    Estimate Remaining Useful Life (RUL) for a bearing via exponential curve fitting.

    Fits y = a·exp(b·t) to the metric trajectory, then extrapolates forward to
    find when the signal crosses the ISO Zone C and Zone D thresholds.

    Args:
        telemetry:        Output from get_bearing_telemetry().
        bearing_id:       Bearing channel name (e.g. 'Bearing1', 'Bearing3_x').
        metric:           Vibration metric to use (default 'rms').
        danger_threshold: Custom failure threshold. If None, uses ISO Zone D
                          lower bound (7.1 for RMS) or 3× mean for other metrics.

    Returns:
        dict with keys:
            bearing_id, metric, fit_a, fit_b, r_squared,
            current_value, current_hour,
            hours_to_zone_c, hours_to_zone_d,
            rul_hours, rul_status, fit_quality.

    Example:
        >>> data   = get_bearing_telemetry(2, "Bearing1", ["rms"])
        >>> result = calc_bearing_rul(data, "Bearing1", metric="rms")
        >>> result["rul_hours"]
        42.3
    """
    values = telemetry["metrics"].get(metric, [])
    hours  = telemetry["hours_elapsed"]

    if not values or not hours:
        return {"error": f"No data for metric '{metric}' in telemetry."}

    t  = np.array(hours,  dtype=float)
    y  = np.array(values, dtype=float)

    # Remove NaNs
    mask = ~(np.isnan(t) | np.isnan(y))
    t, y = t[mask], y[mask]

    if len(t) < 10:
        return {"error": "Insufficient data points for curve fitting (need >= 10)."}

    # Clamp negatives (physical floor)
    y = np.maximum(y, 1e-9)

    # Exponential fit
    try:
        p0  = [float(y[0]), 0.001]
        popt, _ = curve_fit(_exp_model, t, y, p0=p0, maxfev=10000)
        a, b = popt
    except RuntimeError:
        # Fallback: linear fit in log-space
        log_y   = np.log(y)
        slope, intercept, *_ = stats.linregress(t, log_y)
        a = math.exp(intercept)
        b = slope

    # R² in original space
    y_pred   = _exp_model(t, a, b)
    ss_res   = np.sum((y - y_pred) ** 2)
    ss_tot   = np.sum((y - np.mean(y)) ** 2)
    r2       = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Thresholds
    if metric == "rms":
        thresh_c = ISO_10816_ZONES["C"][0]   # 4.5
        thresh_d = ISO_10816_ZONES["D"][0]   # 7.1
    else:
        # Generic: zone C = 2× mean, zone D = 3× mean
        mean_val = float(np.mean(y))
        thresh_c = mean_val * 2.0
        thresh_d = danger_threshold if danger_threshold else mean_val * 3.0

    if danger_threshold is not None:
        thresh_d = danger_threshold

    # Time to threshold: t = ln(threshold/a) / b
    def _time_to_threshold(thr: float) -> Optional[float]:
        if a <= 0 or b <= 0:
            return None   # signal not growing exponentially
        try:
            t_hit = math.log(thr / a) / b
            if t_hit <= t[-1]:
                return None   # already exceeded
            return round(float(t_hit - t[-1]), 2)
        except (ValueError, ZeroDivisionError):
            return None

    hours_to_c = _time_to_threshold(thresh_c)
    hours_to_d = _time_to_threshold(thresh_d)
    rul        = hours_to_d   # primary RUL = time to danger zone

    # Current values
    current_val  = round(float(y[-1]), 6)
    current_hour = round(float(t[-1]), 2)

    # Determine ISO zone for current value
    current_zone = "Unknown"
    if metric == "rms":
        for zone, (lo, hi) in ISO_10816_ZONES.items():
            if lo <= current_val < hi:
                current_zone = zone
                break

    # Fit quality label
    if r2 >= 0.9:
        fit_quality = "excellent"
    elif r2 >= 0.7:
        fit_quality = "good"
    elif r2 >= 0.5:
        fit_quality = "fair"
    else:
        fit_quality = "poor — exponential may not be best model"

    return {
        "bearing_id":     bearing_id,
        "metric":         metric,
        "fit_a":          round(float(a), 8),
        "fit_b":          round(float(b), 8),
        "r_squared":      round(r2, 4),
        "fit_quality":    fit_quality,
        "current_value":  current_val,
        "current_hour":   current_hour,
        "current_iso_zone": current_zone,
        "threshold_zone_c": round(thresh_c, 4),
        "threshold_zone_d": round(thresh_d, 4),
        "hours_to_zone_c":  hours_to_c,
        "hours_to_zone_d":  hours_to_d,
        "rul_hours":         rul,
        "rul_status":        (
            "CRITICAL — already in danger zone" if hours_to_d is None and current_val >= thresh_d
            else "WARNING — approaching danger" if hours_to_d is not None and hours_to_d < 50
            else "MONITOR" if hours_to_d is not None and hours_to_d < 200
            else "HEALTHY"
        ),
    }


# ---------------------------------------------------------------------------
# 5. ISO 10816 Severity
# ---------------------------------------------------------------------------

def iso_10816_severity(rms_value: float) -> dict:
    """
    Classify a vibration RMS value (mm/s) per ISO 10816 zones (Class II machinery).

    ISO 10816 Zone definitions for medium-sized rotating machines (15 kW – 75 kW):
        Zone A (Good):        0.0 – 2.3 mm/s  — new or recently overhauled machine
        Zone B (Acceptable):  2.3 – 4.5 mm/s  — suitable for unrestricted long-term operation
        Zone C (Restricted):  4.5 – 7.1 mm/s  — marginal, plan maintenance within weeks
        Zone D (Danger):      > 7.1 mm/s       — STOP MACHINE, risk of damage

    Args:
        rms_value: Vibration RMS velocity in mm/s (or any consistent unit).

    Returns:
        dict with keys:
            rms_value, zone, zone_description, action_required,
            distance_to_next_zone, distance_to_danger.

    Example:
        >>> result = iso_10816_severity(3.5)
        >>> result["zone"]
        'B'
    """
    zone_labels = {
        "A": "Good — new or recently overhauled machine",
        "B": "Acceptable — suitable for unrestricted long-term operation",
        "C": "Restricted operation — plan maintenance within weeks",
        "D": "DANGER — stop machine, risk of immediate damage",
    }
    actions = {
        "A": "No action required. Continue normal operation.",
        "B": "No action required. Schedule next routine inspection.",
        "C": "Plan maintenance within 2–4 weeks. Monitor closely.",
        "D": "STOP MACHINE IMMEDIATELY. Investigate and repair before restarting.",
    }

    current_zone = "D"
    for zone, (lo, hi) in ISO_10816_ZONES.items():
        if lo <= rms_value < hi:
            current_zone = zone
            break

    # Distance to next zone boundary
    zone_order    = ["A", "B", "C", "D"]
    zone_bounds   = [2.3, 4.5, 7.1, float("inf")]
    current_idx   = zone_order.index(current_zone)
    next_boundary = zone_bounds[current_idx]
    dist_to_next  = max(0.0, round(next_boundary - rms_value, 4)) if next_boundary != float("inf") else None
    dist_to_danger = max(0.0, round(7.1 - rms_value, 4)) if rms_value < 7.1 else 0.0

    return {
        "rms_value":            round(float(rms_value), 6),
        "zone":                 current_zone,
        "zone_description":     zone_labels[current_zone],
        "action_required":      actions[current_zone],
        "distance_to_next_zone": dist_to_next,
        "distance_to_danger":   dist_to_danger,
        "zone_boundaries":      {"A_max": 2.3, "B_max": 4.5, "C_max": 7.1},
    }
