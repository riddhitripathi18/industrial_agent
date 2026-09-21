"""
Agent Core — Raw Gemini SDK Tool Loop
======================================
Wires all Phase 1-3 tools into a Gemini chat session with automatic
function calling enabled. The LLM decides which tools to call; the SDK
executes them and feeds results back automatically.

Architecture:
    User message
        → Gemini LLM (gemini-2.0-flash)
        → Picks tool(s) from registry
        → SDK auto-calls Python function
        → Result injected back into conversation
        → LLM synthesizes final answer
        → Back to user

Composite Agent Tools (simple args → Gemini can call them directly):
    list_equipment()                  → equipment catalog
    get_equipment_snapshot()          → full fleet status at a glance
    get_hx_data()                     → HX telemetry
    get_bearing_data()                → bearing vibration telemetry
    analyze_fouling()                 → Rf + energy penalty (fetches data internally)
    analyze_bearing()                 → RUL + trend + anomalies (fetches data internally)
    search_knowledge()                → RAG semantic search
"""

from __future__ import annotations

import os
import json
import time
import logging
from typing import Optional

import google.genai as genai
from google.genai import types as genai_types

from agent.prompts import SYSTEM_PROMPT
from agent.config  import HX_EXCHANGER_META, BEARING_TEST_META

# Phase 1 — data tools
from agent.data_tools.hx_data      import (
    get_available_exchangers,
    get_hx_telemetry,
    get_hx_snapshot,
    get_hx_operating_summary,
)
from agent.data_tools.bearing_data import (
    get_bearing_tests_overview,
    get_bearing_telemetry,
    get_bearing_health_snapshot,
    get_bearing_critical_events,
)

# Phase 2 — engineering + analysis tools
from agent.tools.engineering_tools import (
    calc_fouling_resistance,
    calc_energy_penalty,
    calc_bearing_rul,
    iso_10816_severity,
)
from agent.tools.analysis_tools import (
    trend_analyzer,
    anomaly_detector,
)

# Phase 3 — RAG
from agent.tools.rag_tools import query_knowledge_base as _query_kb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Composite Agent Tools
# (thin wrappers that accept SIMPLE args the LLM can fill in)
# ---------------------------------------------------------------------------

def list_equipment() -> dict:
    """
    List all available industrial equipment in the monitoring system.

    Returns a catalog of heat exchangers (E01-E05) and bearing test datasets
    (Test 1, 2, 3), including fluid types, dataset sizes, and failure info.

    Call this first if you are unsure what equipment exists.

    Returns:
        dict with keys: heat_exchangers (list), bearing_tests (list).
    """
    hx_list  = get_available_exchangers()
    brg_list = get_bearing_tests_overview()
    return {
        "heat_exchangers": hx_list,
        "bearing_tests":   brg_list,
    }


def get_equipment_snapshot() -> dict:
    """
    Get a real-time status snapshot of ALL equipment in the plant.

    Queries the current (latest) operating point for all 5 heat exchangers
    and all bearings across Test 1, 2, and 3. Use this for a fleet-wide
    health overview before drilling into a specific asset.

    Returns:
        dict with keys:
            heat_exchangers: current delta-T, flow, temperatures for E01-E05
            bearing_tests: RMS, kurtosis, ISO zone for every bearing at final snapshot
    """
    # Use a representative hour near end of campaign for each dataset
    # HX campaign: ~1000 hours; pick hour 800 as "current"
    hx_snap  = get_hx_snapshot(hour=800.0)
    brg_snap = {}
    for test_id in [1, 2, 3]:
        n_snap  = BEARING_TEST_META[test_id]["n_snapshots"]
        # Each snapshot is every 20 minutes = 1/3 hour
        snap_hr = round((n_snap / 3.0) * 0.8, 1)
        brg_snap[f"test_{test_id}"] = get_bearing_health_snapshot(test_id, hour=snap_hr)

    return {
        "heat_exchangers": hx_snap,
        "bearing_tests":   brg_snap,
    }


def get_hx_data(
    exchanger_id: str,
    start_hour: Optional[float] = None,
    end_hour: Optional[float] = None,
    sample_step: int = 10,
) -> dict:
    """
    Fetch telemetry data for a heat exchanger.

    Retrieves time-series measurements: tube-side temperatures (in/out),
    shell-side temperatures (in/out), mass flow rate, and crude oil TAN
    (Total Acid Number). Use this to inspect raw operating trends.

    Args:
        exchanger_id: Heat exchanger ID — one of 'E01', 'E02', 'E03', 'E04', 'E05'.
        start_hour:   Start of time window (hours from campaign start). None = beginning.
        end_hour:     End of time window (hours). None = end of campaign.
        sample_step:  Return every Nth data point (default 10 for manageable output).

    Returns:
        dict with time series: time_hr, tube_temp_in, tube_temp_out,
        shell_temp_in, shell_temp_out, tube_mass_flow_kg_s, crude_tan, n_points.
    """
    # Don't pass None args — use function defaults instead
    kwargs: dict = {"sample_step": sample_step}
    if start_hour is not None:
        kwargs["start_hour"] = start_hour
    if end_hour is not None:
        kwargs["end_hour"] = end_hour
    return get_hx_telemetry(exchanger_id=exchanger_id, **kwargs)


def get_bearing_data(
    test_id: int,
    bearing_id: str,
    metrics: Optional[list] = None,
    start_hour: Optional[float] = None,
    end_hour: Optional[float] = None,
) -> dict:
    """
    Fetch vibration telemetry for a specific bearing.

    Returns RMS velocity, kurtosis, and/or peak values over the test duration.
    Use this to inspect how a bearing's vibration evolved over time.

    Args:
        test_id:    Test run number — 1, 2, or 3.
        bearing_id: Bearing channel name. Options:
                    Test 1: 'Bearing1_x', 'Bearing1_y', 'Bearing2_x', 'Bearing2_y',
                            'Bearing3_x', 'Bearing3_y', 'Bearing4_x', 'Bearing4_y'
                    Test 2 & 3: 'Bearing1', 'Bearing2', 'Bearing3', 'Bearing4'
        metrics:    List of metrics to return. Options: ['rms', 'kurtosis', 'peak'].
                    Default ['rms', 'kurtosis'].
        start_hour: Start of time window in hours. None = beginning.
        end_hour:   End of time window in hours. None = end.

    Returns:
        dict with: test_id, bearing_id, hours_elapsed, metrics (dict of lists), n_points.
    """
    if metrics is None:
        metrics = ["rms", "kurtosis"]
    # Don't pass None args — use function defaults instead
    kwargs: dict = {"metrics": metrics}
    if start_hour is not None:
        kwargs["start_hour"] = start_hour
    if end_hour is not None:
        kwargs["end_hour"] = end_hour
    return get_bearing_telemetry(
        test_id=test_id,
        bearing_id=bearing_id,
        **kwargs,
    )


def analyze_fouling(
    exchanger_id: str,
    fuel_price_per_mmbtu: float = 12.0,
    start_hour: Optional[float] = None,
    end_hour: Optional[float] = None,
) -> dict:
    """
    Full fouling analysis for a heat exchanger.

    Computes fouling resistance Rf (m²·K/W) using the LMTD heat balance,
    checks against TEMA standard limit (0.0005 m²·K/W), and quantifies
    the energy and financial cost of heat loss due to fouling.

    This tool combines data fetching + engineering computation in one call.

    Args:
        exchanger_id:         One of 'E01', 'E02', 'E03', 'E04', 'E05'.
        fuel_price_per_mmbtu: Fuel cost in USD/MMBtu (default $12/MMBtu).
        start_hour:           Analysis window start (hours). None = full campaign.
        end_hour:             Analysis window end (hours). None = full campaign.

    Returns:
        dict with:
            exchanger_id, U_clean, U_final, Rf_final, Rf_max,
            tema_breached (bool), tema_breach_hour,
            total_energy_loss_USD, peak_cost_per_hr_USD,
            trend (slope, direction, projected_breach_hour),
            recommendation (string summary).

    Example:
        >>> result = analyze_fouling("E02")
        >>> result["tema_breached"]
        True
    """
    # Fetch telemetry — don't pass None args, use data tool defaults
    tel_kwargs: dict = {}
    if start_hour is not None:
        tel_kwargs["start_hour"] = start_hour
    if end_hour is not None:
        tel_kwargs["end_hour"] = end_hour
    tel = get_hx_telemetry(exchanger_id, **tel_kwargs)

    # Fouling resistance
    rf_result  = calc_fouling_resistance(tel, exchanger_id)
    # Energy penalty
    en_result  = calc_energy_penalty(tel, exchanger_id, fuel_price_per_mmbtu=fuel_price_per_mmbtu)
    # Trend on Rf
    rf_vals    = [v for v in rf_result["Rf"] if v is not None]
    hours      = [h for h, v in zip(rf_result["time_hr"], rf_result["Rf"]) if v is not None]
    trend      = trend_analyzer(
        rf_vals, hours,
        threshold=0.0005,
        label=f"{exchanger_id} Rf",
    ) if len(rf_vals) >= 3 else {}

    # U_final (last valid U value)
    from agent.tools.engineering_tools import calc_overall_heat_transfer_coeff
    u_res   = calc_overall_heat_transfer_coeff(tel, exchanger_id)
    u_vals  = [v for v in u_res["U_W_m2K"] if v is not None]
    u_final = round(float(u_vals[-1]), 2) if u_vals else None

    # Build recommendation string
    rec = _fouling_recommendation(
        exchanger_id,
        rf_result["Rf_final"],
        rf_result["tema_breached"],
        rf_result["tema_breach_hour"],
        en_result["total_loss_USD"],
        trend,
    )

    return {
        "exchanger_id":          exchanger_id.upper(),
        "U_clean_W_m2K":         rf_result["U_clean"],
        "U_final_W_m2K":         u_final,
        "U_degradation_pct":     round((1 - u_final / rf_result["U_clean"]) * 100, 1) if u_final else None,
        "Rf_final_m2KW":         rf_result["Rf_final"],
        "Rf_max_m2KW":           rf_result["Rf_max"],
        "tema_limit_m2KW":       0.0005,
        "tema_breached":         rf_result["tema_breached"],
        "tema_breach_hour":      rf_result["tema_breach_hour"],
        "total_energy_loss_USD": en_result["total_loss_USD"],
        "peak_cost_per_hr_USD":  round(max(v for v in en_result["cost_per_hr_USD"] if v is not None), 2),
        "fuel_price_per_mmbtu":  fuel_price_per_mmbtu,
        "trend":                 {
            "slope_per_hr":           trend.get("slope"),
            "direction":              trend.get("trend_direction"),
            "r_squared":              trend.get("r_squared"),
            "projected_breach_hour":  trend.get("projected_breach_hour"),
        },
        "recommendation": rec,
    }


def analyze_bearing(
    test_id: int,
    bearing_id: str,
    metric: str = "rms",
) -> dict:
    """
    Full health analysis for a bearing including RUL, ISO zone, trend, and anomalies.

    This tool combines data fetching + all engineering computations in one call.
    Use this when you need a complete condition assessment for a specific bearing.

    Args:
        test_id:    Test run — 1, 2, or 3.
        bearing_id: Bearing channel — e.g. 'Bearing1', 'Bearing3_x'.
        metric:     Primary metric for RUL and trend analysis.
                    Options: 'rms' (default), 'kurtosis', 'peak'.

    Returns:
        dict with:
            test_id, bearing_id, metric,
            current_rms, current_kurtosis, current_iso_zone,
            iso_zone_description, iso_action,
            rul_hours, rul_status,
            trend (slope, direction, projected_breach),
            n_anomalies, first_anomaly_hour, last_anomaly_hour,
            critical_events (list of high-vibration events),
            recommendation (string).

    Example:
        >>> result = analyze_bearing(2, "Bearing1")
        >>> result["rul_hours"]
        12.3
    """
    # Always import at top of function scope to avoid UnboundLocalError
    from agent.tools.engineering_tools import iso_10816_severity as _iso_severity

    # Fetch data — don't pass None args, use data tool defaults
    tel    = get_bearing_telemetry(test_id, bearing_id, metrics=["rms", "kurtosis"])
    n_snap = BEARING_TEST_META[test_id]["n_snapshots"]
    # Each snapshot is taken every 20 minutes = 1/3 hour
    duration_hr = round(n_snap / 3.0, 1)
    snap_hr     = round(duration_hr * 0.8, 1)
    snap   = get_bearing_health_snapshot(test_id, hour=snap_hr)
    events = get_bearing_critical_events(test_id)

    # RUL
    rul_result = calc_bearing_rul(tel, bearing_id, metric=metric)

    # Trend on chosen metric
    vals  = [v for v in tel["metrics"][metric] if v is not None]
    hours = [h for h, v in zip(tel["hours_elapsed"], tel["metrics"][metric]) if v is not None]
    trend = trend_analyzer(
        vals, hours,
        threshold=rul_result.get("threshold_zone_c"),
        label=f"{bearing_id} {metric}",
    ) if len(vals) >= 3 else {}

    # Anomaly detection on kurtosis (most sensitive for bearing faults)
    kurt  = [v for v in tel["metrics"].get("kurtosis", []) if v is not None]
    khrs  = [h for h, v in zip(tel["hours_elapsed"], tel["metrics"].get("kurtosis", [])) if v is not None]
    anom  = anomaly_detector(kurt, khrs, method="zscore", label=f"{bearing_id} kurtosis") if len(kurt) >= 10 else {}

    # Current values — read directly from telemetry (last valid point)
    rms_list  = [v for v in tel["metrics"].get("rms", [])      if v is not None]
    kurt_list = [v for v in tel["metrics"].get("kurtosis", []) if v is not None]
    curr_rms  = round(float(rms_list[-1]),  6) if rms_list  else None
    curr_kurt = round(float(kurt_list[-1]), 6) if kurt_list else None

    # ISO zone from bearing snapshot if available, otherwise compute
    brg_snap  = snap.get("bearings", {}).get(bearing_id, {})
    curr_zone = brg_snap.get("iso_zone", "Unknown")
    if curr_zone == "Unknown" and curr_rms is not None:
        curr_zone = _iso_severity(curr_rms).get("zone", "Unknown")

    # ISO zone details
    iso_detail = _iso_severity(curr_rms) if curr_rms is not None else {}

    # Events for this bearing
    brg_events = [e for e in events.get("events", []) if e.get("bearing_id") == bearing_id]

    rec = _bearing_recommendation(
        bearing_id, test_id,
        curr_rms, curr_zone,
        rul_result.get("rul_hours"),
        rul_result.get("rul_status"),
        anom.get("n_anomalies", 0),
    )

    return {
        "test_id":              test_id,
        "bearing_id":           bearing_id,
        "metric_analyzed":      metric,
        "n_data_points":        len(vals),
        "current_rms":          curr_rms,
        "current_kurtosis":     curr_kurt,
        "current_iso_zone":     curr_zone,
        "iso_zone_description": iso_detail.get("zone_description", ""),
        "iso_action":           iso_detail.get("action_required", ""),
        "rul_hours":            rul_result.get("rul_hours"),
        "rul_status":           rul_result.get("rul_status"),
        "fit_quality":          rul_result.get("fit_quality"),
        "r_squared":            rul_result.get("r_squared"),
        "trend": {
            "slope_per_hr":          trend.get("slope"),
            "direction":             trend.get("trend_direction"),
            "projected_breach_hour": trend.get("projected_breach_hour"),
        },
        "n_anomalies":          anom.get("n_anomalies", 0),
        "anomaly_rate_pct":     anom.get("anomaly_rate_pct", 0),
        "first_anomaly_hour":   anom.get("first_anomaly_hour"),
        "last_anomaly_hour":    anom.get("last_anomaly_hour"),
        "n_critical_events":    len(brg_events),
        "critical_events":      brg_events[:5],   # top 5 only to keep output manageable
        "recommendation":       rec,
    }


def search_knowledge(
    query: str,
    source: Optional[str] = None,
) -> dict:
    """
    Search the industrial engineering knowledge base.

    Performs semantic search over:
      - TEMA Standards: fouling limits, U-values, cleaning methods, economics
      - ISO 10816: vibration zone definitions, fault frequencies, actions
      - Bearing Failure Catalog: failure modes, IMS dataset specifics, diagnosis
      - Maintenance SOPs: step-by-step cleaning and replacement procedures

    Args:
        query:  Natural language question or keyword phrase. Examples:
                  "TEMA fouling resistance limit crude oil"
                  "ISO Zone D vibration action to take"
                  "steps to replace a bearing"
                  "outer race spalling diagnosis"
                  "asphaltene chemical cleaning procedure"
        source: Optional filter. Options:
                  'TEMA_standards'
                  'ISO_10816_vibration_standard'
                  'bearing_failure_catalog'
                  'maintenance_sop'

    Returns:
        dict with: top_answer (most relevant passage), results (list of passages
        with source, section, relevance_score).
    """
    return _query_kb(query, n_results=4, source_filter=source)


# ---------------------------------------------------------------------------
# Recommendation helpers
# ---------------------------------------------------------------------------

def _fouling_recommendation(eid, rf_final, breached, breach_hour, total_loss, trend) -> str:
    if rf_final is None:
        return f"{eid}: Insufficient data for fouling assessment."
    rf_pct = (rf_final / 0.0005) * 100
    parts  = []
    if breached:
        parts.append(
            f"⚠️  {eid} has BREACHED the TEMA fouling limit "
            f"(Rf={rf_final:.5f} m²·K/W = {rf_pct:.0f}% of limit). "
            f"Cleaning required within 4 weeks."
        )
    else:
        parts.append(
            f"{eid} is below TEMA limit "
            f"(Rf={rf_final:.5f} m²·K/W = {rf_pct:.0f}% of limit). "
            f"No immediate cleaning required."
        )
    if total_loss > 0:
        parts.append(f"Total energy loss over campaign: ${total_loss:,.0f}.")
    if trend:
        proj = trend.get("projected_breach_hour")
        if proj:
            parts.append(f"Trend projects TEMA breach at hour {proj:.0f}.")
    return " ".join(parts)


def _bearing_recommendation(bid, tid, rms, zone, rul, status, n_anom) -> str:
    parts = []
    if zone == "D":
        parts.append(f"⚠️  CRITICAL: {bid} (Test {tid}) is in ISO Zone D (RMS={rms:.3f} mm/s). STOP MACHINE NOW.")
    elif zone == "C":
        parts.append(f"⚠️  WARNING: {bid} (Test {tid}) in ISO Zone C. Plan maintenance within 2–4 weeks.")
    else:
        parts.append(f"{bid} (Test {tid}) in ISO Zone {zone} — current RMS={rms:.3f} mm/s.")
    if rul is not None:
        parts.append(f"Estimated RUL: {rul:.1f} hours.")
    if n_anom > 0:
        parts.append(f"{n_anom} vibration anomalies detected — inspect bearing seals and lubrication.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Agent Session
# ---------------------------------------------------------------------------

# All 7 tools the LLM can call
AGENT_TOOLS = [
    list_equipment,
    get_equipment_snapshot,
    get_hx_data,
    get_bearing_data,
    analyze_fouling,
    analyze_bearing,
    search_knowledge,
]


class IndustrialAgent:
    """
    Gemini-powered industrial plant monitoring agent.

    Uses the google.genai SDK (new) with automatic function calling enabled.
    Gemini decides which tools to use; the SDK executes them and feeds
    results back into the conversation automatically.

    Usage:
        agent = IndustrialAgent()
        response = agent.ask("How is E02 doing?")
        print(response)
    """

    def __init__(self, model_name: str = None, verbose: bool = False):
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key or api_key == "your_google_api_key_here":
            raise ValueError(
                "GOOGLE_API_KEY not set. "
                "Add it to your .env file: GOOGLE_API_KEY=your_key_here\n"
                "Get a key at: https://aistudio.google.com/app/apikey"
            )

        self.model_name = model_name or os.environ.get("LLM_MODEL", "gemini-2.0-flash")
        self.verbose    = verbose
        self._history   = []   # list of dicts: {"role": ..., "parts": [...]}

        # Initialize client
        self.client = genai.Client(api_key=api_key)

        # Automatic function calling config
        self._afc_config = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=AGENT_TOOLS,
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                disable=False,
            ),
        )

        if self.verbose:
            logger.info(f"Agent initialized: {self.model_name}, {len(AGENT_TOOLS)} tools registered")

    def ask(self, message: str) -> str:
        """
        Send a message to the agent and return the text response.

        The SDK automatically:
          1. Sends message to Gemini
          2. Detects tool calls in the response
          3. Executes the Python functions
          4. Sends results back to Gemini
          5. Returns the final synthesized answer

        Args:
            message: Natural language question or command.

        Returns:
            Agent's text response as a string.
        """
        t0 = time.perf_counter()
        # Append user turn
        self._history.append({"role": "user", "parts": [{"text": message}]})

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=self._history,
                config=self._afc_config,
            )
            elapsed = time.perf_counter() - t0
            text    = response.text
            # Append assistant turn
            self._history.append({"role": "model", "parts": [{"text": text}]})
            if self.verbose:
                logger.info(f"Response in {elapsed:.2f}s")
            return text
        except Exception as e:
            self._history.pop()   # remove failed user turn
            return f"Agent error: {e}"

    def reset(self):
        """Start a fresh conversation (clears history)."""
        self._history = []

    @property
    def history(self) -> list:
        """Return the raw conversation history."""
        return self._history
