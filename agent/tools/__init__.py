"""
Engineering & Analysis Tools
==============================
Phase 2 tools: deterministic physics calculations and signal diagnostics.

Engineering tools (first-principles):
    calc_overall_heat_transfer_coeff()
    calc_fouling_resistance()
    calc_energy_penalty()
    calc_bearing_rul()
    iso_10816_severity()

Analysis tools (signal & statistical):
    trend_analyzer()
    anomaly_detector()
    correlation_matrix()
    compute_fft_spectrum()
"""
from agent.tools.engineering_tools import (
    calc_overall_heat_transfer_coeff,
    calc_fouling_resistance,
    calc_energy_penalty,
    calc_bearing_rul,
    iso_10816_severity,
)
from agent.tools.analysis_tools import (
    trend_analyzer,
    anomaly_detector,
    correlation_matrix,
    compute_fft_spectrum,
)

__all__ = [
    "calc_overall_heat_transfer_coeff",
    "calc_fouling_resistance",
    "calc_energy_penalty",
    "calc_bearing_rul",
    "iso_10816_severity",
    "trend_analyzer",
    "anomaly_detector",
    "correlation_matrix",
    "compute_fft_spectrum",
]
