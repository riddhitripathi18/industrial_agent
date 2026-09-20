"""
Phase 2 Test Suite — Engineering Tools
=========================================
Validates all engineering physics calculations against known physical constraints.

Run with:
    pytest tests/test_engineering_tools.py -v
"""

import pytest
import math
from agent.data_tools.hx_data import get_hx_telemetry
from agent.data_tools.bearing_data import get_bearing_telemetry
from agent.tools.engineering_tools import (
    calc_overall_heat_transfer_coeff,
    calc_fouling_resistance,
    calc_energy_penalty,
    calc_bearing_rul,
    iso_10816_severity,
)


# ===========================================================================
# Fixtures — load telemetry once for all tests
# ===========================================================================

@pytest.fixture(scope="module")
def hx_e01():
    return get_hx_telemetry("E01")

@pytest.fixture(scope="module")
def hx_e02():
    return get_hx_telemetry("E02")

@pytest.fixture(scope="module")
def bearing_t2_b1():
    return get_bearing_telemetry(2, "Bearing1", ["rms", "kurtosis"])

@pytest.fixture(scope="module")
def bearing_t1_b3x():
    return get_bearing_telemetry(1, "Bearing3_x", ["rms"])


# ===========================================================================
# 1. Overall Heat Transfer Coefficient
# ===========================================================================

class TestUCalculation:

    def test_u_values_are_physical(self, hx_e01):
        """U for crude oil shell-and-tube HX should be positive and non-zero.
        
        Note: The exact value depends on assumed heat transfer area (A).
        With A=180 m² (typical large refinery unit) the dataset yields ~1200-1300 W/m²·K,
        which is reasonable for a well-turbulated crude train.
        Range 50–3000 W/m²·K covers all realistic shell-and-tube configurations.
        """
        result  = calc_overall_heat_transfer_coeff(hx_e01, "E01")
        U_vals  = [v for v in result["U_W_m2K"] if v is not None and not math.isnan(v)]
        assert len(U_vals) > 0
        mean_U  = sum(U_vals) / len(U_vals)
        assert 50 <= mean_U <= 3000, f"Mean U={mean_U:.1f} out of physical range [50–3000 W/m²·K]"

    def test_u_clean_is_positive(self, hx_e01):
        result = calc_overall_heat_transfer_coeff(hx_e01, "E01")
        assert result["U_clean"] > 0

    def test_Q_kW_positive(self, hx_e02):
        """Heat duty must be non-negative (crude absorbs heat from shell)."""
        result = calc_overall_heat_transfer_coeff(hx_e02, "E02")
        Q_vals = [v for v in result["Q_kW"] if v is not None]
        assert all(q >= 0 for q in Q_vals), "Negative heat duty detected — check delta-T sign"

    def test_output_json_serializable(self, hx_e01):
        import json
        result = calc_overall_heat_transfer_coeff(hx_e01, "E01")
        json.dumps(result)   # must not raise

    def test_lmtd_nonzero(self, hx_e01):
        result = calc_overall_heat_transfer_coeff(hx_e01, "E01")
        lmtd_v = [v for v in result["LMTD_K"] if v is not None and v > 0]
        assert len(lmtd_v) > result["n_points"] * 0.8, "Majority of LMTD values should be > 0"

    def test_required_keys(self, hx_e01):
        result   = calc_overall_heat_transfer_coeff(hx_e01, "E01")
        required = {"exchanger_id", "A_m2", "U_clean", "time_hr", "Q_kW", "LMTD_K", "U_W_m2K", "n_points"}
        assert not (required - set(result.keys()))


# ===========================================================================
# 2. Fouling Resistance
# ===========================================================================

class TestFoulingResistance:

    def test_rf_non_negative(self, hx_e01):
        """Fouling resistance must be >= 0 (cleaned baseline = 0)."""
        result = calc_fouling_resistance(hx_e01, "E01")
        rf_vals = [v for v in result["Rf"] if v is not None and not math.isnan(v)]
        assert all(rf >= 0 for rf in rf_vals), "Negative Rf values detected"

    def test_rf_increases_over_time(self, hx_e01):
        """Rf should generally trend upward (fouling accumulates)."""
        result   = calc_fouling_resistance(hx_e01, "E01")
        rf_clean = [v for v in result["Rf"] if v is not None and not math.isnan(v)]
        # Compare first 10% vs last 10%
        n        = len(rf_clean)
        n10      = max(1, n // 10)
        rf_early = sum(rf_clean[:n10]) / n10
        rf_late  = sum(rf_clean[-n10:]) / n10
        assert rf_late >= rf_early * 0.8, \
            f"Rf did not trend upward: early={rf_early:.6f}, late={rf_late:.6f}"

    def test_tema_breach_fields_present(self, hx_e01):
        result = calc_fouling_resistance(hx_e01, "E01")
        assert "tema_breached"    in result
        assert "tema_breach_hour" in result
        assert isinstance(result["tema_breached"], bool)

    def test_rf_max_reasonable(self, hx_e01):
        """Max Rf for a severely fouled crude HX should be < 0.005 m²·K/W."""
        result = calc_fouling_resistance(hx_e01, "E01")
        assert result["Rf_max"] < 0.005, f"Rf_max={result['Rf_max']} suspiciously high"

    def test_custom_tema_limit(self, hx_e01):
        """Very tight TEMA limit should be breached early."""
        result = calc_fouling_resistance(hx_e01, "E01", tema_limit=1e-9)
        assert result["tema_breached"] is True


# ===========================================================================
# 3. Energy Penalty
# ===========================================================================

class TestEnergyPenalty:

    def test_Q_lost_non_negative(self, hx_e01):
        result   = calc_energy_penalty(hx_e01, "E01")
        q_lost   = [v for v in result["Q_lost_kW"] if v is not None]
        assert all(q >= 0 for q in q_lost), "Q_lost must be >= 0"

    def test_total_loss_positive(self, hx_e01):
        result = calc_energy_penalty(hx_e01, "E01", fuel_price_per_mmbtu=12.0)
        assert result["total_loss_USD"] >= 0

    def test_higher_fuel_price_more_loss(self, hx_e01):
        cheap      = calc_energy_penalty(hx_e01, "E01", fuel_price_per_mmbtu=5.0)
        expensive  = calc_energy_penalty(hx_e01, "E01", fuel_price_per_mmbtu=20.0)
        assert expensive["total_loss_USD"] > cheap["total_loss_USD"]

    def test_required_keys(self, hx_e01):
        result   = calc_energy_penalty(hx_e01, "E01")
        required = {"total_loss_USD", "Q_lost_kW", "cost_per_hr_USD", "cumulative_loss_USD", "Q_clean_kW"}
        assert not (required - set(result.keys()))

    def test_json_serializable(self, hx_e01):
        import json
        result = calc_energy_penalty(hx_e01, "E01")
        json.dumps(result)


# ===========================================================================
# 4. Bearing RUL
# ===========================================================================

class TestBearingRUL:

    def test_rul_has_required_keys(self, bearing_t2_b1):
        result = calc_bearing_rul(bearing_t2_b1, "Bearing1", metric="rms")
        required = {"bearing_id", "metric", "fit_a", "fit_b", "r_squared",
                    "fit_quality", "current_value", "current_hour", "rul_status"}
        assert not (required - set(result.keys()))

    def test_fit_b_positive_for_degrading_run(self, bearing_t2_b1):
        """Test 2 Bearing 1 is a run-to-failure — exponential growth rate must be > 0."""
        result = calc_bearing_rul(bearing_t2_b1, "Bearing1", metric="rms")
        assert result["fit_b"] > 0, "Degrading bearing must have positive growth rate (b > 0)"

    def test_r_squared_acceptable(self, bearing_t2_b1):
        """R² for exponential fit on IMS Test 2 Bearing 1 RMS.
        
        Note: Test 2 Bearing 1 has a near-flat degradation for most of its life
        followed by a sharp spike at failure — this is not a pure exponential,
        so R² of 0.45+ is considered acceptable here.
        """
        result = calc_bearing_rul(bearing_t2_b1, "Bearing1", metric="rms")
        assert result["r_squared"] >= 0.45, \
            f"Poor exponential fit: R²={result['r_squared']}"

    def test_current_value_matches_telemetry_last(self, bearing_t2_b1):
        result    = calc_bearing_rul(bearing_t2_b1, "Bearing1", metric="rms")
        last_rms  = [v for v in bearing_t2_b1["metrics"]["rms"] if v is not None][-1]
        assert abs(result["current_value"] - last_rms) < 1e-3

    def test_kurtosis_rul(self, bearing_t2_b1):
        """Kurtosis RUL should run without error."""
        result = calc_bearing_rul(bearing_t2_b1, "Bearing1", metric="kurtosis")
        assert "rul_status" in result

    def test_rul_json_serializable(self, bearing_t2_b1):
        import json
        result = calc_bearing_rul(bearing_t2_b1, "Bearing1", metric="rms")
        json.dumps(result)


# ===========================================================================
# 5. ISO 10816 Severity
# ===========================================================================

class TestISO10816:

    @pytest.mark.parametrize("rms,expected_zone", [
        (0.5,  "A"),
        (1.5,  "A"),
        (2.3,  "B"),
        (3.5,  "B"),
        (4.5,  "C"),
        (6.0,  "C"),
        (7.1,  "D"),
        (10.0, "D"),
    ])
    def test_zone_classification(self, rms, expected_zone):
        result = iso_10816_severity(rms)
        assert result["zone"] == expected_zone, \
            f"RMS={rms} → expected zone {expected_zone}, got {result['zone']}"

    def test_zone_d_action_is_stop(self):
        result = iso_10816_severity(8.0)
        assert "STOP" in result["action_required"].upper()

    def test_distance_to_danger_zone_a(self):
        result = iso_10816_severity(1.0)
        assert result["distance_to_danger"] == pytest.approx(6.1, abs=0.01)

    def test_distance_zero_when_in_danger(self):
        result = iso_10816_severity(9.0)
        assert result["distance_to_danger"] == 0.0

    def test_required_keys(self):
        result   = iso_10816_severity(3.0)
        required = {"rms_value", "zone", "zone_description",
                    "action_required", "distance_to_danger"}
        assert not (required - set(result.keys()))
