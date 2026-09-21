"""
Phase 4 Test Suite — Agent Core (No API Key Required)
=======================================================
Tests the composite agent tools and agent wiring WITHOUT calling
the Gemini API. The tools themselves call Phase 1-2 data functions
and engineering calculations on real data.

These tests verify:
  - All 7 composite tools run without error on real data
  - Outputs are JSON-serializable (LLM-tool-ready)
  - Recommendation strings are generated correctly
  - analyze_fouling correctly chains data fetch + engineering compute
  - analyze_bearing correctly chains data fetch + RUL + trend + anomaly

Run with:
    pytest tests/test_agent_core.py -v
"""

import pytest
import json
import math

# Import the composite tools directly (no agent session needed)
from agent.core import (
    list_equipment,
    get_equipment_snapshot,
    get_hx_data,
    get_bearing_data,
    analyze_fouling,
    analyze_bearing,
    search_knowledge,
    AGENT_TOOLS,
)


# ===========================================================================
# 1. Tool Registry
# ===========================================================================

class TestToolRegistry:

    def test_seven_tools_registered(self):
        assert len(AGENT_TOOLS) == 7

    def test_all_tools_are_callable(self):
        for fn in AGENT_TOOLS:
            assert callable(fn)

    def test_all_tools_have_docstrings(self):
        """Gemini uses docstrings to understand when to call each tool."""
        for fn in AGENT_TOOLS:
            assert fn.__doc__ and len(fn.__doc__.strip()) > 20, \
                f"{fn.__name__} has missing or too-short docstring"

    def test_tool_names_as_expected(self):
        names = {fn.__name__ for fn in AGENT_TOOLS}
        expected = {
            "list_equipment", "get_equipment_snapshot",
            "get_hx_data", "get_bearing_data",
            "analyze_fouling", "analyze_bearing", "search_knowledge",
        }
        assert names == expected


# ===========================================================================
# 2. list_equipment
# ===========================================================================

class TestListEquipment:

    @pytest.fixture(scope="class")
    def result(self):
        return list_equipment()

    def test_has_heat_exchangers_key(self, result):
        assert "heat_exchangers" in result

    def test_has_bearing_tests_key(self, result):
        assert "bearing_tests" in result

    def test_five_heat_exchangers(self, result):
        assert len(result["heat_exchangers"]) == 5

    def test_three_bearing_tests(self, result):
        assert len(result["bearing_tests"]) == 3

    def test_json_serializable(self, result):
        json.dumps(result)


# ===========================================================================
# 3. get_equipment_snapshot
# ===========================================================================

@pytest.fixture(scope="module")
def snapshot_result():
    return get_equipment_snapshot()


class TestGetEquipmentSnapshot:

    def test_has_heat_exchangers_key(self, snapshot_result):
        assert "heat_exchangers" in snapshot_result

    def test_has_bearing_tests_key(self, snapshot_result):
        assert "bearing_tests" in snapshot_result

    def test_has_all_three_bearing_tests(self, snapshot_result):
        assert "test_1" in snapshot_result["bearing_tests"]
        assert "test_2" in snapshot_result["bearing_tests"]
        assert "test_3" in snapshot_result["bearing_tests"]

    def test_json_serializable(self, snapshot_result):
        json.dumps(snapshot_result)


# ===========================================================================
# 4. get_hx_data
# ===========================================================================

@pytest.fixture(scope="module")
def hx_data_result():
    return get_hx_data("E01", sample_step=20)


class TestGetHXData:

    def test_exchanger_id_in_result(self, hx_data_result):
        assert hx_data_result.get("exchanger_id") == "E01"

    def test_has_temperature_keys(self, hx_data_result):
        for key in ("tube_temp_in", "tube_temp_out", "shell_temp_in", "shell_temp_out"):
            assert key in hx_data_result, f"Missing key: {key}"

    def test_downsampling_applied(self, hx_data_result):
        """sample_step=20 should give fewer rows than sample_step=1."""
        full = get_hx_data("E01", sample_step=1)
        assert hx_data_result["n_points"] < full["n_points"]

    def test_case_insensitive_id(self):
        """Lowercase exchanger ID should be accepted."""
        result = get_hx_data("e02")
        assert result["exchanger_id"] == "E02"

    def test_json_serializable(self, hx_data_result):
        json.dumps(hx_data_result)


# ===========================================================================
# 5. get_bearing_data
# ===========================================================================

@pytest.fixture(scope="module")
def bearing_data_result():
    return get_bearing_data(2, "Bearing1")


class TestGetBearingData:

    def test_test_id_correct(self, bearing_data_result):
        assert bearing_data_result["test_id"] == 2

    def test_bearing_id_correct(self, bearing_data_result):
        assert bearing_data_result["bearing_id"] == "Bearing1"

    def test_default_metrics_present(self, bearing_data_result):
        """Default metrics = ['rms', 'kurtosis']."""
        assert "rms"      in bearing_data_result["metrics"]
        assert "kurtosis" in bearing_data_result["metrics"]

    def test_n_points_test2(self, bearing_data_result):
        """Test 2 has 984 snapshots."""
        assert bearing_data_result["n_points"] == 984

    def test_dual_axis_test1(self):
        result = get_bearing_data(1, "Bearing3_x")
        assert result["bearing_id"] == "Bearing3_x"
        assert result["n_points"] > 0

    def test_json_serializable(self, bearing_data_result):
        json.dumps(bearing_data_result)


# ===========================================================================
# 6. analyze_fouling (composite — chains data + engineering)
# ===========================================================================

@pytest.fixture(scope="module")
def fouling_result():
    return analyze_fouling("E01", fuel_price_per_mmbtu=12.0)


class TestAnalyzeFouling:

    def test_exchanger_id_correct(self, fouling_result):
        assert fouling_result["exchanger_id"] == "E01"

    def test_required_keys_present(self, fouling_result):
        keys = {
            "exchanger_id", "U_clean_W_m2K", "U_final_W_m2K",
            "Rf_final_m2KW", "Rf_max_m2KW", "tema_breached",
            "total_energy_loss_USD", "trend", "recommendation",
        }
        assert not (keys - set(fouling_result.keys()))

    def test_u_clean_positive(self, fouling_result):
        assert fouling_result["U_clean_W_m2K"] > 0

    def test_rf_final_non_negative(self, fouling_result):
        assert fouling_result["Rf_final_m2KW"] is None or fouling_result["Rf_final_m2KW"] >= 0

    def test_energy_loss_non_negative(self, fouling_result):
        assert fouling_result["total_energy_loss_USD"] >= 0

    def test_tema_breached_is_bool(self, fouling_result):
        assert isinstance(fouling_result["tema_breached"], bool)

    def test_recommendation_is_string(self, fouling_result):
        assert isinstance(fouling_result["recommendation"], str)
        assert len(fouling_result["recommendation"]) > 10

    def test_higher_fuel_price_more_cost(self):
        cheap     = analyze_fouling("E02", fuel_price_per_mmbtu=5.0)
        expensive = analyze_fouling("E02", fuel_price_per_mmbtu=25.0)
        assert expensive["total_energy_loss_USD"] > cheap["total_energy_loss_USD"]

    def test_json_serializable(self, fouling_result):
        json.dumps(fouling_result)

    def test_trend_has_direction(self, fouling_result):
        assert fouling_result["trend"]["direction"] is not None


# ===========================================================================
# 7. analyze_bearing (composite — chains data + RUL + trend + anomaly)
# ===========================================================================

@pytest.fixture(scope="module")
def bearing_analysis_result():
    return analyze_bearing(2, "Bearing1", metric="rms")


class TestAnalyzeBearing:

    def test_required_keys_present(self, bearing_analysis_result):
        keys = {
            "test_id", "bearing_id", "current_rms", "current_kurtosis",
            "current_iso_zone", "rul_hours", "rul_status", "trend",
            "n_anomalies", "recommendation",
        }
        assert not (keys - set(bearing_analysis_result.keys()))

    def test_test_id_correct(self, bearing_analysis_result):
        assert bearing_analysis_result["test_id"] == 2

    def test_bearing_id_correct(self, bearing_analysis_result):
        assert bearing_analysis_result["bearing_id"] == "Bearing1"

    def test_current_rms_positive(self, bearing_analysis_result):
        assert bearing_analysis_result["current_rms"] > 0

    def test_iso_zone_valid(self, bearing_analysis_result):
        assert bearing_analysis_result["current_iso_zone"] in {"A", "B", "C", "D"}

    def test_trend_direction_set(self, bearing_analysis_result):
        assert bearing_analysis_result["trend"]["direction"] is not None

    def test_recommendation_string(self, bearing_analysis_result):
        assert isinstance(bearing_analysis_result["recommendation"], str)
        assert len(bearing_analysis_result["recommendation"]) > 10

    def test_n_anomalies_non_negative(self, bearing_analysis_result):
        assert bearing_analysis_result["n_anomalies"] >= 0

    def test_json_serializable(self, bearing_analysis_result):
        json.dumps(bearing_analysis_result)

    def test_test3_bearing3_analyzable(self):
        """Longest dataset (6,324 rows) should also work."""
        result = analyze_bearing(3, "Bearing3", metric="rms")
        assert "rul_status" in result

    def test_test1_dual_axis_analyzable(self):
        """Dual-axis Test 1 bearings should work."""
        result = analyze_bearing(1, "Bearing3_x", metric="rms")
        assert result["bearing_id"] == "Bearing3_x"


# ===========================================================================
# 8. search_knowledge (calls Phase 3 RAG)
# ===========================================================================

class TestSearchKnowledge:

    def test_fouling_query(self):
        result = search_knowledge("TEMA fouling resistance limit crude oil")
        assert result["n_results"] > 0
        assert isinstance(result["top_answer"], str)

    def test_vibration_query(self):
        result = search_knowledge("ISO 10816 Zone D action required")
        assert result["n_results"] > 0

    def test_source_filter_respected(self):
        result = search_knowledge("fouling resistance", source="TEMA_standards")
        for r in result["results"]:
            assert r["source"] == "TEMA_standards"

    def test_json_serializable(self):
        result = search_knowledge("bearing outer race failure")
        json.dumps(result)
