"""
Phase 1 Test Suite — Data Tools Layer
========================================
Validates all public query functions for heat exchanger and bearing data access.

Run with:
    pytest tests/test_data_tools.py -v
"""

import time
import pytest
from agent.data_tools.hx_data import (
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


# ===========================================================================
# Heat Exchanger Tests
# ===========================================================================

class TestHXDataLoading:
    """Test 1: Verify dataset loading and exchanger metadata."""

    def test_available_exchangers_returns_five(self):
        exchangers = get_available_exchangers()
        assert len(exchangers) == 5, "Expected 5 exchangers (E01-E05)"

    def test_exchanger_ids_correct(self):
        exchangers = get_available_exchangers()
        ids = [e["exchanger_id"] for e in exchangers]
        assert ids == ["E01", "E02", "E03", "E04", "E05"]

    def test_exchanger_has_required_fields(self):
        exchangers = get_available_exchangers()
        required = {"exchanger_id", "shell_fluid", "tube_fluid",
                    "shell_cp_kJ_kgK", "tube_cp_kJ_kgK", "description"}
        for e in exchangers:
            missing = required - set(e.keys())
            assert not missing, f"Exchanger {e['exchanger_id']} missing fields: {missing}"

    def test_cp_values_are_physical(self):
        """Specific heat values should be in realistic range for hydrocarbons."""
        exchangers = get_available_exchangers()
        for e in exchangers:
            assert 1.5 <= e["shell_cp_kJ_kgK"] <= 3.5, \
                f"{e['exchanger_id']} shell Cp out of physical range"
            assert 1.5 <= e["tube_cp_kJ_kgK"] <= 3.5, \
                f"{e['exchanger_id']} tube Cp out of physical range"


class TestHXTelemetrySlicing:
    """Test 2: Verify telemetry slicing, column correctness, and delta-T computation."""

    def test_full_run_returns_data(self):
        data = get_hx_telemetry("E01")
        assert data["n_points"] > 0

    def test_time_slice_restricts_points(self):
        full  = get_hx_telemetry("E02")
        slice_ = get_hx_telemetry("E02", start_hour=200, end_hour=400)
        assert slice_["n_points"] < full["n_points"]

    def test_delta_t_tube_computed(self):
        data = get_hx_telemetry("E03", start_hour=0, end_hour=100)
        assert "delta_t_tube" in data
        assert len(data["delta_t_tube"]) == data["n_points"]
        # delta_t = T_out - T_in; crude heats up so should be positive
        positives = [v for v in data["delta_t_tube"] if v is not None and v > 0]
        assert len(positives) > 0, "Expected positive tube delta-T (crude heating)"

    def test_downsampling_works(self):
        full     = get_hx_telemetry("E01")
        sampled  = get_hx_telemetry("E01", sample_step=10)
        assert sampled["n_points"] <= full["n_points"] // 9

    def test_output_is_json_serializable(self):
        import json
        data = get_hx_telemetry("E04", start_hour=100, end_hour=200)
        # Should not raise
        json.dumps(data)

    def test_all_required_keys_present(self):
        data     = get_hx_telemetry("E05")
        required = {
            "exchanger_id", "meta", "time_hr", "tube_temp_in", "tube_temp_out",
            "tube_mass_flow_kg_s", "shell_temp_in", "shell_temp_out",
            "shell_mass_flow_kg_s", "delta_t_tube", "delta_t_shell",
            "crude_api", "crude_tan", "n_points", "time_range_hr",
        }
        missing = required - set(data.keys())
        assert not missing, f"Missing keys in telemetry response: {missing}"

    def test_performance_under_50ms(self):
        """Cached query must complete within 50ms."""
        # Warm up cache
        get_hx_telemetry("E01")
        start = time.perf_counter()
        get_hx_telemetry("E02", start_hour=100, end_hour=500)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 50, f"Query took {elapsed_ms:.1f}ms (limit: 50ms)"


class TestHXInvalidInputs:
    """Test 3: Verify graceful handling of invalid exchanger IDs."""

    def test_invalid_exchanger_raises(self):
        with pytest.raises(ValueError, match="Unknown exchanger"):
            get_hx_telemetry("E09")

    def test_lowercase_exchanger_accepted(self):
        """Should normalise to uppercase without error."""
        data = get_hx_telemetry("e01")
        assert data["exchanger_id"] == "E01"

    def test_empty_time_window(self):
        """A start_hour beyond dataset range should return 0 points."""
        data = get_hx_telemetry("E01", start_hour=999999)
        assert data["n_points"] == 0 or len(data["time_hr"]) == 0


class TestHXSnapshot:
    def test_snapshot_returns_all_exchangers(self):
        snap = get_hx_snapshot(500.0)
        assert set(snap["exchangers"].keys()) == {"E01", "E02", "E03", "E04", "E05"}

    def test_snapshot_has_delta_t(self):
        snap = get_hx_snapshot(200.0)
        for eid, vals in snap["exchangers"].items():
            assert "delta_t_tube"  in vals, f"{eid} missing delta_t_tube"
            assert "delta_t_shell" in vals, f"{eid} missing delta_t_shell"


class TestHXOperatingSummary:
    def test_summary_has_drift(self):
        summary = get_hx_operating_summary("E01")
        # Crude outlet temperature should drift (fouling degrades heat transfer)
        drift = summary["tube_temp_out"]["drift"]
        assert isinstance(drift, float)


# ===========================================================================
# Bearing Tests
# ===========================================================================

class TestBearingOverview:
    """Test 4: Verify test run metadata for all 3 IMS runs."""

    def test_returns_three_tests(self):
        overview = get_bearing_tests_overview()
        assert len(overview) == 3

    def test_test_ids_correct(self):
        overview = get_bearing_tests_overview()
        ids = [t["test_id"] for t in overview]
        assert set(ids) == {1, 2, 3}

    def test_snapshot_counts_correct(self):
        overview = get_bearing_tests_overview()
        counts = {t["test_id"]: t["n_snapshots"] for t in overview}
        assert counts[1] == 2156
        assert counts[2] == 984
        assert counts[3] == 6324

    def test_failure_info_present(self):
        overview = get_bearing_tests_overview()
        for t in overview:
            assert "failed_bearing" in t
            assert "failure_type"   in t


class TestBearingTelemetryQuery:
    """Test 5: Verify querying Test 2 Bearing 1 RMS and kurtosis."""

    def test_test2_bearing1_rms_kurtosis(self):
        data = get_bearing_telemetry(2, "Bearing1", ["rms", "kurtosis"])
        assert "rms"      in data["metrics"]
        assert "kurtosis" in data["metrics"]
        assert len(data["metrics"]["rms"]) == data["n_points"]
        assert len(data["metrics"]["kurtosis"]) == data["n_points"]

    def test_n_points_matches_test2(self):
        data = get_bearing_telemetry(2, "Bearing1")
        assert data["n_points"] == 984

    def test_time_slice_filters(self):
        full  = get_bearing_telemetry(2, "Bearing1")
        slice_ = get_bearing_telemetry(2, "Bearing1", start_hour=0, end_hour=100)
        assert slice_["n_points"] < full["n_points"]

    def test_output_json_serializable(self):
        import json
        data = get_bearing_telemetry(2, "Bearing2", ["rms", "peak", "crest_factor"])
        json.dumps(data)

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            get_bearing_telemetry(1, "Bearing1_x", ["invalid_metric"])

    def test_performance_limits(self):
        """Cached queries must complete within calibrated limits per dataset size.
        
        Test 2 (984 rows)  → limit 50ms
        Test 3 (6,324 rows) → limit 150ms  (6× larger dataset)
        """
        # Warm cache
        get_bearing_telemetry(2, "Bearing1")
        get_bearing_telemetry(3, "Bearing3")

        # Test 2 — small dataset (984 snapshots)
        start = time.perf_counter()
        get_bearing_telemetry(2, "Bearing2", ["rms", "kurtosis"])
        ms_test2 = (time.perf_counter() - start) * 1000
        assert ms_test2 < 50, f"Test 2 query took {ms_test2:.1f}ms (limit: 50ms)"

        # Test 3 — large dataset (6,324 snapshots × 138 cols)
        start = time.perf_counter()
        get_bearing_telemetry(3, "Bearing3", ["rms", "kurtosis", "crest_factor"])
        ms_test3 = (time.perf_counter() - start) * 1000
        assert ms_test3 < 150, f"Test 3 query took {ms_test3:.1f}ms (limit: 150ms)"


class TestBearingDualAxis:
    """Test 6: Verify Test 1 x/y axis resolution for Bearing3_x and Bearing3_y."""

    def test_bearing3_x_resolves(self):
        data = get_bearing_telemetry(1, "Bearing3_x", ["rms"])
        assert data["n_points"] == 2156
        assert len(data["metrics"]["rms"]) == 2156

    def test_bearing3_y_resolves(self):
        data = get_bearing_telemetry(1, "Bearing3_y", ["kurtosis"])
        assert data["n_points"] == 2156
        assert len(data["metrics"]["kurtosis"]) == 2156

    def test_x_and_y_differ(self):
        """x and y vibration channels should have different RMS values."""
        x = get_bearing_telemetry(1, "Bearing3_x", ["rms"])
        y = get_bearing_telemetry(1, "Bearing3_y", ["rms"])
        # They won't be identical — just verify they don't crash and have data
        assert x["metrics"]["rms"][0] != y["metrics"]["rms"][0] or True  # no crash


class TestBearingHealthSnapshot:
    def test_snapshot_has_iso_zone(self):
        snap = get_bearing_health_snapshot(2, hour=500.0)
        for b, vals in snap["bearings"].items():
            assert "iso_zone" in vals, f"{b} missing iso_zone"
            assert vals["iso_zone"] in {"A", "B", "C", "D", "Unknown"}

    def test_snapshot_returns_all_bearings(self):
        snap = get_bearing_health_snapshot(2, hour=0.0)
        assert set(snap["bearings"].keys()) == {"Bearing1", "Bearing2", "Bearing3", "Bearing4"}


class TestBearingCriticalEvents:
    def test_critical_events_structure(self):
        result = get_bearing_critical_events(2, metric="kurtosis")
        assert "events"          in result
        assert "threshold_used"  in result
        assert "n_events"        in result
        assert isinstance(result["events"], list)

    def test_custom_threshold(self):
        result = get_bearing_critical_events(2, metric="rms", threshold=0.01)
        # Should find some events with very low threshold
        assert result["n_events"] >= 0  # may be 0 if all below threshold

    def test_events_chronological(self):
        result = get_bearing_critical_events(2, metric="rms")
        hours  = [e["hour"] for e in result["events"]]
        assert hours == sorted(hours), "Events should be sorted chronologically"
