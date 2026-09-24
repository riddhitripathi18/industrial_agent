"""
Tests for Milling Data Tool (Predictive Maintenance Dataset ai4i2020.csv)
========================================================================
"""

import pytest
from agent.data_tools.milling_data import (
    get_milling_fleet_overview,
    get_milling_machine_telemetry,
    filter_high_risk_milling_machines,
)


class TestMillingData:

    def test_fleet_overview_keys(self):
        summary = get_milling_fleet_overview()
        assert summary["total_machines"] == 10000
        assert "failure_mode_breakdown" in summary
        assert "TWF_tool_wear_failures" in summary["failure_mode_breakdown"]
        assert "HDF_heat_dissipation_failures" in summary["failure_mode_breakdown"]
        assert "PWF_power_failures" in summary["failure_mode_breakdown"]
        assert "OSF_overstrain_failures" in summary["failure_mode_breakdown"]
        assert "RNF_random_failures" in summary["failure_mode_breakdown"]
        assert summary["total_failures"] > 0

    def test_single_machine_telemetry_valid(self):
        tel = get_milling_machine_telemetry(1)
        assert tel["udi"] == 1
        assert tel["product_id"] == "M14860"
        assert tel["type"] == "M"
        assert "air_temperature_K" in tel
        assert "process_temperature_K" in tel
        assert "rotational_speed_rpm" in tel
        assert "torque_Nm" in tel
        assert "tool_wear_min" in tel
        assert "failure_modes" in tel

    def test_single_machine_telemetry_invalid(self):
        tel = get_milling_machine_telemetry(999999)
        assert "error" in tel

    def test_filter_high_risk_machines(self):
        high_risk = filter_high_risk_milling_machines(min_tool_wear=200, limit=3)
        assert len(high_risk) == 3
        for item in high_risk:
            assert item["tool_wear_min"] >= 200
