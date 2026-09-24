"""
Tests for Phase 5 Web Application Server (FastAPI)
===================================================
Tests health endpoints, fleet status, equipment queries,
session reset, and chat interface with mocked LLM.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.server import app


@pytest.fixture(scope="module")
def client():
    """Create FastAPI test client."""
    return TestClient(app)


class TestWebServerEndpoints:
    """Test suite for IPMA REST API."""

    def test_health_endpoint(self, client):
        """GET /api/health should return healthy status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "timestamp" in data

    def test_fleet_endpoint(self, client):
        """GET /api/fleet should return catalog and snapshot data."""
        response = client.get("/api/fleet")
        assert response.status_code == 200
        data = response.json()
        assert "catalog" in data
        assert "snapshot" in data
        assert "heat_exchangers" in data["catalog"]
        assert "bearing_tests" in data["catalog"]
        assert len(data["catalog"]["heat_exchangers"]) == 5

    def test_hx_detail_endpoint(self, client):
        """GET /api/equipment/hx/E02 should return fouling calculation."""
        response = client.get("/api/equipment/hx/E02")
        assert response.status_code == 200
        data = response.json()
        assert data["exchanger_id"] == "E02"
        assert "Rf_final_m2KW" in data
        assert "tema_breached" in data

    def test_hx_detail_invalid(self, client):
        """GET /api/equipment/hx/INVALID should return 404."""
        response = client.get("/api/equipment/hx/E99")
        assert response.status_code == 404

    def test_bearing_detail_endpoint(self, client):
        """GET /api/equipment/bearing/2/Bearing1 should return bearing diagnostic."""
        response = client.get("/api/equipment/bearing/2/Bearing1")
        assert response.status_code == 200
        data = response.json()
        assert data["test_id"] == 2
        assert data["bearing_id"] == "Bearing1"
        assert "rul_hours" in data
        assert "current_iso_zone" in data

    def test_bearing_detail_invalid(self, client):
        """GET /api/equipment/bearing/99/Bearing1 should return 404."""
        response = client.get("/api/equipment/bearing/99/Bearing1")
        assert response.status_code == 404

    def test_milling_detail_endpoint(self, client):
        """GET /api/equipment/milling/142 should return machine telemetry and prediction."""
        response = client.get("/api/equipment/milling/142")
        assert response.status_code == 200
        data = response.json()
        assert data["machine_id"] == 142
        assert "failure_modes" in data
        assert "TWF" in data["failure_modes"]
        assert "predicted_failure" in data

    def test_milling_detail_invalid(self, client):
        """GET /api/equipment/milling/99999 should return 404."""
        response = client.get("/api/equipment/milling/99999")
        assert response.status_code == 404

    def test_predict_endpoints(self, client):
        """POST and GET /api/predict for all three machine classes."""
        # 1. Milling
        res_m = client.post("/api/predict", json={"equipment_type": "milling", "equipment_id": "142"})
        assert res_m.status_code == 200
        assert "overall_failure_probability" in res_m.json()

        # 2. Heat Exchanger
        res_hx = client.post("/api/predict", json={"equipment_type": "heat_exchanger", "equipment_id": "E02"})
        assert res_hx.status_code == 200
        assert "tema_breach_probability" in res_hx.json()

        # 3. Bearing
        res_b = client.post("/api/predict", json={"equipment_type": "bearing", "equipment_id": "Bearing1", "test_id": 2})
        assert res_b.status_code == 200
        assert "degradation_stage" in res_b.json()
        assert "predicted_rul_hours" in res_b.json()

        # 4. GET endpoint
        res_get = client.get("/api/predict/milling/142")
        assert res_get.status_code == 200
        assert res_get.json()["machine_id"] == 142

    def test_chat_endpoint_mocked(self, client):
        """POST /api/chat should invoke agent.ask and return response."""
        mock_agent = MagicMock()
        mock_agent.ask.return_value = "E02 fouling resistance is 0.0011 m²K/W."
        mock_agent.model_name = "gemini-2.5-flash"

        with patch("app.server.get_agent", return_value=mock_agent):
            response = client.post("/api/chat", json={"message": "Analyze E02"})
            assert response.status_code == 200
            data = response.json()
            assert "E02 fouling resistance" in data["response"]
            assert data["model"] == "gemini-2.5-flash"
            mock_agent.ask.assert_called_once_with("Analyze E02")

    def test_chat_endpoint_empty_message(self, client):
        """POST /api/chat with whitespace should return 400 or 422."""
        mock_agent = MagicMock()
        with patch("app.server.get_agent", return_value=mock_agent):
            response = client.post("/api/chat", json={"message": "   "})
            assert response.status_code in (400, 422)

    def test_reset_endpoint(self, client):
        """POST /api/reset should call agent.reset()."""
        mock_agent = MagicMock()
        with patch("app.server.get_agent", return_value=mock_agent):
            response = client.post("/api/reset")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "reset"
            mock_agent.reset.assert_called_once()

    def test_index_html_served(self, client):
        """GET / should serve the HTML application."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "IPMA" in response.text
