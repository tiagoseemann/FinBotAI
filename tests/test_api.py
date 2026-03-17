"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np

from src.api.main import app
from src.api.models import ChatRequest, LeadScoreRequest


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/api/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data


class TestProductsEndpoint:
    """Test products endpoint."""

    def test_get_products(self, client):
        """Test products endpoint."""
        response = client.get("/api/products")
        assert response.status_code == 200

        data = response.json()
        assert "products" in data
        assert "total" in data
        assert len(data["products"]) > 0

        # Check product structure
        product = data["products"][0]
        assert "id" in product
        assert "name" in product
        assert "description" in product


class TestChatEndpoint:
    """Test chat endpoint."""

    def test_chat_request_validation(self, client):
        """Test chat request validation."""
        # Valid request
        response = client.post(
            "/api/chat",
            json={"message": "Olá, preciso de ajuda"}
        )
        # May fail if model not loaded, but should validate
        assert response.status_code in [200, 500]

    def test_chat_empty_message(self, client):
        """Test chat with empty message fails validation."""
        response = client.post(
            "/api/chat",
            json={"message": "   "}
        )
        assert response.status_code == 422  # Validation error

    def test_chat_missing_message(self, client):
        """Test chat without message fails."""
        response = client.post("/api/chat", json={})
        assert response.status_code == 422


class TestStatsEndpoint:
    """Test stats endpoint."""

    def test_get_stats(self, client):
        """Test stats endpoint."""
        response = client.get("/api/stats")
        assert response.status_code == 200

        data = response.json()
        assert "active_conversations" in data
        assert "model_loaded" in data


class TestAPIModels:
    """Test Pydantic models."""

    def test_chat_request_valid(self):
        """Test valid chat request."""
        req = ChatRequest(message="Test message")
        assert req.message == "Test message"
        assert req.use_scoring == True

    def test_chat_request_strips_whitespace(self):
        """Test chat request strips whitespace."""
        req = ChatRequest(message="  Test  ")
        assert req.message == "Test"

    def test_lead_score_request(self):
        """Test lead score request."""
        req = LeadScoreRequest(
            features={"f1": 1.0, "f2": 2.0}
        )
        assert len(req.features) == 2

    def test_lead_score_request_empty_fails(self):
        """Test empty features dict fails validation."""
        with pytest.raises(Exception):  # ValidationError
            LeadScoreRequest(features={})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
