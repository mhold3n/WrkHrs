import pytest
import json
import hashlib
import time
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'gateway'))

from app import api, validate_api_key, check_rate_limit, create_jwt_token, verify_jwt_token, extract_domain_weights


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(api)


@pytest.fixture
def api_key():
    """Generate test API key"""
    secret = "test-secret"
    return hashlib.sha256(secret.encode()).hexdigest()


@pytest.fixture
def jwt_token():
    """Generate test JWT token"""
    return create_jwt_token({"sub": "test_user", "username": "test_user"})


class TestAuthentication:
    """Test authentication functionality"""

    def test_validate_api_key_success(self):
        """Test successful API key validation"""
        with patch.dict(os.environ, {"API_KEY_SECRET": "test-secret"}):
            # The API key should be the secret itself, not the hash
            assert validate_api_key("test-secret") == True

    def test_validate_api_key_failure(self):
        """Test failed API key validation"""
        with patch.dict(os.environ, {"API_KEY_SECRET": "test-secret"}):
            assert validate_api_key("wrong-key") == False

    def test_validate_api_key_empty(self):
        """Test empty API key validation"""
        assert validate_api_key("") == False
        assert validate_api_key(None) == False

    def test_jwt_token_creation_and_verification(self):
        """Test JWT token creation and verification"""
        test_data = {"sub": "test_user", "username": "test_user"}
        token = create_jwt_token(test_data)
        
        # Verify token
        payload = verify_jwt_token(token)
        assert payload is not None
        assert payload["sub"] == "test_user"
        assert payload["username"] == "test_user"

    def test_jwt_token_invalid(self):
        """Test invalid JWT token"""
        payload = verify_jwt_token("invalid.token.here")
        assert payload is None

    @patch('app.request_counts')
    def test_rate_limiting(self, mock_request_counts):
        """Test rate limiting functionality"""
        client_ip = "192.168.1.1"
        
        # Mock initial state
        mock_request_counts.__getitem__ = Mock(return_value=[])
        mock_request_counts.__setitem__ = Mock()
        
        # First request should pass
        with patch.dict(os.environ, {"RATE_LIMIT_REQUESTS_PER_MINUTE": "5"}):
            assert check_rate_limit(client_ip) == True

    def test_rate_limiting_disabled_auth(self):
        """Test rate limiting when auth is disabled"""
        with patch.dict(os.environ, {"ENABLE_AUTHENTICATION": "false"}):
            assert check_rate_limit("any_ip") == True


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check_success(self, client):
        """Test successful health check"""
        with patch.dict(os.environ, {"ENABLE_AUTHENTICATION": "false"}):
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert "embedding_model_loaded" in data
            assert "authentication_enabled" in data


class TestLoginEndpoint:
    """Test login functionality"""

    def test_login_success(self, client):
        """Test successful login"""
        with patch.dict(os.environ, {"API_KEY_SECRET": "test-password"}):
            response = client.post("/auth/login", json={
                "username": "admin",
                "password": "test-password"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert data["token_type"] == "bearer"
            assert "expires_in" in data

    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        response = client.post("/auth/login", json={
            "username": "admin",
            "password": "wrong-password"
        })
        
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"


class TestDomainWeighting:
    """Test domain weighting functionality"""

    def test_extract_domain_weights_chemistry(self):
        """Test chemistry domain weight extraction"""
        text = "What is the molecular weight of H2O and its pH level?"
        weights = extract_domain_weights(text)
        
        assert weights.chemistry > 0
        assert weights.mechanical == 0
        assert weights.materials == 0

    def test_extract_domain_weights_mechanical(self):
        """Test mechanical domain weight extraction"""
        text = "Calculate the stress and strain on this beam under 100N force"
        weights = extract_domain_weights(text)
        
        assert weights.chemistry == 0
        assert weights.mechanical > 0
        assert weights.materials == 0

    def test_extract_domain_weights_materials(self):
        """Test materials domain weight extraction"""
        text = "What is the hardness of steel and aluminum alloy properties?"
        weights = extract_domain_weights(text)
        
        assert weights.chemistry == 0
        assert weights.mechanical == 0
        assert weights.materials > 0

    def test_extract_domain_weights_mixed(self):
        """Test mixed domain weight extraction"""
        text = "Steel beam stress analysis with chemical corrosion effects"
        weights = extract_domain_weights(text)
        
        # Should have weights in multiple domains
        assert weights.mechanical > 0
        assert weights.materials > 0


class TestChatCompletions:
    """Test chat completions endpoint"""

    def test_chat_completions_without_auth(self, client):
        """Test chat completions when auth is disabled"""
        with patch.dict(os.environ, {"ENABLE_AUTHENTICATION": "false"}):
            with patch('app.requests.post') as mock_post:
                # Mock the orchestrator response
                mock_response = Mock()
                mock_response.json.return_value = {"response": "test response"}
                mock_response.status_code = 200
                mock_post.return_value = mock_response
                
                response = client.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "test message"}],
                    "model": "test-model"
                })
                
                assert response.status_code == 200

    def test_chat_completions_with_valid_api_key(self, client):
        """Test chat completions with valid API key"""
        with patch.dict(os.environ, {
            "ENABLE_AUTHENTICATION": "true",
            "API_KEY_SECRET": "test-secret"
        }):
            with patch('app.requests.post') as mock_post:
                # Mock the orchestrator response
                mock_response = Mock()
                mock_response.json.return_value = {"response": "test response"}
                mock_response.status_code = 200
                mock_post.return_value = mock_response
                
                response = client.post("/v1/chat/completions", 
                    json={
                        "messages": [{"role": "user", "content": "test message"}],
                        "model": "test-model"
                    },
                    headers={"X-API-Key": "test-secret"}
                )
                
                assert response.status_code == 200

    def test_chat_completions_unauthorized(self, client):
        """Test chat completions without authentication"""
        with patch.dict(os.environ, {"ENABLE_AUTHENTICATION": "true"}):
            response = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "test message"}],
                "model": "test-model"
            })
            
            assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__])