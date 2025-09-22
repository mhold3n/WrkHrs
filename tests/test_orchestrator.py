import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'orchestrator'))

from app import (
    api, 
    search_knowledge_base, 
    transcribe_audio, 
    get_domain_data, 
    get_available_tools,
    analyze_request,
    gather_context,
    generate_response,
    create_workflow,
    WorkflowState
)


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(api)


@pytest.fixture
def sample_workflow_state():
    """Create sample workflow state for testing"""
    return WorkflowState(
        messages=[{"role": "user", "content": "test message"}],
        current_step="analyze",
        context={},
        tools_available=[],
        domain_weights={"chemistry": 0.5, "mechanical": 0.3, "materials": 0.2},
        response="",
        metadata={}
    )


class TestTools:
    """Test tool functions"""

    @patch('app.requests.post')
    def test_search_knowledge_base_success(self, mock_post):
        """Test successful knowledge base search"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [{"text": "test result", "score": 0.95}],
            "status": "success"
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = search_knowledge_base("test query", {"chemistry": 0.8})
        
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["text"] == "test result"

    @patch('app.requests.post')
    def test_search_knowledge_base_failure(self, mock_post):
        """Test knowledge base search failure"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        result = search_knowledge_base("test query")
        
        assert result["error"] is not None
        assert "results" not in result

    @patch('app.requests.post')
    def test_transcribe_audio_success(self, mock_post):
        """Test successful audio transcription"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "transcription": "test transcription",
            "segments": [{"start": 0, "end": 5, "text": "test"}]
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        result = transcribe_audio("base64_audio_data")
        
        assert result["transcription"] == "test transcription"
        assert len(result["segments"]) == 1

    @patch('app.requests.get')
    def test_get_domain_data_success(self, mock_get):
        """Test successful domain data retrieval"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {"property": "value"},
            "domain": "chemistry"
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = get_domain_data("chemistry", "test query")
        
        assert result["data"]["property"] == "value"
        assert result["domain"] == "chemistry"

    @patch('app.requests.get')
    def test_get_available_tools_success(self, mock_get):
        """Test successful tool registry query"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "tools": [
                {"name": "calculator", "description": "Basic calculator"},
                {"name": "converter", "description": "Unit converter"}
            ]
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = get_available_tools()
        
        assert len(result["tools"]) == 2
        assert result["tools"][0]["name"] == "calculator"


class TestWorkflowFunctions:
    """Test workflow step functions"""

    def test_analyze_request_basic(self, sample_workflow_state):
        """Test basic request analysis"""
        result = analyze_request(sample_workflow_state)
        
        assert result["current_step"] == "gather_context"
        assert "analysis" in result["metadata"]
        assert result["metadata"]["needs_context"] is True

    def test_analyze_request_chemistry_detection(self, sample_workflow_state):
        """Test chemistry domain detection in request analysis"""
        sample_workflow_state.messages = [
            {"role": "user", "content": "What is the molecular weight of H2O?"}
        ]
        
        result = analyze_request(sample_workflow_state)
        
        assert result["domain_weights"]["chemistry"] > 0.5

    def test_analyze_request_mechanical_detection(self, sample_workflow_state):
        """Test mechanical domain detection in request analysis"""
        sample_workflow_state.messages = [
            {"role": "user", "content": "Calculate the stress on a steel beam under 100N force"}
        ]
        
        result = analyze_request(sample_workflow_state)
        
        assert result["domain_weights"]["mechanical"] > 0.5

    @patch('app.search_knowledge_base')
    @patch('app.get_available_tools')
    def test_gather_context_success(self, mock_tools, mock_search, sample_workflow_state):
        """Test successful context gathering"""
        # Mock responses
        mock_search.return_value = {
            "results": [{"text": "relevant context", "score": 0.9}]
        }
        mock_tools.return_value = {
            "tools": [{"name": "calculator", "description": "Basic calculator"}]
        }
        
        sample_workflow_state.current_step = "gather_context"
        result = gather_context(sample_workflow_state)
        
        assert result["current_step"] == "generate"
        assert "knowledge_results" in result["context"]
        assert len(result["tools_available"]) > 0

    @patch('app.requests.post')
    def test_generate_response_success(self, mock_post, sample_workflow_state):
        """Test successful response generation"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated response"}}]
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        sample_workflow_state.current_step = "generate"
        sample_workflow_state.context = {"knowledge_results": [{"text": "context"}]}
        
        result = generate_response(sample_workflow_state)
        
        assert result["response"] == "Generated response"
        assert result["current_step"] == "complete"


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check_success(self, client):
        """Test successful health check"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "workflow_ready" in data


class TestChatEndpoint:
    """Test chat endpoint"""

    @patch('app.create_workflow')
    def test_chat_endpoint_success(self, mock_workflow, client):
        """Test successful chat endpoint"""
        # Mock workflow
        mock_workflow_instance = Mock()
        mock_workflow_instance.invoke.return_value = {
            "response": "Test response",
            "current_step": "complete",
            "metadata": {"tokens_used": 50}
        }
        mock_workflow.return_value = mock_workflow_instance
        
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test message"}],
            "model": "test-model"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["content"] == "Test response"

    def test_chat_endpoint_missing_messages(self, client):
        """Test chat endpoint with missing messages"""
        response = client.post("/v1/chat/completions", json={
            "model": "test-model"
        })
        
        assert response.status_code == 422  # Validation error

    @patch('app.create_workflow')
    def test_chat_endpoint_workflow_error(self, mock_workflow, client):
        """Test chat endpoint with workflow error"""
        mock_workflow_instance = Mock()
        mock_workflow_instance.invoke.side_effect = Exception("Workflow error")
        mock_workflow.return_value = mock_workflow_instance
        
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test message"}],
            "model": "test-model"
        })
        
        assert response.status_code == 500


class TestWorkflowStatus:
    """Test workflow status endpoint"""

    def test_workflow_status_success(self, client):
        """Test workflow status endpoint"""
        response = client.get("/workflow/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "active_workflows" in data
        assert "total_processed" in data
        assert "average_response_time" in data


class TestWorkflowCreation:
    """Test workflow creation and configuration"""

    def test_create_workflow(self):
        """Test workflow creation"""
        workflow = create_workflow()
        
        # Check that workflow is properly configured
        assert workflow is not None
        # Workflow should have the required nodes and edges
        # This is a basic test - in practice you'd test the graph structure


class TestWorkflowState:
    """Test WorkflowState model"""

    def test_workflow_state_creation(self):
        """Test WorkflowState creation with default values"""
        state = WorkflowState(
            messages=[{"role": "user", "content": "test"}]
        )
        
        assert len(state.messages) == 1
        assert state.current_step == "analyze"
        assert state.context == {}
        assert state.tools_available == []
        assert state.response == ""

    def test_workflow_state_with_custom_values(self):
        """Test WorkflowState creation with custom values"""
        state = WorkflowState(
            messages=[{"role": "user", "content": "test"}],
            current_step="generate",
            context={"key": "value"},
            domain_weights={"chemistry": 0.8}
        )
        
        assert state.current_step == "generate"
        assert state.context["key"] == "value"
        assert state.domain_weights["chemistry"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__])