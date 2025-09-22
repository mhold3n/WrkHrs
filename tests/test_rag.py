import pytest
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime
import io

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'rag'))

from app import api, rag_service, RAGService


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(api)


@pytest.fixture
def mock_rag_service():
    """Create mock RAG service for testing"""
    service = Mock(spec=RAGService)
    service.embedding_model = Mock()
    service.qdrant_client = Mock()
    service.collection_name = "test_collection"
    return service


@pytest.fixture
def sample_document():
    """Sample document for testing"""
    return {
        "content": "This is a test document about molecular chemistry and H2O properties.",
        "metadata": {"title": "Test Document", "author": "Test Author"},
        "domain": "chemistry",
        "source": "test_source"
    }


class TestRAGService:
    """Test RAG service functionality"""

    def test_rag_service_initialization(self):
        """Test RAG service initialization"""
        with patch('app.SentenceTransformer') as mock_transformer, \
             patch('app.QdrantClient') as mock_client:
            
            mock_transformer.return_value = Mock()
            mock_client.return_value = Mock()
            
            service = RAGService()
            service.initialize()
            
            assert service.embedding_model is not None
            assert service.qdrant_client is not None

    def test_chunk_text_basic(self):
        """Test basic text chunking"""
        service = RAGService()
        text = "This is a long text that needs to be chunked into smaller pieces for better processing and retrieval."
        
        chunks = service.chunk_text(text, chunk_size=50, overlap=10)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 60 for chunk in chunks)  # 50 + 10 overlap buffer

    def test_chunk_text_short_text(self):
        """Test chunking with text shorter than chunk size"""
        service = RAGService()
        text = "Short text."
        
        chunks = service.chunk_text(text, chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_calculate_domain_score_chemistry(self):
        """Test domain score calculation for chemistry"""
        service = RAGService()
        text = "The molecular weight of H2O is 18 g/mol with pH levels affecting solubility."
        domain_weights = {"chemistry": 0.8, "mechanical": 0.1, "materials": 0.1}
        
        score = service.calculate_domain_score(text, domain_weights)
        
        assert score > 0.5  # Should have high chemistry relevance

    def test_calculate_domain_score_mechanical(self):
        """Test domain score calculation for mechanical"""
        service = RAGService()
        text = "The stress and strain on the beam under 100N force causes deformation."
        domain_weights = {"chemistry": 0.1, "mechanical": 0.8, "materials": 0.1}
        
        score = service.calculate_domain_score(text, domain_weights)
        
        assert score > 0.5  # Should have high mechanical relevance

    @patch('app.rag_service.qdrant_client')
    @patch('app.rag_service.embedding_model')
    def test_add_document_success(self, mock_embedding, mock_client, sample_document):
        """Test successful document addition"""
        # Mock embedding generation
        mock_embedding.encode.return_value = np.random.rand(384).tolist()
        
        # Mock Qdrant client
        mock_client.upsert = Mock()
        
        # Test adding document
        with patch.object(rag_service, 'chunk_text') as mock_chunk:
            mock_chunk.return_value = [sample_document["content"]]
            
            result = rag_service.add_document(
                content=sample_document["content"],
                metadata=sample_document["metadata"],
                domain=sample_document["domain"],
                source=sample_document["source"]
            )
            
            assert result["status"] == "success"
            assert result["chunks_added"] == 1
            mock_client.upsert.assert_called_once()

    @patch('app.rag_service.qdrant_client')
    @patch('app.rag_service.embedding_model')
    def test_search_documents_success(self, mock_embedding, mock_client):
        """Test successful document search"""
        # Mock embedding generation
        mock_embedding.encode.return_value = np.random.rand(384).tolist()
        
        # Mock Qdrant search results
        mock_result = Mock()
        mock_result.payload = {
            "text": "Sample result text",
            "metadata": {"title": "Test"},
            "domain": "chemistry",
            "domain_score": 0.8
        }
        mock_result.score = 0.9
        
        mock_client.search.return_value = [mock_result]
        
        # Test search
        results = rag_service.search(
            query="molecular weight",
            domain_weights={"chemistry": 0.8},
            k=5,
            threshold=0.7
        )
        
        assert len(results) == 1
        assert results[0]["score"] == 0.9
        assert results[0]["text"] == "Sample result text"

    @patch('app.rag_service.qdrant_client')
    @patch('app.rag_service.embedding_model')
    def test_search_documents_no_results(self, mock_embedding, mock_client):
        """Test search with no results above threshold"""
        # Mock embedding generation
        mock_embedding.encode.return_value = np.random.rand(384).tolist()
        
        # Mock low-score results
        mock_result = Mock()
        mock_result.score = 0.5  # Below threshold
        mock_client.search.return_value = [mock_result]
        
        # Test search with high threshold
        results = rag_service.search(
            query="test query",
            threshold=0.8
        )
        
        assert len(results) == 0


class TestHealthEndpoint:
    """Test health check endpoint"""

    @patch('app.rag_service')
    def test_health_check_success(self, mock_service, client):
        """Test successful health check"""
        # Mock service state
        mock_service.embedding_model = Mock()
        mock_service.qdrant_client = Mock()
        mock_service.qdrant_client.get_collection.return_value = Mock()
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "embedding_model_loaded" in data
        assert "qdrant_connected" in data

    @patch('app.rag_service')
    def test_health_check_qdrant_error(self, mock_service, client):
        """Test health check with Qdrant connection error"""
        mock_service.embedding_model = Mock()
        mock_service.qdrant_client = Mock()
        mock_service.qdrant_client.get_collection.side_effect = Exception("Connection error")
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["qdrant_connected"] is False


class TestSearchEndpoint:
    """Test search documents endpoint"""

    @patch('app.rag_service')
    def test_search_endpoint_success(self, mock_service, client):
        """Test successful search endpoint"""
        # Mock search results
        mock_service.search.return_value = [
            {
                "text": "Test result",
                "score": 0.9,
                "metadata": {"title": "Test Doc"},
                "domain": "chemistry"
            }
        ]
        
        response = client.post("/search", json={
            "query": "molecular weight",
            "domain_weights": {"chemistry": 0.8},
            "k": 5,
            "threshold": 0.7
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["score"] == 0.9

    @patch('app.rag_service')
    def test_search_endpoint_error(self, mock_service, client):
        """Test search endpoint with service error"""
        mock_service.search.side_effect = Exception("Search error")
        
        response = client.post("/search", json={
            "query": "test query"
        })
        
        assert response.status_code == 500

    def test_search_endpoint_missing_query(self, client):
        """Test search endpoint with missing query"""
        response = client.post("/search", json={})
        
        assert response.status_code == 422  # Validation error


class TestAddDocumentEndpoint:
    """Test add document endpoint"""

    @patch('app.rag_service')
    def test_add_document_success(self, mock_service, client):
        """Test successful document addition"""
        mock_service.add_document.return_value = {
            "status": "success",
            "chunks_added": 3,
            "document_id": "test_doc_123"
        }
        
        response = client.post("/documents", json={
            "content": "Test document content",
            "metadata": {"title": "Test"},
            "domain": "chemistry"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["chunks_added"] == 3

    @patch('app.rag_service')
    def test_add_document_error(self, mock_service, client):
        """Test document addition with service error"""
        mock_service.add_document.side_effect = Exception("Add document error")
        
        response = client.post("/documents", json={
            "content": "Test content"
        })
        
        assert response.status_code == 500


class TestUploadDocumentEndpoint:
    """Test upload document endpoint"""

    @patch('app.rag_service')
    def test_upload_document_success(self, mock_service, client):
        """Test successful document upload"""
        mock_service.add_document.return_value = {
            "status": "success",
            "chunks_added": 2,
            "document_id": "uploaded_doc_456"
        }
        
        # Create test file
        test_content = b"This is a test document for upload."
        test_file = io.BytesIO(test_content)
        
        response = client.post("/documents/upload", 
            files={"file": ("test.txt", test_file, "text/plain")},
            data={"domain": "materials", "metadata": "{}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_upload_document_invalid_file_type(self, client):
        """Test upload with invalid file type"""
        test_content = b"Binary content"
        test_file = io.BytesIO(test_content)
        
        response = client.post("/documents/upload",
            files={"file": ("test.bin", test_file, "application/octet-stream")},
            data={"domain": "general"}
        )
        
        assert response.status_code == 400


class TestCollectionEndpoints:
    """Test collection management endpoints"""

    @patch('app.rag_service')
    def test_get_collection_info_success(self, mock_service, client):
        """Test successful collection info retrieval"""
        mock_collection = Mock()
        mock_collection.points_count = 100
        mock_collection.vectors_count = 100
        mock_service.qdrant_client.get_collection.return_value = mock_collection
        
        response = client.get("/collection/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "collection_name" in data
        assert "points_count" in data

    @patch('app.rag_service')
    def test_clear_collection_success(self, mock_service, client):
        """Test successful collection clearing"""
        mock_service.qdrant_client.delete_collection = Mock()
        mock_service.ensure_collection_exists = Mock()
        
        response = client.delete("/collection/clear")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "cleared" in data["message"]


class TestDomainScoring:
    """Test domain-specific scoring functionality"""

    def test_domain_score_mixed_content(self):
        """Test domain scoring with mixed content"""
        service = RAGService()
        text = "Steel beam analysis shows stress patterns in molecular bonding."
        
        # Test different domain weights
        chem_weights = {"chemistry": 0.8, "mechanical": 0.1, "materials": 0.1}
        mech_weights = {"chemistry": 0.1, "mechanical": 0.8, "materials": 0.1}
        
        chem_score = service.calculate_domain_score(text, chem_weights)
        mech_score = service.calculate_domain_score(text, mech_weights)
        
        # Both should have some relevance, but different weightings
        assert chem_score > 0
        assert mech_score > 0

    def test_domain_score_no_weights(self):
        """Test domain scoring without weights"""
        service = RAGService()
        text = "General text content."
        
        score = service.calculate_domain_score(text, None)
        
        assert score >= 0
        assert score <= 1


if __name__ == "__main__":
    pytest.main([__file__])