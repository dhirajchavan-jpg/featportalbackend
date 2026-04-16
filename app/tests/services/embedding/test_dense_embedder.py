import pytest
from unittest.mock import MagicMock, patch, ANY
import numpy as np
from app.services.embedding.dense_embedder import DenseEmbedder
from app.config import settings  # Import settings directly
import requests

# --- Fixtures ---

@pytest.fixture
def embedder():
    """Return a fresh instance of DenseEmbedder."""
    return DenseEmbedder()

# --- Tests ---

def test_initialization(embedder):
    """Test initialization logging."""
    # Since init is just logging, we ensure it instantiates without error
    assert isinstance(embedder, DenseEmbedder)

def test_embed_documents_success(embedder):
    """Test successful embedding request."""
    texts = ["hello", "world"]
    fake_embeddings = [[0.1, 0.2], [0.3, 0.4]]
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": fake_embeddings}
    
    with patch("app.services.embedding.dense_embedder.requests.post", return_value=mock_response) as mock_post:
        result = embedder.embed_documents(texts)
        
        assert len(result) == 2
        assert result == fake_embeddings
        
        # Verify Payload
        # FIX: Added /embed to the URL to match actual implementation
        mock_post.assert_called_once_with(
            f"{settings.MODEL_SERVER_URLS}/embed", 
            json={"texts": texts}, 
            timeout=300
        )

def test_embed_documents_empty_input(embedder):
    """Test empty input handling."""
    result = embedder.embed_documents([])
    assert result == []

def test_embed_documents_server_error(embedder):
    """Test handling of 500 error from server."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Error"
    
    with patch("app.services.embedding.dense_embedder.requests.post", return_value=mock_response):
        result = embedder.embed_documents(["test"])
        assert result == []

def test_embed_documents_connection_error(embedder):
    """Test handling of connection refusal."""
    with patch("app.services.embedding.dense_embedder.requests.post", side_effect=requests.exceptions.ConnectionError):
        result = embedder.embed_documents(["test"])
        assert result == []

def test_embed_query(embedder):
    """Test single query embedding wrapper."""
    fake_embedding = [0.1, 0.9]
    
    with patch.object(embedder, 'embed_documents', return_value=[fake_embedding]) as mock_batch:
        result = embedder.embed_query("query")
        
        assert result == fake_embedding
        mock_batch.assert_called_once_with(["query"])

def test_embed_query_failure(embedder):
    """Test query embedding when batch fails."""
    with patch.object(embedder, 'embed_documents', return_value=[]):
        result = embedder.embed_query("query")
        assert result == []

def test_get_embedding_dimension(embedder):
    """Test dimension check."""
    assert embedder.get_embedding_dimension() == 1024

def test_compute_similarity(embedder):
    """Test cosine similarity calculation locally."""
    vec1 = [1.0, 0.0]
    vec2 = [1.0, 0.0]
    # Dot = 1, Norm1=1, Norm2=1 -> Sim = 1.0
    assert embedder.compute_similarity(vec1, vec2) == 1.0
    
    vec3 = [0.0, 1.0]
    # Dot = 0 -> Sim = 0.0
    assert embedder.compute_similarity(vec1, vec3) == 0.0

def test_compute_similarity_zero_vector(embedder):
    """Test division by zero protection."""
    vec1 = [0.0, 0.0]
    vec2 = [1.0, 1.0]
    assert embedder.compute_similarity(vec1, vec2) == 0.0

def test_compute_similarity_exception(embedder):
    """Test exception handling in math (e.g. invalid types)."""
    # Passing None to provoke numpy error or generic exception
    assert embedder.compute_similarity(None, None) == 0.0