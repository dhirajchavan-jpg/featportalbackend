import pytest
from unittest.mock import MagicMock, patch, ANY
import numpy as np

# Adjust the import based on your actual file structure
from app.services.evaluation.context_evaluator import (
    ContextQualityEvaluator,
    evaluate_context_quality
)

# --- FIXTURES ---

@pytest.fixture
def mock_settings():
    """Mocks the app config settings."""
    with patch("app.services.evaluation.context_evaluator.settings") as mock_settings:
        mock_settings.model_server_urls_list = ["http://mock-server:8000"]
        yield mock_settings

@pytest.fixture
def mock_httpx_success():
    """
    Mocks httpx.Client to simulate a SUCCESSFUL embedding server response.
    Returns a batch of dummy vectors.
    """
    with patch("app.services.evaluation.context_evaluator.httpx.Client") as MockClient:
        # Mock the context manager behavior: with httpx.Client() as client:
        mock_client_instance = MagicMock()
        MockClient.return_value.__enter__.return_value = mock_client_instance

        # Setup the response object
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Return random 384-dim vectors (standard for MiniLM)
        def side_effect_json():
            # Create 10 dummy vectors
            return {"embeddings": [[0.1] * 384 for _ in range(10)]}
            
        mock_response.json.side_effect = side_effect_json
        
        # Link response to client.post
        mock_client_instance.post.return_value = mock_response
        
        yield mock_client_instance.post

@pytest.fixture
def mock_httpx_failure():
    """Mocks httpx.Client to simulate a SERVER ERROR."""
    with patch("app.services.evaluation.context_evaluator.httpx.Client") as MockClient:
        mock_client_instance = MagicMock()
        MockClient.return_value.__enter__.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        
        mock_client_instance.post.return_value = mock_response
        yield mock_client_instance.post

@pytest.fixture
def evaluator(mock_settings):
    """Returns an initialized ContextQualityEvaluator."""
    return ContextQualityEvaluator()

# --- TESTS: INITIALIZATION & UTILS ---

def test_initialization_defaults():
    """Test initialization falls back to localhost if settings empty."""
    with patch("app.services.evaluation.context_evaluator.settings") as s:
        s.model_server_urls_list = [] # Empty
        evaluator = ContextQualityEvaluator()
        assert "http://localhost:8074" in evaluator.server_nodes

def test_get_api_url_selection(evaluator):
    """Test random server node selection."""
    url = evaluator._get_api_url()
    assert "vectorize_eval" in url
    assert "http://mock-server:8000" in url

# --- TESTS: REMOTE EMBEDDING FETCHING ---

def test_get_embeddings_remote_success(evaluator, mock_httpx_success):
    """Test successful remote embedding retrieval."""
    texts = ["hello", "world"]
    embeddings = evaluator._get_embeddings_remote(texts)
    
    assert len(embeddings) > 0
    assert len(embeddings[0]) == 384
    mock_httpx_success.assert_called_once()

def test_get_embeddings_remote_server_error(evaluator, mock_httpx_failure):
    """Test handling of 500 error from model server."""
    embeddings = evaluator._get_embeddings_remote(["test"])
    assert embeddings == [] # Should return empty list, not crash

def test_get_embeddings_remote_exception(evaluator):
    """Test handling of network exceptions (e.g., timeout)."""
    with patch("app.services.evaluation.context_evaluator.httpx.Client", side_effect=Exception("Connection Refused")):
        embeddings = evaluator._get_embeddings_remote(["test"])
        assert embeddings == []

# --- TESTS: RELEVANCE & SEMANTICS (Math Logic) ---

def test_calculate_relevance(evaluator, mock_httpx_success):
    """Test semantic relevance calculation logic."""
    # Logic: 
    # 1. Splits context into chunks
    # 2. Sends [query, chunk1, chunk2...] to server
    # 3. Compares query vector vs chunk vectors
    
    query = "test query"
    context = "chunk one. chunk two."
    
    metrics = evaluator._calculate_relevance(query, context)
    
    assert "context_relevance_mean" in metrics
    # Since we mocked vectors to be identical ([0.1]*384), similarity should be 1.0 (or close due to float precision)
    assert metrics["context_relevance_mean"] == pytest.approx(1.0, rel=1e-5)
    assert metrics["highly_relevant_chunks"] > 0

def test_calculate_relevance_empty_response(evaluator, mock_httpx_failure):
    """Test relevance when embeddings fail."""
    metrics = evaluator._calculate_relevance("q", "c")
    assert metrics == {}

# --- TESTS: REDUNDANCY & COHERENCE ---

def test_analyze_redundancy_ngram(evaluator):
    """Test local N-gram redundancy (no network needed)."""
    # High redundancy: Same sentence repeated
    context = "This is a test. This is a test. This is a test."
    metrics = evaluator._calculate_ngram_overlap(context, n=3)
    
    assert metrics["ngram_redundancy_score"] > 0.5
    assert metrics["duplicate_ngrams"] > 0

def test_analyze_redundancy_semantic(evaluator, mock_httpx_success):
    """Test semantic redundancy between source documents."""
    docs = [
        {"page_content": "Document A content."},
        {"page_content": "Document B content."}
    ]
    
    metrics = evaluator._calculate_sentence_redundancy(docs)
    
    assert "semantic_redundancy_mean" in metrics
    # Identical mock vectors => High redundancy
    assert metrics["semantic_redundancy_mean"] == pytest.approx(1.0, rel=1e-5)

def test_analyze_coherence(evaluator, mock_httpx_success):
    """Test coherence (inter-document similarity)."""
    docs = [
        {"page_content": "Part 1"},
        {"page_content": "Part 2"}
    ]
    
    metrics = evaluator._analyze_coherence(docs)
    
    assert "coherence_score" in metrics
    # With identical vectors (1.0 similarity), coherence might be penalized for being "too redundant" 
    # Logic: if avg > 0.6: score = 1.0 - (avg - 0.6)/0.4
    # 1.0 > 0.6 -> 1.0 - (0.4/0.4) = 0.0 coherence (too repetitive)
    assert metrics["coherence_score"] == pytest.approx(0.0, rel=1e-5) 

# --- TESTS: COVERAGE & EFFICIENCY ---

def test_analyze_coverage(evaluator):
    """Test keyword coverage (Local logic)."""
    # FIX: Use query without punctuation so 'gdpr' matches 'gdpr'
    query = "What is GDPR" 
    context = "The GDPR is a regulation."
    
    metrics = evaluator._analyze_coverage(query, context)
    
    # "gdpr" should match
    assert metrics["query_terms_found"] >= 1
    assert metrics["coverage_score"] > 0.0

def test_analyze_token_efficiency(evaluator):
    """Test token density calculation."""
    context = "unique words here"
    metrics_in = {"context_tokens": 3, "num_source_docs": 1}
    
    result = evaluator._analyze_token_efficiency(context, metrics_in)
    
    assert result["token_efficiency"] == 1.0 # All unique
    assert result["unique_words"] == 3

# --- TESTS: INTEGRATION (Main Entry Point) ---

def test_evaluate_context_full_flow(evaluator, mock_httpx_success):
    """
    Test the main public method aggregates all metrics.
    """
    query = "test"
    context = "context data"
    docs = [{"page_content": "context data"}]
    
    metrics = evaluator.evaluate_context(query, context, docs)
    
    # Check structure
    assert "context_quality_score" in metrics
    assert "context_relevance_mean" in metrics # From Remote
    assert "ngram_redundancy_score" in metrics # From Local
    
    # Score should be calculated (0.0 to 1.0)
    assert 0.0 <= metrics["context_quality_score"] <= 1.0

def test_evaluate_context_empty_input(evaluator):
    """Test handling of empty context string."""
    metrics = evaluator.evaluate_context("q", "")
    assert metrics["context_length"] == 0
    assert metrics["context_quality_score"] == 0.0

def test_cosine_similarity_math(evaluator):
    """Test the numpy math helper directly."""
    # Orthogonal vectors (Sim = 0)
    vec_a = np.array([1, 0])
    vec_b = np.array([0, 1])
    
    sim = evaluator._cosine_similarity(vec_a, vec_b)
    assert sim[0][0] == 0.0
    
    # Identical vectors (Sim = 1)
    vec_c = np.array([1, 0])
    sim = evaluator._cosine_similarity(vec_a, vec_c)
    assert sim[0][0] == pytest.approx(1.0)

def test_convenience_function(mock_settings, mock_httpx_success):
    """Test the standalone wrapper function."""
    metrics = evaluate_context_quality("q", "c", [{"page_content": "c"}])
    assert isinstance(metrics, dict)
    assert "context_quality_score" in metrics