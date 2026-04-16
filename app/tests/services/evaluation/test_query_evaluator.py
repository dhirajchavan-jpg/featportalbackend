import pytest
from unittest.mock import MagicMock, patch, ANY
import numpy as np
from app.services.evaluation.query_evaluator import (
    QueryProcessingEvaluator,
    evaluate_query_processing
)

# --- Fixtures ---

@pytest.fixture
def mock_httpx():
    """
    Mocks httpx.Client to simulate the embedding server response.
    This replaces the local SentenceTransformer mock.
    """
    # We patch where httpx is used in the module
    with patch("app.services.evaluation.query_evaluator.httpx.Client") as MockClient:
        # Mock context manager
        mock_client_instance = MagicMock()
        MockClient.return_value.__enter__.return_value = mock_client_instance

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Return dummy embeddings (list of 384-dim vectors)
        # We return 2 vectors: one for 'original', one for 'expanded'
        dummy_vec_a = [0.1] * 384
        dummy_vec_b = [0.2] * 384 # Different to allow similarity calc
        mock_response.json.return_value = {"embeddings": [dummy_vec_a, dummy_vec_b]}
        
        mock_client_instance.post.return_value = mock_response
        yield mock_client_instance.post

@pytest.fixture
def evaluator():
    """Returns an initialized evaluator."""
    return QueryProcessingEvaluator()

# --- Tests ---

def test_initialization_success():
    """Test successful initialization."""
    evaluator = QueryProcessingEvaluator()
    assert isinstance(evaluator, QueryProcessingEvaluator)
    # Check for server_nodes instead of model
    assert hasattr(evaluator, "server_nodes")

def test_initialization_failure():
    """Test initialization is robust (it usually doesn't fail unless config is missing)."""
    evaluator = QueryProcessingEvaluator()
    assert evaluator.server_nodes is not None

def test_evaluate_query_processing_metrics(evaluator, mock_httpx):
    """Test calculation of basic metrics and semantic score."""
    original = "RAG"
    expanded = "Retrieval Augmented Generation"
    
    metrics = evaluator.evaluate_query_processing(original, expanded, processing_time_ms=10.0)
    
    # Basic Metrics
    assert metrics["processing_time_ms"] == 10.0
    assert metrics["original_length"] == 3
    assert metrics["expanded_length"] == 30
    assert metrics["expansion_ratio"] == 10.0
    
    # Semantic Metrics (Calculated via mocked embeddings)
    assert "semantic_similarity" in metrics
    assert isinstance(metrics["semantic_similarity"], float)
    
    # Expansion Quality Score
    assert "expansion_quality_score" in metrics
    assert metrics["expansion_quality_score"] >= 0

def test_analyze_clarity_improvement(evaluator):
    """Test clarity heuristics (Logic only)."""
    original = "rag system"
    expanded = "Retrieval Augmented Generation system."
    
    metrics = evaluator._analyze_clarity_improvement(original, expanded)
    
    # Expectations based on heuristics:
    # 1. Added context: len(expanded) > len(original)*1.2
    assert metrics.get("improvement_added_context") is True
    # 2. Proper capitalization: Starts with Upper
    assert metrics.get("improvement_proper_capitalization") is True
    # 3. Punctuation: Ends with '.'
    assert metrics.get("improvement_ends_with_punctuation") is True
    
    assert metrics["clarity_score"] > 0.0

def test_analyze_semantic_preservation_failure(evaluator, mock_httpx):
    """Test behavior when remote server fails."""
    # Simulate server failure
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_httpx.return_value = mock_response
    
    metrics = evaluator._analyze_semantic_preservation("a", "b")
    # Should handle gracefully (return empty dict)
    assert metrics == {}

def test_calculate_expansion_quality_logic(evaluator):
    """Test score calculation logic."""
    # Case 1: Good Expansion
    metrics_good = {
        'semantic_similarity': 0.9, 
        'clarity_score': 1.0, 
        'expansion_ratio': 1.2 
    }
    score = evaluator._calculate_expansion_quality(metrics_good)
    assert score > 0.8

    # Case 2: Too much expansion (hallucination risk)
    metrics_verbose = {
        'semantic_similarity': 0.5, 
        'clarity_score': 0.5,
        'expansion_ratio': 3.0 # > 1.5 threshold
    }
    score = evaluator._calculate_expansion_quality(metrics_verbose)
    # Penalty should reduce score significantly
    assert score < 0.5

def test_convenience_function(mock_httpx):
    """Test the standalone wrapper function."""
    result = evaluate_query_processing("a", "b")
    assert "expansion_quality_score" in result