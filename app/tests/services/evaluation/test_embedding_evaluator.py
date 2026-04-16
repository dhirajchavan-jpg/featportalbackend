import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import builtins

# Adjust the import based on your actual file structure
from app.services.evaluation.embedding_evaluator import (
    EmbeddingQualityEvaluator,
    evaluate_embedding_quality,
    get_mteb_reference_score
)

# --- FIXTURES ---

@pytest.fixture
def mock_sentence_transformer():
    """Mocks SentenceTransformer to avoid loading actual heavy models."""
    with patch("app.services.evaluation.embedding_evaluator.SentenceTransformer") as MockClass:
        model_instance = MagicMock()
        
        # Mock encode to return a valid numpy array shape
        def side_effect(sentences, convert_to_tensor=False):
            if isinstance(sentences, str):
                return np.random.rand(384)
            return np.random.rand(len(sentences), 384)
        
        model_instance.encode.side_effect = side_effect
        MockClass.return_value = model_instance
        yield MockClass

@pytest.fixture
def mock_util():
    """Mocks sentence_transformers.util for similarity calculations."""
    with patch("app.services.evaluation.embedding_evaluator.util") as mock_util:
        yield mock_util

@pytest.fixture
def evaluator(mock_sentence_transformer):
    """Returns an evaluator instance with a mocked model."""
    return EmbeddingQualityEvaluator(model_name="mock-model")

# --- TESTS: INITIALIZATION ---

def test_initialization_success(mock_sentence_transformer):
    """Test successful initialization loads the model."""
    evaluator = EmbeddingQualityEvaluator(model_name="test-model")
    assert evaluator.similarity_model is not None
    mock_sentence_transformer.assert_called_with("test-model")

def test_initialization_failure():
    """Test initialization handles model load failure gracefully."""
    with patch("app.services.evaluation.embedding_evaluator.SentenceTransformer", side_effect=Exception("Load Error")):
        evaluator = EmbeddingQualityEvaluator(model_name="bad-model")
        assert evaluator.similarity_model is None
        # Should log warning but not crash initialization

# --- TESTS: VECTOR ANALYSIS ---

def test_analyze_embedding_vector(evaluator):
    """Test statistical analysis of a single embedding vector."""
    # Create a vector with known properties: [1, -1] -> Norm sqrt(2) approx 1.414
    vector = np.array([1.0, -1.0]) 
    stats = evaluator._analyze_embedding_vector(vector, prefix="test")
    
    assert stats["test_dimension"] == 2
    assert stats["test_mean"] == 0.0
    assert stats["test_max"] == 1.0
    assert stats["test_min"] == -1.0
    assert stats["test_norm"] == pytest.approx(1.414, 0.001)

# --- TESTS: SIMILARITY ANALYSIS ---

def test_analyze_similarities(evaluator):
    """
    Test cosine similarity distribution analysis.
    Verifies min/max/mean and classification counts.
    """
    query_emb = np.array([1.0, 0.0])
    doc_embs = [
        np.array([1.0, 0.0]),   # Sim 1.0 (High)
        np.array([0.0, 1.0]),   # Sim 0.0 (Low)
        np.array([-1.0, 0.0])   # Sim -1.0 (Low)
    ]
    
    metrics = evaluator._analyze_similarities(query_emb, doc_embs)
    
    assert metrics["similarity_max"] == pytest.approx(1.0)
    assert metrics["similarity_min"] == pytest.approx(-1.0)
    assert metrics["similarity_mean"] == pytest.approx(0.0)
    assert metrics["high_similarity_count"] == 1  # > 0.7
    assert metrics["low_similarity_count"] == 2   # < 0.5

def test_analyze_similarities_empty(evaluator):
    """Test similarity analysis with no documents."""
    query_emb = np.array([1.0, 0.0])
    metrics = evaluator._analyze_similarities(query_emb, [])
    assert metrics == {}

# --- TESTS: SEMANTIC COHERENCE ---

def test_analyze_semantic_coherence(evaluator, mock_util):
    """
    Test semantic coherence logic.
    Mocks `util.cos_sim` to return specific similarity matrices for
    Query-Doc and Doc-Doc comparisons.
    """
    query = "test query"
    doc_texts = ["doc1", "doc2"]
    
    # 1. Setup Mock Return Values
    # util.cos_sim returns a Tensor. code calls .cpu().numpy().
    
    # Mock result for Query vs Docs (Shape: [1, 2])
    # Values: 0.8 and 0.6. Mean = 0.7
    mock_qd_tensor = MagicMock()
    mock_qd_tensor.cpu.return_value.numpy.return_value = np.array([0.8, 0.6])
    
    # Mock result for Docs vs Docs (Shape: [2, 2])
    # Values: [[1.0, 0.5], [0.5, 1.0]]. Upper triangle (k=1) is [0.5]. Mean = 0.5
    mock_dd_tensor = MagicMock()
    mock_dd_tensor.cpu.return_value.numpy.return_value = np.array([[1.0, 0.5], [0.5, 1.0]])

    # We need to simulate the result of indexing [0] for the first call (query_doc_sims)
    # The code does: util.cos_sim(...)[0]
    mock_qd_result_wrapper = MagicMock()
    mock_qd_result_wrapper.__getitem__.side_effect = lambda idx: mock_qd_tensor if idx == 0 else MagicMock()
    
    # The code does: util.cos_sim(...) for the second call (doc_doc_sims) directly
    mock_dd_result_wrapper = mock_dd_tensor

    # 2. Apply side effects
    # First call: Query vs Docs. Second call: Docs vs Docs.
    mock_util.cos_sim.side_effect = [mock_qd_result_wrapper, mock_dd_result_wrapper]
    
    # 3. Execute
    metrics = evaluator._analyze_semantic_coherence(query, doc_texts)
    
    # 4. Assertions
    assert metrics["semantic_query_doc_mean"] == pytest.approx(0.7)
    assert metrics["semantic_inter_doc_mean"] == pytest.approx(0.5)
    assert metrics["semantic_coherence_score"] == pytest.approx(0.7)

def test_analyze_semantic_coherence_no_model(evaluator):
    """Test coherence skipped if no model loaded."""
    evaluator.similarity_model = None
    metrics = evaluator._analyze_semantic_coherence("q", ["d"])
    assert metrics == {}

# --- TESTS: MAIN EVALUATION FLOW ---

def test_evaluate_embeddings_full_flow(evaluator):
    """Test the main public method aggregates all metrics correctly."""
    query = "query"
    query_emb = np.array([1.0, 0.0])
    doc_embs = [np.array([0.0, 1.0])]
    doc_texts = ["text"]
    
    # Patch internal methods to return simplified data
    with patch.object(evaluator, '_analyze_embedding_vector', return_value={"norm": 1}) as mock_vec, \
         patch.object(evaluator, '_analyze_similarities', return_value={"sim": 0.5}) as mock_sim, \
         patch.object(evaluator, '_analyze_semantic_coherence', return_value={"coh": 0.8}) as mock_coh:
        
        metrics = evaluator.evaluate_embeddings(
            query=query,
            query_embedding=query_emb,
            doc_embeddings=doc_embs,
            doc_texts=doc_texts,
            embedding_time_ms=100
        )
        
        assert metrics["embedding_time_ms"] == 100
        assert metrics["norm"] == 1 # From vector analysis
        assert metrics["sim"] == 0.5 # From similarity analysis
        assert metrics["coh"] == 0.8 # From coherence analysis
        assert metrics["evaluation_method"] == "production_proxy"

# --- TESTS: MTEB OFFLINE (Mocked) ---

def test_evaluate_mteb_offline_success(evaluator):
    """
    Test offline benchmark runner wrapper.
    Mocks the 'mteb' package import and execution.
    """
    mock_mteb_module = MagicMock()
    mock_mteb_class = MagicMock()
    mock_mteb_module.MTEB = mock_mteb_class
    
    # Setup MTEB instance behavior
    mock_instance = mock_mteb_class.return_value
    mock_instance.run.return_value = {"score": 0.99}

    # Patch sys.modules to simulate 'mteb' being installed
    with patch.dict(sys.modules, {'mteb': mock_mteb_module}):
        results = evaluator.evaluate_mteb_offline("model_x", tasks=["Task1"])
        
        assert results["score"] == 0.99
        mock_mteb_class.assert_called_with(tasks=["Task1"])

def test_evaluate_mteb_offline_missing_package(evaluator):
    """
    Test handling when 'mteb' package is not installed.
    Uses builtins.__import__ patching to simulate ImportError.
    """
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == 'mteb':
            raise ImportError("No module named mteb")
        return real_import(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=mock_import):
        results = evaluator.evaluate_mteb_offline("model_x")
        
        assert "error" in results
        assert results["error"] == "mteb not installed"

# --- TESTS: UTILS ---

def test_convenience_function():
    """Test the standalone wrapper function."""
    with patch("app.services.evaluation.embedding_evaluator.EmbeddingQualityEvaluator") as MockEval:
        mock_inst = MockEval.return_value
        mock_inst.evaluate_embeddings.return_value = {"metric": 1}
        
        res = evaluate_embedding_quality("q")
        assert res["metric"] == 1

def test_get_mteb_reference_score():
    """Test retrieval of reference scores."""
    score = get_mteb_reference_score("bge-m3")
    assert score is not None
    assert "avg_score" in score
    
    missing = get_mteb_reference_score("unknown_model")
    assert missing is None