import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

# Adjust import based on your actual file structure
from app.services.evaluation.reranking_evaluator import (
    RerankingEffectivenessEvaluator,
    evaluate_reranking_effectiveness
)

# --- FIXTURES ---

@pytest.fixture
def evaluator():
    """Returns an initialized RerankingEffectivenessEvaluator."""
    return RerankingEffectivenessEvaluator()

@pytest.fixture
def sample_docs_before():
    """Returns a list of sample documents before reranking."""
    return [
        {'id': '1', 'metadata': {'file_name': 'doc1.pdf'}, 'relevance_score': 0.8},
        {'id': '2', 'metadata': {'file_name': 'doc2.pdf'}, 'relevance_score': 0.7},
        {'id': '3', 'metadata': {'file_name': 'doc3.pdf'}, 'relevance_score': 0.6},
        {'id': '4', 'metadata': {'file_name': 'doc4.pdf'}, 'relevance_score': 0.5},
    ]

@pytest.fixture
def sample_docs_after():
    """Returns a list of sample documents after reranking (reordered)."""
    return [
        {'id': '3', 'metadata': {'file_name': 'doc3.pdf'}, 'rerank_score': 0.95}, # Moved to top
        {'id': '1', 'metadata': {'file_name': 'doc1.pdf'}, 'rerank_score': 0.90},
        {'id': '2', 'metadata': {'file_name': 'doc2.pdf'}, 'rerank_score': 0.80},
        {'id': '4', 'metadata': {'file_name': 'doc4.pdf'}, 'rerank_score': 0.75},
    ]

# --- TESTS: METRIC CALCULATION ---

def test_initialization(evaluator):
    """Test successful initialization."""
    assert evaluator.name == "reranking_effectiveness"

def test_evaluate_reranking_metrics_calculation(evaluator, sample_docs_before, sample_docs_after):
    """
    Test full metric calculation for a standard reranking scenario.
    Checks Score Changes, Position Changes, and Counts.
    """
    metrics = evaluator.evaluate_reranking(
        docs_before_rerank=sample_docs_before,
        docs_after_rerank=sample_docs_after,
        reranking_time_ms=50.0
    )
    
    # 1. Basic Metadata
    assert metrics['reranking_time_ms'] == 50.0
    assert metrics['num_docs_before'] == 4
    assert metrics['num_docs_after'] == 4
    
    # 2. Score Improvements
    # Doc 3: 0.6 -> 0.95 (+0.35)
    # Doc 1: 0.8 -> 0.90 (+0.10)
    # Doc 2: 0.7 -> 0.80 (+0.10)
    # Doc 4: 0.5 -> 0.75 (+0.25)
    # All improved
    assert metrics['docs_improved'] == 4
    assert metrics['docs_degraded'] == 0
    assert metrics['score_improvement_mean'] > 0
    
    # 3. Position Changes
    # Doc 3 moved from pos 2 (0-indexed) to pos 0. Change: 2 - 0 = +2 (Moved Up)
    assert metrics['top_5_moved_up'] >= 1
    
    # 4. Top 3 Set Change
    # Before={1,2,3}, After={3,1,2}. The SET of IDs is identical, just order changed.
    # So top_3_changed should be False (set equality)
    assert metrics['top_3_changed'] is False

def test_evaluate_reranking_empty_input(evaluator):
    """Test handling of empty document lists."""
    metrics = evaluator.evaluate_reranking([], [], 10.0)
    
    # Should return empty metrics dict
    assert metrics['reranking_effectiveness_score'] == 0.0
    assert metrics['num_docs_before'] == 0

# --- TESTS: SCORE EXTRACTION ---

def test_extract_scores_logic(evaluator):
    """
    Test extraction of scores from diverse document formats.
    Verifies priority: relevance > dense > sparse
    """
    docs = [
        {'rerank_score': 0.9},        # After format
        {'relevance_score': 0.8},     # Before format
        {'dense_score': 0.7},         # Alternative Before
        {'sparse_score': 0.6},        # Alternative Before
        {}                            # Missing score
    ]
    
    # Test 'after' mode (looks for rerank_score)
    scores_after = evaluator._extract_scores(docs, 'after')
    assert scores_after == [0.9, 0.0, 0.0, 0.0, 0.0]
    
    # Test 'before' mode (looks for relevance/dense/sparse)
    scores_before = evaluator._extract_scores(docs, 'before')
    assert scores_before == [0.0, 0.8, 0.7, 0.6, 0.0]

# --- TESTS: RANKING CORRELATION ---

def test_analyze_ranking_changes_perfect_correlation(evaluator):
    """Test ranking metrics when order is unchanged (Spearman = 1.0)."""
    docs = [{'id': '1'}, {'id': '2'}]
    
    metrics = evaluator._analyze_ranking_changes(docs, docs)
    
    assert metrics['ranking_spearman_correlation'] == pytest.approx(1.0)
    assert metrics['ranking_stability'] == pytest.approx(1.0)

def test_analyze_ranking_changes_inverse_order(evaluator):
    """Test ranking metrics when order is completely reversed (Spearman = -1.0)."""
    docs_before = [{'id': '1'}, {'id': '2'}, {'id': '3'}]
    docs_after = [{'id': '3'}, {'id': '2'}, {'id': '1'}]
    
    metrics = evaluator._analyze_ranking_changes(docs_before, docs_after)
    
    # Spearman correlation for reversed list is -1.0
    assert metrics['ranking_spearman_correlation'] == pytest.approx(-1.0)

def test_analyze_ranking_changes_no_overlap(evaluator):
    """Test correlation when lists have no common documents."""
    docs_before = [{'id': '1'}]
    docs_after = [{'id': '2'}]
    
    metrics = evaluator._analyze_ranking_changes(docs_before, docs_after)
    assert metrics['ranking_correlation'] == 0.0

# --- TESTS: POSITION CHANGES ---

def test_analyze_position_changes_logic(evaluator):
    """
    Test specific position tracking math.
    Scenario:
        Before: A, B, C
        After:  C, A, B
    """
    docs_before = [{'id': 'A'}, {'id': 'B'}, {'id': 'C'}]
    docs_after = [{'id': 'C'}, {'id': 'A'}, {'id': 'B'}]
    
    metrics = evaluator._analyze_position_changes(docs_before, docs_after)
    
    # C moved from index 2 to 0 (+2) -> Up
    # A moved from index 0 to 1 (-1) -> Down
    # B moved from index 1 to 2 (-1) -> Down
    
    assert metrics['top_5_moved_up'] == 1   # C
    assert metrics['top_5_moved_down'] == 2 # A, B
    assert metrics['top_5_position_changes_max'] == 2

# --- TESTS: OVERALL EFFECTIVENESS SCORE ---

def test_calculate_overall_effectiveness_score(evaluator):
    """
    Test weighted score aggregation logic.
    Formula uses: improvement, docs_improved, and max_score.
    """
    # Setup metrics for calculation
    input_metrics = {
        'score_improvement_mean': 0.2,   # (0.2+0.2)/0.4 = 1.0 * 0.4 weight = 0.4
        'docs_improved': 10,
        'num_docs_after': 10,            # Ratio 1.0 * 0.3 weight = 0.3
        'score_after_max': 0.9           # 0.9 * 0.3 weight = 0.27
    }
    
    # Expected Total: 0.4 + 0.3 + 0.27 = 0.97
    score = evaluator._calculate_overall_effectiveness(input_metrics)
    assert score == pytest.approx(0.97)

def test_calculate_overall_effectiveness_neutral(evaluator):
    """Test effectiveness score defaults to 0.5 when metrics are missing."""
    score = evaluator._calculate_overall_effectiveness({})
    assert score == 0.5

# --- TESTS: ID EXTRACTION & UTILS ---

def test_get_doc_id_extraction(evaluator):
    """Test robustness of ID extraction from various metadata locations."""
    # Priority: metadata.source > metadata.file_name > id > hash
    
    doc1 = {'metadata': {'source': 's1'}}
    assert evaluator._get_doc_id(doc1) == 's1'
    
    doc2 = {'metadata': {'file_name': 'f1'}}
    assert evaluator._get_doc_id(doc2) == 'f1'
    
    doc3 = {'id': 'id1'}
    assert evaluator._get_doc_id(doc3) == 'id1'
    
    doc4 = {'page_content': 'text content'}
    # Should return a string hash
    assert isinstance(evaluator._get_doc_id(doc4), str)
    assert len(evaluator._get_doc_id(doc4)) > 0

def test_convenience_function():
    """Test the standalone wrapper function calls the evaluator correctly."""
    with patch("app.services.evaluation.reranking_evaluator.RerankingEffectivenessEvaluator") as MockClass:
        instance = MockClass.return_value
        instance.evaluate_reranking.return_value = {"metric": 1}
        
        result = evaluate_reranking_effectiveness([], [], 10)
        
        assert result["metric"] == 1
        MockClass.assert_called()