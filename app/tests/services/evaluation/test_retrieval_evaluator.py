import pytest
from typing import List, Dict, Any
import numpy as np

# Adjust import based on your actual structure
from app.services.evaluation.retrieval_evaluator import (
    RetrievalMetricsEvaluator,
    calculate_retrieval_metrics
)

# --- FIXTURES ---

@pytest.fixture
def evaluator():
    """Returns an initialized RetrievalMetricsEvaluator."""
    return RetrievalMetricsEvaluator()

@pytest.fixture
def retrieved_docs_standard():
    """Sample retrieved docs with standard metadata structure."""
    return [
        {'metadata': {'source': 'doc1'}, 'relevance_score': 0.9},
        {'metadata': {'source': 'doc2'}, 'relevance_score': 0.8},
        {'metadata': {'source': 'doc3'}, 'relevance_score': 0.4},
        {'metadata': {'source': 'doc4'}, 'relevance_score': 0.2},
    ]

# --- TESTS: GROUND TRUTH EVALUATION ---

def test_evaluate_with_ground_truth_full_metrics(evaluator, retrieved_docs_standard):
    """
    Test calculation of all standard IR metrics when ground truth is provided.
    Scenario:
        Retrieved: [doc1, doc2, doc3, doc4]
        Relevant:  [doc1, doc3]
    """
    ground_truth = ['doc1', 'doc3']
    k_values = [3]
    
    metrics = evaluator.evaluate_retrieval(
        query="test",
        retrieved_docs=retrieved_docs_standard,
        k_values=k_values,
        ground_truth_docs=ground_truth
    )
    
    # 1. General Checks
    assert metrics['total_retrieved'] == 4
    assert metrics['total_relevant'] == 2
    assert metrics['evaluation_method'] == 'ground_truth'
    
    # 2. Check Metrics at K=3
    # Top 3 Retrieved: [doc1, doc2, doc3]
    # Hits: doc1 (Rank 1), doc3 (Rank 3)
    
    # Hit Rate: Found at least one? Yes (doc1).
    assert metrics['hit_rate_at_3'] == 1.0
    
    # Precision: 2 hits / 3 retrieved = 0.666...
    assert metrics['precision_at_3'] == pytest.approx(0.666, 0.01)
    
    # Recall: 2 hits / 2 total relevant = 1.0
    assert metrics['recall_at_3'] == 1.0
    
    # MRR: First relevant doc is at Rank 1. MRR = 1/1 = 1.0
    assert metrics['mrr'] == 1.0
    
    # NDCG Check: 
    # Ideal: [doc1, doc3, doc2] (Relevant, Relevant, Irrelevant)
    # Actual: [doc1, doc2, doc3] (Relevant, Irrelevant, Relevant)
    # Actual DCG < Ideal DCG, so NDCG should be < 1.0 but > 0.0
    assert 0.0 < metrics['ndcg_at_3'] < 1.0

def test_calculate_mrr_logic(evaluator):
    """Test MRR (Mean Reciprocal Rank) specific logic."""
    # Case 1: First doc relevant -> MRR 1.0
    assert evaluator._calculate_mrr(['good', 'bad'], ['good']) == 1.0
    
    # Case 2: Second doc relevant -> MRR 0.5
    assert evaluator._calculate_mrr(['bad', 'good'], ['good']) == 0.5
    
    # Case 3: No relevant docs -> MRR 0.0
    assert evaluator._calculate_mrr(['bad', 'bad'], ['good']) == 0.0

def test_calculate_ndcg_logic(evaluator):
    """Test NDCG calculation details."""
    # Perfect Ranking
    retrieved_perfect = [{'metadata': {'source': 'A'}}, {'metadata': {'source': 'B'}}]
    ground_truth = ['A', 'B']
    assert evaluator._calculate_ndcg(retrieved_perfect, ground_truth) == 1.0

    # Terrible Ranking
    retrieved_bad = [{'metadata': {'source': 'X'}}, {'metadata': {'source': 'Y'}}]
    assert evaluator._calculate_ndcg(retrieved_bad, ground_truth) == 0.0

# --- TESTS: SCORE-BASED EVALUATION ---

def test_evaluate_score_based_standard_threshold(evaluator, retrieved_docs_standard):
    """
    Test proxy metrics using standard threshold (0.7).
    Max score is 0.9, so it treats scores > 0.7 as relevant.
    """
    # Docs: 0.9 (Rel), 0.8 (Rel), 0.4 (Irrel), 0.2 (Irrel)
    
    metrics = evaluator.evaluate_retrieval(
        query="test",
        retrieved_docs=retrieved_docs_standard,
        k_values=[3]
    )
    
    assert metrics['evaluation_method'] == 'score_based'
    
    # Estimated Precision@3: 2 relevant (0.9, 0.8) out of 3 = 0.666
    assert metrics['estimated_precision_at_3'] == pytest.approx(0.666, 0.01)
    
    # Estimated Hit Rate: Found high score? Yes.
    assert metrics['estimated_hit_rate_at_3'] == 1.0
    
    # Stats
    assert metrics['max_score'] == 0.9
    assert metrics['avg_score'] == pytest.approx(0.575) # (0.9+0.8+0.4+0.2)/4

def test_evaluate_score_based_adaptive_threshold(evaluator):
    """
    Test adaptive threshold logic for low scores (e.g., RRF).
    If max score < 0.2, threshold becomes max_score * 0.7.
    """
    # Max score 0.1. Threshold = 0.07.
    docs = [
        {'rrf_score': 0.1},   # 0.1 >= 0.07 -> Relevant
        {'rrf_score': 0.08},  # 0.08 >= 0.07 -> Relevant
        {'rrf_score': 0.05}   # 0.05 < 0.07 -> Irrelevant
    ]
    
    metrics = evaluator.evaluate_retrieval(
        query="test",
        retrieved_docs=docs,
        k_values=[3]
    )
    
    # Precision@3: 2/3 relevant
    assert metrics['estimated_precision_at_3'] == pytest.approx(0.666, 0.01)
    assert metrics['max_score'] == 0.1

def test_extract_scores_priority(evaluator):
    """
    Test that the evaluator picks the correct score field based on priority.
    Priority: rrf > rerank > relevance > dense > sparse
    """
    docs = [
        {'rrf_score': 10, 'rerank_score': 5},        # Should pick 10 (rrf)
        {'rerank_score': 5, 'relevance_score': 2},   # Should pick 5 (rerank)
        {'sparse_score': 1}                          # Should pick 1 (sparse)
    ]
    
    # We check 'max_score' and 'avg_score' to infer which were picked
    metrics = evaluator._evaluate_score_based(docs, [1])
    
    # Scores extracted: [10, 5, 1]
    assert metrics['max_score'] == 10
    assert metrics['avg_score'] == pytest.approx(5.333, 0.01) # (10+5+1)/3

# --- TESTS: EDGE CASES ---

def test_evaluate_empty_input(evaluator):
    """Test handling of empty retrieval lists."""
    metrics = evaluator.evaluate_retrieval("q", [], [3])
    
    assert metrics['total_retrieved'] == 0
    assert metrics['hit_rate_at_3'] == 0.0
    assert metrics['evaluation_method'] == 'none'
    assert metrics['mrr'] == 0.0

def test_missing_metadata(evaluator):
    """Test robustness when 'metadata' or 'source' is missing."""
    docs = [{'relevance_score': 0.9}] # No metadata
    ground_truth = ['doc1']
    
    # Should not crash, just fail to match ID
    metrics = evaluator.evaluate_retrieval("q", docs, [1], ground_truth)
    assert metrics['hit_rate_at_1'] == 0.0

def test_convenience_function():
    """Test the functional wrapper."""
    res = calculate_retrieval_metrics("q", [], [1])
    assert res['total_retrieved'] == 0