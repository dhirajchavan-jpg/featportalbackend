import pytest
from unittest.mock import AsyncMock, MagicMock, patch, ANY  # <--- Added ANY here
from typing import Dict, Any

# Import the module under test
from app.services.evaluation.pipeline_evaluator import (
    ComprehensivePipelineEvaluator,
    evaluate_complete_rag_pipeline
)

# --- Fixtures ---

@pytest.fixture
def mock_sub_evaluators():
    """
    Mock all the sub-evaluator classes so we don't need real models/APIs.
    """
    with patch("app.services.evaluation.pipeline_evaluator.QueryProcessingEvaluator") as MockQuery, \
         patch("app.services.evaluation.pipeline_evaluator.EmbeddingQualityEvaluator") as MockEmbed, \
         patch("app.services.evaluation.pipeline_evaluator.RetrievalMetricsEvaluator") as MockRetrieval, \
         patch("app.services.evaluation.pipeline_evaluator.RerankingEffectivenessEvaluator") as MockRerank, \
         patch("app.services.evaluation.pipeline_evaluator.ContextQualityEvaluator") as MockContext, \
         patch("app.services.evaluation.pipeline_evaluator.GeminiJudgeEvaluator") as MockGenJudge, \
         patch("app.services.evaluation.pipeline_evaluator.GeminiRetrievalEvaluator") as MockRetJudge:
        
        # Setup return values for evaluation methods
        
        # 1. Query Evaluator
        query_inst = MockQuery.return_value
        query_inst.evaluate_query_processing.return_value = {"expansion_quality_score": 0.8}
        
        # 2. Embedding Evaluator
        embed_inst = MockEmbed.return_value
        embed_inst.evaluate_embeddings.return_value = {"semantic_coherence_score": 0.9}
        
        # 3. Retrieval Evaluator
        ret_inst = MockRetrieval.return_value
        ret_inst.evaluate_retrieval.return_value = {"estimated_hit_rate_at_3": 0.7, "max_score": 1.0}
        
        # 4. Reranking Evaluator
        rerank_inst = MockRerank.return_value
        rerank_inst.evaluate_reranking.return_value = {"reranking_effectiveness_score": 0.6, "top_3_changed": True}
        
        # 5. Context Evaluator
        ctx_inst = MockContext.return_value
        ctx_inst.evaluate_context.return_value = {"context_quality_score": 0.85, "coverage_score": 0.9}
        
        # 6. Generation Evaluator (Async)
        gen_inst = MockGenJudge.return_value
        gen_inst.evaluate_response = AsyncMock(return_value={
            "overall_score": 90, 
            "faithfulness_score": 95,
            "relevance_score": 90,
            "toxicity_score": 0
        })
        
        yield {
            "query": query_inst,
            "embed": embed_inst,
            "retrieval": ret_inst,
            "rerank": rerank_inst,
            "context": ctx_inst,
            "gen": gen_inst
        }

@pytest.fixture
def evaluator(mock_sub_evaluators):
    """Returns an initialized ComprehensivePipelineEvaluator."""
    return ComprehensivePipelineEvaluator()

@pytest.fixture
def mock_otel():
    """Mock OpenTelemetry tracing."""
    # We mock the 'tracer' object imported in the module
    with patch("app.services.evaluation.pipeline_evaluator.tracer") as mock_tracer:
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        yield mock_tracer, mock_span

# --- Tests ---

def test_initialization(mock_sub_evaluators):
    """Test that sub-evaluators are initialized correctly based on flags."""
    # Default: All True
    evaluator = ComprehensivePipelineEvaluator()
    assert evaluator.query_evaluator is not None
    assert evaluator.generation_evaluator is not None
    
    # Selective Initialization
    partial_eval = ComprehensivePipelineEvaluator(evaluate_query=False, evaluate_generation=False)
    assert partial_eval.query_evaluator is None
    assert partial_eval.generation_evaluator is None
    assert partial_eval.context_evaluator is not None # Default True

def test_get_active_stages(evaluator):
    """Test report of active stages."""
    stages = evaluator._get_active_stages()
    expected = ["query_processing", "embedding", "retrieval", "reranking", "context", "generation"]
    assert sorted(stages) == sorted(expected)

@pytest.mark.asyncio
async def test_evaluate_complete_pipeline_full_flow(evaluator, mock_sub_evaluators, mock_otel):
    """
    Test the main evaluation orchestration method.
    Verifies that data flows to sub-evaluators and metrics are aggregated.
    """
    mock_tracer, mock_span = mock_otel
    
    # Input Data
    inputs = {
        "original_query": "What is RAG?",
        "expanded_query": "What is Retrieval Augmented Generation?",
        "query_embedding": [0.1, 0.2],
        "docs_before_rerank": [{"content": "Doc A"}],
        "docs_after_rerank": [{"content": "Doc A (Ranked)"}],
        "assembled_context": "Context: Doc A",
        "generated_response": "RAG is cool.",
        "user_id": "u1",
        "model_config": {"search_strategy": "hybrid"}
    }
    
    # Run Evaluation
    results = await evaluator.evaluate_complete_pipeline(**inputs)
    
    # Assertions on Results
    assert results["query"] == "What is RAG?"
    assert "query_processing" in results
    assert "embedding" in results
    assert "retrieval" in results
    assert "reranking" in results
    assert "context" in results
    assert "generation" in results
    
    # Check Pipeline Health Score
    assert results["pipeline_health_score"] > 0.0
    assert results["pipeline_health_score"] <= 1.0
    
    # Verify Metadata
    assert results["model_config"]["search_strategy"] == "hybrid"
    
    # Verify Sub-Evaluator Calls
    mock_sub_evaluators["query"].evaluate_query_processing.assert_called_once()
    mock_sub_evaluators["gen"].evaluate_response.assert_called_once()
    
    # Verify Phoenix Tracing
    # FIX: Use ANY from unittest.mock, not pytest.any
    mock_tracer.start_as_current_span.assert_called_with("comprehensive_evaluation", attributes=ANY)
    
    # Check if attributes were set on the span
    mock_span.set_attribute.assert_any_call("evaluation.generation.overall", 90.0)

@pytest.mark.asyncio
async def test_evaluate_pipeline_missing_stages(evaluator):
    """Test that evaluator skips stages where input data is missing."""
    # Only provide query and generation input
    inputs = {
        "original_query": "Q",
        "generated_response": "A"
        # Missing expanded_query, embeddings, docs, context
    }
    
    results = await evaluator.evaluate_complete_pipeline(**inputs)
    
    # Should perform Generation eval, but skip others
    assert "generation" in results
    assert "query_processing" not in results
    assert "retrieval" not in results
    
    # Health score should still be calculated
    assert "pipeline_health_score" in results

def test_calculate_pipeline_health_score(evaluator):
    """Test the weighted health score calculation."""
    # Case 1: Perfect Scores
    metrics_perfect = {
        'query_processing': {'expansion_quality_score': 1.0},
        'retrieval': {'ndcg_at_3': 1.0},
        'context': {'context_quality_score': 1.0},
        'generation': {'overall_score': 100, 'toxicity_score': 0}
    }
    score = evaluator._calculate_pipeline_health(metrics_perfect)
    assert score == pytest.approx(1.0, 0.01)
    
    # Case 2: Critical Failure (Toxic)
    metrics_toxic = {
        'generation': {'overall_score': 100, 'toxicity_score': 90} # High toxicity
    }
    score_toxic = evaluator._calculate_pipeline_health(metrics_toxic)
    assert score_toxic <= 0.4 # Should be capped
    
    # Case 3: Poor Retrieval
    metrics_bad_ret = {
        'retrieval': {'estimated_hit_rate_at_3': 0.1}, # < 0.2 threshold
        'generation': {'overall_score': 90, 'toxicity_score': 0}
    }
    score_bad = evaluator._calculate_pipeline_health(metrics_bad_ret)
    assert score_bad <= 0.4

def test_extract_stage_scores(evaluator):
    """Test summary extraction helper."""
    metrics = {
        'query_processing': {'expansion_quality_score': 0.5},
        'generation': {'overall_score': 80}
    }
    
    summary = evaluator._extract_stage_scores(metrics)
    
    assert summary['query'] == 0.5
    assert summary['generation'] == 0.8 # Normalized
    assert 'retrieval' not in summary

@pytest.mark.asyncio
async def test_convenience_function():
    """Test the evaluate_complete_rag_pipeline wrapper."""
    with patch("app.services.evaluation.pipeline_evaluator.ComprehensivePipelineEvaluator") as MockClass:
        instance = MockClass.return_value
        instance.evaluate_complete_pipeline = AsyncMock(return_value={"status": "ok"})
        
        result = await evaluate_complete_rag_pipeline("query")
        
        assert result["status"] == "ok"
        instance.evaluate_complete_pipeline.assert_called_once()

@pytest.mark.asyncio
async def test_phoenix_logging_failure_handling(evaluator, mock_sub_evaluators, mock_otel):
    """Test that evaluation continues even if Phoenix logging fails."""
    mock_tracer, _ = mock_otel
    # Simulate tracing error
    mock_tracer.start_as_current_span.side_effect = Exception("Phoenix Down")
    
    # Should NOT raise exception
    results = await evaluator.evaluate_complete_pipeline(
        original_query="Q",
        generated_response="A"
    )
    
    assert results is not None
    assert "generation" in results