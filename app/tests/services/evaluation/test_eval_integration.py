import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys

# Import the module under test
from app.services.evaluation.eval_integration import (
    LiveEvaluationWrapper,
    evaluate_rag_query_live,
    get_live_eval_stats,
    get_evaluation_history,
    get_live_evaluator
)

# --- Fixtures ---

@pytest.fixture
def mock_db():
    """Mocks the database client used in eval_integration."""
    # We patch the db object where it is defined/imported
    with patch("app.services.evaluation.eval_integration.db") as mock_db_instance:
        mock_collection = MagicMock()
        # Allow dictionary access: db["collection_name"]
        mock_db_instance.__getitem__.return_value = mock_collection
        
        # Setup AsyncMock for common collection methods
        mock_collection.insert_one = AsyncMock()
        mock_collection.find = MagicMock()
        mock_collection.aggregate = MagicMock()
        
        yield mock_db_instance, mock_collection

@pytest.fixture
def wrapper(mock_db):
    """
    Returns a LiveEvaluationWrapper instance with mocked judges.
    """
    _, mock_collection = mock_db

    # Patch the GETTER functions in their original module
    with patch("app.services.evaluation.gemini_judge_evaluator.get_gemini_judge_evaluator") as MockGetGemini, \
         patch("app.services.evaluation.gemini_judge_evaluator.get_retrieval_judge_evaluator") as MockGetRetrieval:
        
        # 1. Setup Gemini Judge Mock
        mock_gemini_instance = AsyncMock()
        mock_gemini_instance.evaluate_response.return_value = {
            "faithfulness_score": 0.9,
            "relevance_score": 0.8,
            "completeness_score": 0.85,
            "hallucination_score": 0.1,
            "toxicity_score": 0.0,
            "overall_score": 0.88,
            "faithfulness_explanation": "Good"
        }
        MockGetGemini.return_value = mock_gemini_instance

        # 2. Setup Retrieval Judge Mock
        mock_retrieval_instance = AsyncMock()
        mock_retrieval_instance.evaluate_retrieved_chunks.return_value = {
            "overall_score": 0.75,
            "num_relevant_chunks": 3
        }
        MockGetRetrieval.return_value = mock_retrieval_instance

        # 3. Instantiate Wrapper
        w = LiveEvaluationWrapper(sample_rate=1.0)
        yield w

# --- Tests ---

@pytest.mark.asyncio
async def test_evaluate_query_result_success(wrapper, mock_db):
    """Test full evaluation flow with valid inputs."""
    _, mock_collection = mock_db
    
    rag_result = {
        "result": "This is the RAG response.",
        "source_documents": [{"content": "chunk 1"}, {"content": "chunk 2"}],
        "model_used": "gpt-4",
        "processing_time": 1.2
    }

    # Force random to allow sampling
    with patch("random.random", return_value=0.1):
        result = await wrapper.evaluate_query_result(
            query="test query",
            rag_result=rag_result,
            user_id="user_1",
            project_id="proj_1"
        )

    # Assertions
    assert result is not None
    assert result["overall_response_score"] == 0.88
    
    # Verify DB insertion
    mock_collection.insert_one.assert_called_once()
    inserted_doc = mock_collection.insert_one.call_args[0][0]
    assert inserted_doc["user_id"] == "user_1"
    assert inserted_doc["evaluation_method"] == "gemini_judge"

@pytest.mark.asyncio
async def test_evaluate_query_result_sampling_skip(wrapper):
    """Test that query is skipped if random > sample_rate."""
    wrapper.sample_rate = 0.5
    
    with patch("random.random", return_value=0.8):
        result = await wrapper.evaluate_query_result("q", {}, "u", "p")
    
    assert result is None

@pytest.mark.asyncio
async def test_evaluate_query_result_no_docs(wrapper, mock_db):
    """Test evaluation behavior when source documents are missing."""
    _, mock_collection = mock_db
    rag_result = {"result": "Response", "source_documents": []}
    
    with patch("random.random", return_value=0.1):
        result = await wrapper.evaluate_query_result("q", rag_result, "u", "p")
    
    # Expect None (skipped)
    assert result is None
    mock_collection.insert_one.assert_not_called()

@pytest.mark.asyncio
async def test_evaluate_query_result_with_otel_tracing(wrapper):
    """Test interaction with OpenTelemetry tracing."""
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    
    # FIX: Patch 'opentelemetry.trace' globally because it is imported inside the function
    with patch("opentelemetry.trace.get_tracer", return_value=mock_tracer):
        
        rag_result = {"result": "Res", "source_documents": [{"c": "1"}]}
        
        with patch("random.random", return_value=0.1):
            await wrapper.evaluate_query_result(
                query="q", 
                rag_result=rag_result, 
                user_id="u", 
                project_id="p", 
                span_context=MagicMock()
            )
            
    # Verify attributes set
    mock_span.set_attribute.assert_any_call("eval.faithfulness_score", 0.9)

@pytest.mark.asyncio
async def test_evaluate_query_result_exception_handling(wrapper):
    """Test robust error handling."""
    # Force the judge to raise an exception
    wrapper.response_judge.evaluate_response.side_effect = Exception("Judge Error")
    
    rag_result = {"result": "A", "source_documents": [{"c": "1"}]}
    
    with patch("random.random", return_value=0.1):
        result = await wrapper.evaluate_query_result("q", rag_result, "u", "p")
    
    # Should return None and log error, not crash
    assert result is None

@pytest.mark.asyncio
async def test_get_live_eval_stats(wrapper, mock_db):
    """Test aggregation of stats from DB."""
    _, mock_collection = mock_db
    
    mock_agg_result = {
        "total_evaluated": 50,
        "avg_overall_score": 0.92
    }
    
    # Setup Aggregate Cursor logic
    mock_cursor = AsyncMock()
    mock_cursor.to_list.return_value = [mock_agg_result]
    mock_collection.aggregate.return_value = mock_cursor
    
    # Patch the singleton getter
    with patch("app.services.evaluation.eval_integration.LiveEvaluationWrapper", return_value=wrapper):
        stats = await get_live_eval_stats(hours=24)
    
    assert stats["total_evaluated"] == 50
    assert stats["avg_overall_score"] == 0.92
    mock_collection.aggregate.assert_called_once()

@pytest.mark.asyncio
async def test_get_live_eval_stats_empty(wrapper, mock_db):
    """Test stats when no data found."""
    _, mock_collection = mock_db
    
    mock_cursor = AsyncMock()
    mock_cursor.to_list.return_value = []
    mock_collection.aggregate.return_value = mock_cursor
    
    with patch("app.services.evaluation.eval_integration.LiveEvaluationWrapper", return_value=wrapper):
        stats = await get_live_eval_stats()
    
    assert stats["total_evaluated"] == 0
    assert "message" in stats

@pytest.mark.asyncio
async def test_evaluate_rag_query_live_convenience_function():
    """Test the standalone wrapper function."""
    with patch("app.services.evaluation.eval_integration.LiveEvaluationWrapper") as MockClass:
        instance = MockClass.return_value
        instance.evaluate_query_result = AsyncMock(return_value="Success")
        
        result = await evaluate_rag_query_live("q", {}, "u", "p")
        
        assert result == "Success"
        instance.evaluate_query_result.assert_called_once()

@pytest.mark.asyncio
async def test_get_evaluation_history(mock_db):
    """Test history retrieval endpoint logic."""
    _, mock_collection = mock_db
    
    mock_cursor = MagicMock()
    mock_collection.find.return_value = mock_cursor
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.limit.return_value = mock_cursor
    mock_cursor.to_list = AsyncMock(return_value=[{"_id": "1", "score": 0.9}])
    
    history = await get_evaluation_history(limit=5)
    
    assert len(history) == 1
    mock_collection.find.assert_called()

def test_singleton_pattern():
    """Verify get_live_evaluator returns a singleton."""
    import app.services.evaluation.eval_integration as ei
    if hasattr(ei.get_live_evaluator, "_instance"):
        del ei.get_live_evaluator._instance

    with patch("app.services.evaluation.gemini_judge_evaluator.get_gemini_judge_evaluator"), \
         patch("app.services.evaluation.gemini_judge_evaluator.get_retrieval_judge_evaluator"):
        
        eval1 = get_live_evaluator()
        eval2 = get_live_evaluator()
        
        assert eval1 is eval2
        assert isinstance(eval1, LiveEvaluationWrapper)