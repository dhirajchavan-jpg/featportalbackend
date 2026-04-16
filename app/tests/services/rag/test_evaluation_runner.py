import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

# Import the module under test
from app.services.rag.evaluation_runner import (
    _run_comprehensive_evaluation_background,
    _run_evaluation_sync,
    get_comprehensive_evaluator
)

# --- Fixtures ---

@pytest.fixture
def mock_evaluator():
    """Mocks the ComprehensivePipelineEvaluator class and instance."""
    with patch("app.services.rag.evaluation_runner.ComprehensivePipelineEvaluator") as MockClass:
        instance = MockClass.return_value
        # Mock the async evaluate method
        instance.evaluate_complete_pipeline = AsyncMock(return_value={
            "pipeline_health_score": 0.95,
            "stage_scores_summary": {"retrieval": 0.9},
            "generation": {"hallucination_score": 0.05} # Low hallucination
        })
        yield instance

@pytest.fixture
def mock_db_collection():
    """Mocks the MongoDB collection insertion."""
    with patch("app.database.db") as mock_db:
        mock_collection = MagicMock()
        mock_collection.insert_one = AsyncMock()
        # Allow dictionary-style access: db['collection_name']
        mock_db.__getitem__.return_value = mock_collection
        yield mock_collection

@pytest.fixture
def mock_cache_service():
    """Mocks the _cache_result function."""
    with patch("app.services.rag.evaluation_runner._cache_result", new_callable=AsyncMock) as mock_cache:
        yield mock_cache

# --- Tests ---

def test_get_comprehensive_evaluator_singleton():
    """Test that the evaluator uses the Singleton pattern."""
    # Reset singleton manually for test isolation
    import app.services.rag.evaluation_runner as runner
    runner._comprehensive_evaluator = None
    
    with patch("app.services.rag.evaluation_runner.ComprehensivePipelineEvaluator") as MockClass:
        e1 = get_comprehensive_evaluator()
        e2 = get_comprehensive_evaluator()
        
        assert e1 is e2
        assert MockClass.call_count == 1

def test_run_evaluation_sync_logic(mock_evaluator):
    """
    Test the synchronous wrapper.
    It should create a new event loop and run the async evaluation method.
    """
    eval_data = {
        "original_query": "test query",
        "_trace_context": "trace_123" 
    }
    
    # Patch asyncio internals to verify loop creation without side effects
    with patch("asyncio.new_event_loop") as mock_new_loop, \
         patch("asyncio.set_event_loop"): 
        
        mock_loop_instance = MagicMock()
        mock_new_loop.return_value = mock_loop_instance
        
        # Configure run_until_complete to return a dummy dict
        mock_loop_instance.run_until_complete.return_value = {"score": 100}
        
        result = _run_evaluation_sync(eval_data)
        
        # Verify it ran the evaluation
        mock_loop_instance.run_until_complete.assert_called_once()
        
        # Verify Context Restoration
        assert result["_trace_context"] == "trace_123"
        assert result["score"] == 100
        
        # Verify Loop Cleanup
        mock_loop_instance.close.assert_called_once()

@pytest.mark.asyncio
async def test_run_background_task_execution_flow(mock_evaluator, mock_db_collection):
    """
    Test the main background runner.
    It should offload to executor, wait for result, and save to DB.
    """
    eval_data = {
        "original_query": "What is compliance?",
        "user_id": "user_1",
        "project_id": "proj_A"
    }
    
    # Mock the synchronous wrapper call to return metrics immediately
    with patch("app.services.rag.evaluation_runner._run_evaluation_sync") as mock_sync_wrapper:
        mock_sync_wrapper.return_value = {
            "pipeline_health_score": 0.8,
            "generated_response": "AI Answer",
            "generation": {"hallucination_score": 0.5} 
        }
        
        await _run_comprehensive_evaluation_background(eval_data)
        
        # 1. Verify it ran in executor (mocked wrapper called)
        mock_sync_wrapper.assert_called_once_with(eval_data)
        
        # 2. Verify DB Insertion
        mock_db_collection.insert_one.assert_called_once()
        
        # 3. Verify Metadata Injection
        call_args = mock_db_collection.insert_one.call_args[0][0]
        assert call_args["user_query"] == "What is compliance?"
        assert call_args["user_id"] == "user_1"
        assert call_args["project_id"] == "proj_A"
        assert "timestamp" in call_args

@pytest.mark.asyncio
async def test_caching_approved(mock_db_collection, mock_cache_service):
    """
    Test that caching IS called when hallucination score is LOW.
    """
    eval_data = {
        "original_query": "q",
        "user_id": "u1",
        "cache_key": "valid_key_123",
        "hallucination_threshold": 10.0 # Strict threshold
    }

    # Mock evaluation returning score 5% (Approved)
    metrics = {
        "generation": {"hallucination_score": 0.05} # 5%
    }
    
    with patch("app.services.rag.evaluation_runner._run_evaluation_sync", return_value=metrics):
        await _run_comprehensive_evaluation_background(eval_data)

    # Assert Caching Called
    mock_cache_service.assert_called_once()
    
    # Check arguments
    call_kwargs = mock_cache_service.call_args[1]
    assert call_kwargs["cache_key"] == "valid_key_123"
    assert call_kwargs["user_id"] == "u1"

@pytest.mark.asyncio
async def test_caching_rejected(mock_db_collection, mock_cache_service):
    """
    Test that caching is BLOCKED when hallucination score is HIGH.
    """
    eval_data = {
        "original_query": "q",
        "cache_key": "valid_key_123",
        "hallucination_threshold": 10.0 
    }

    # Mock evaluation returning score 50% (Rejected)
    metrics = {
        "generation": {"hallucination_score": 0.5} # 50% > 10%
    }
    
    with patch("app.services.rag.evaluation_runner._run_evaluation_sync", return_value=metrics):
        await _run_comprehensive_evaluation_background(eval_data)

    # Assert Caching NOT Called
    mock_cache_service.assert_not_called()

@pytest.mark.asyncio
async def test_background_task_error_handling(mock_db_collection):
    """Test that errors in the background task don't crash the app."""
    # Simulate an error in the synchronous wrapper execution
    with patch("app.services.rag.evaluation_runner._run_evaluation_sync", side_effect=Exception("Eval Failed")):
        
        # Should execute without raising exception to the caller (it just logs)
        await _run_comprehensive_evaluation_background({"q": "test"})
        
        # DB Insert should NOT happen
        mock_db_collection.insert_one.assert_not_called()