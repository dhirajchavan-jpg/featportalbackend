import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
import json

# Import the module under test
from app.services.rag.pipeline_orchestrator import (
    _normalize_response,
    query_rag_pipeline
)
from app.dependencies import UserPayload

# --- Fixtures ---

@pytest.fixture
def mock_user():
    return UserPayload(user_id="test_user", email="test@example.com", role="user")

@pytest.fixture
def mock_dependencies():
    """Mocks all external services used by the orchestrator."""
    with patch("app.services.rag.pipeline_orchestrator.cache_collection") as m_cache, \
         patch("app.services.rag.pipeline_orchestrator._get_chat_history") as m_hist, \
         patch("app.services.rag.pipeline_orchestrator.get_query_processor") as m_qp, \
         patch("app.services.rag.pipeline_orchestrator.get_model_router") as m_router, \
         patch("app.services.rag.pipeline_orchestrator.get_llm") as m_llm, \
         patch("app.services.rag.pipeline_orchestrator._retrieve_by_sector_comparative") as m_retr, \
         patch("app.services.rag.pipeline_orchestrator.get_context_manager") as m_cm, \
         patch("app.services.rag.pipeline_orchestrator._save_to_history") as m_save, \
         patch("app.services.rag.pipeline_orchestrator._cache_result") as m_cache_res, \
         patch("app.services.rag.pipeline_orchestrator.get_available_sectors") as m_sectors, \
         patch("app.services.rag.pipeline_orchestrator.tracer") as m_tracer:

        # 1. Cache (Default Miss)
        m_cache.find_one = AsyncMock(return_value=None)
        
        # 2. History
        m_hist.return_value = []
        
        # 3. Query Processor
        qp_inst = m_qp.return_value
        qp_inst.process_query = AsyncMock(return_value={
            "is_valid": True, 
            "is_greeting": False, 
            "expanded_query": "expanded query",
            "metadata": {},
            "greeting_response": None
        })
        
        # 4. Router
        router_inst = m_router.return_value
        router_inst.route_query = AsyncMock(return_value="complex")
        router_inst.get_model_info.return_value = {"model_name": "gpt-4"}
        
        # 5. LLM
        llm_inst = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Final LLM Answer"
        llm_inst.ainvoke = AsyncMock(return_value=mock_response)
        m_llm.return_value = llm_inst
        
        # 6. Retrieval
        m_retr.return_value = (
            [{"sector": "finance", "chunks_found": 1, "chunks": [{"content": "doc1"}]}], 
            [{"content": "doc1"}], 
            50.0 
        )
        
        # 7. Context Manager
        cm_inst = m_cm.return_value
        cm_inst.build_comparative_context.return_value = "Built Context"
        cm_inst.prepare_prompt.return_value = [{"role": "user", "content": "prompt"}]
        
        # 8. Sectors
        m_sectors.return_value = ["Finance"]

        # 9. Tracing
        mock_span = MagicMock()
        m_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        yield {
            "cache": m_cache,
            "llm": llm_inst,
            "retrieval": m_retr,
            "save_hist": m_save,
            "save_cache": m_cache_res,
            "qp": qp_inst,
            "get_llm": m_llm
        }

# --- Tests: Response Normalization ---

@pytest.mark.asyncio
async def test_normalize_response_no_chinese():
    """Test fast pass when no Chinese characters exist."""
    text = "Hello world."
    result = await _normalize_response(text, None)
    assert result == text

@pytest.mark.asyncio
async def test_normalize_response_with_chinese_success():
    """Test successful translation of detected Chinese text."""
    text = "Hello 世界" 
    
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Hello World" # Fixed
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    
    # FIX: Patch the missing 'get_simple_llm' using create=True
    # This prevents NameError by injecting the function into the module dynamically
    with patch("app.services.rag.pipeline_orchestrator.get_simple_llm", create=True) as m_get_simple:
        m_get_simple.return_value = mock_llm
        
        result = await _normalize_response(text, None)
    
    assert result == "Hello World"
    mock_llm.ainvoke.assert_called_once()

@pytest.mark.asyncio
async def test_normalize_response_failure():
    """Test fallback when translation fails/persists Chinese."""
    text = "世界"
    
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Still 世界" # LLM failed to fix it
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    
    # FIX: Patch the missing 'get_simple_llm' using create=True
    with patch("app.services.rag.pipeline_orchestrator.get_simple_llm", create=True) as m_get_simple:
        m_get_simple.return_value = mock_llm
        
        result = await _normalize_response(text, None)
    
    # Should attempt retries (2 times) then fail safe
    assert mock_llm.ainvoke.call_count == 2
    assert result == "Please ask your question again."

# --- Tests: Pipeline Orchestrator ---

@pytest.mark.asyncio
async def test_pipeline_cache_hit(mock_dependencies, mock_user):
    """Test that pipeline returns immediately if cache exists."""
    deps = mock_dependencies
    deps["cache"].find_one = AsyncMock(return_value={"llm_answer": "Cached Answer"})
    
    result = await query_rag_pipeline(
        query="test",
        current_user=mock_user,
        project_id="p1",
        ai_config={} 
    )
    
    assert result["result"] == "Cached Answer"
    assert result["from_cache"] is True
    deps["retrieval"].assert_not_called()
    deps["llm"].ainvoke.assert_not_called()

@pytest.mark.asyncio
async def test_pipeline_full_flow_success(mock_dependencies, mock_user):
    """Test the happy path of the entire pipeline."""
    deps = mock_dependencies
    
    result = await query_rag_pipeline(
        query="What are the risks?",
        current_user=mock_user,
        project_id="p1",
        sectors=["Finance"],
        ai_config={} 
    )
    
    assert result["result"] == "Final LLM Answer"
    assert result["model_used"] == "gpt-4"
    assert len(result["source_documents"]) == 1
    
    deps["qp"].process_query.assert_called()
    deps["retrieval"].assert_called()
    deps["llm"].ainvoke.assert_called()
    deps["save_hist"].assert_called()
    deps["save_cache"].assert_called()

@pytest.mark.asyncio
async def test_pipeline_invalid_query_block(mock_dependencies, mock_user):
    """Test that validation blocks invalid queries."""
    deps = mock_dependencies
    deps["qp"].process_query = AsyncMock(return_value={"is_valid": False})
    
    result = await query_rag_pipeline(
        query="blah", 
        current_user=mock_user, 
        project_id="p1",
        ai_config={} 
    )
    
    assert "Please ask a valid compliance related question" in result["result"]
    deps["retrieval"].assert_not_called()

@pytest.mark.asyncio
async def test_pipeline_no_documents_found(mock_dependencies, mock_user):
    """Test handling when retrieval finds nothing."""
    deps = mock_dependencies
    deps["retrieval"].return_value = ([], [], 0)
    
    result = await query_rag_pipeline(
        query="query", 
        current_user=mock_user, 
        project_id="p1",
        ai_config={} 
    )
    
    assert "could not find relevant information" in result["result"]
    deps["llm"].ainvoke.assert_not_called()

@pytest.mark.asyncio
async def test_pipeline_fallback_on_llm_error(mock_dependencies, mock_user):
    """Test that pipeline switches to simple model if complex model crashes."""
    deps = mock_dependencies
    
    # 1. Primary LLM Fails
    primary_llm = MagicMock()
    primary_llm.ainvoke.side_effect = Exception("Primary LLM Down")
    
    # 2. Fallback LLM Succeeds
    fallback_llm = MagicMock()
    fallback_resp = MagicMock()
    fallback_resp.content = "Fallback Answer"
    fallback_llm.ainvoke = AsyncMock(return_value=fallback_resp)
    
    # Setup get_llm to return primary first, then fallback
    def side_effect(model_name):
        if "simple" in model_name or "gpt-3.5" in model_name: 
            return fallback_llm
        return primary_llm
        
    deps["get_llm"].side_effect = side_effect
    
    # Need to mock settings to ensure fallback name is known
    with patch("app.services.rag.pipeline_orchestrator.settings") as m_settings:
        m_settings.LLM_MODEL_SIMPLE = "gpt-3.5-turbo"
        m_settings.LLM_MODEL = "gpt-4"
        
        result = await query_rag_pipeline(
            query="q", 
            current_user=mock_user, 
            project_id="p1",
            ai_config={} 
        )
        
        assert result["result"] == "Fallback Answer"
        assert "Fallback" in result["model_used"]

@pytest.mark.asyncio
async def test_pipeline_config_overrides(mock_dependencies, mock_user):
    """Test that ai_config passes parameters correctly."""
    deps = mock_dependencies
    
    custom_config = {
        "retrieval_depth": 20,
        "enable_reranking": False,
        "search_strategy": "vector"
    }
    
    await query_rag_pipeline(
        "q", mock_user, "p1", 
        ai_config=custom_config
    )
    
    # Check Retrieval Call Args
    call_kwargs = deps["retrieval"].call_args.kwargs
    
    # Results per sector = max(3, 20/2) = 10
    assert call_kwargs["results_per_sector"] == 10
    assert call_kwargs["use_reranking"] is False
    assert call_kwargs["dense_weight"] == 1.0
    assert call_kwargs["sparse_weight"] == 0.0

@pytest.mark.asyncio
async def test_background_evaluation_trigger(mock_dependencies, mock_user):
    """Test that background evaluation task is scheduled."""
    with patch("app.services.rag.pipeline_orchestrator._run_comprehensive_evaluation_background") as m_runner:
        background_tasks = MagicMock()
        
        await query_rag_pipeline(
            "q", mock_user, "p1", 
            background_tasks=background_tasks,
            ai_config={} 
        )
        
        background_tasks.add_task.assert_called_once()
        args = background_tasks.add_task.call_args
        assert args[0][0] == m_runner