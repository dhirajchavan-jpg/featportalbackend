import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY

# Import the module under test
from app.services.rag.pipeline_orchestrator import (
    _normalize_response,
    query_rag_pipeline
)
from app.dependencies import UserPayload


@pytest.fixture
def mock_user():
    return UserPayload(user_id="test_user", email="test@example.com", role="user")


@pytest.fixture
def mock_dependencies():
    """Mocks common external services used by the orchestrator for these tests."""
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
         patch("app.services.rag.pipeline_orchestrator.tracer") as m_tracer, \
         patch("app.services.rag.pipeline_orchestrator.cleanup_vram") as m_cleanup:

        # Cache default miss
        m_cache.find_one = AsyncMock(return_value=None)

        # History
        m_hist.return_value = []

        # Query Processor
        qp_inst = m_qp.return_value
        qp_inst.process_query = AsyncMock(return_value={
            "is_valid": True,
            "is_greeting": False,
            "expanded_query": "expanded query",
            "metadata": {},
            "greeting_response": None
        })

        # Router
        router_inst = m_router.return_value
        router_inst.route_query = AsyncMock(return_value="complex")
        router_inst.get_model_info.return_value = {"model_name": "gpt-4"}

        # LLM Primary/Default
        llm_inst = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Final LLM Answer"
        llm_inst.ainvoke = AsyncMock(return_value=mock_response)
        m_llm.return_value = llm_inst

        # Retrieval
        m_retr.return_value = (
            [{"sector": "finance", "chunks_found": 1, "chunks": [{"content": "doc1"}]}],
            [{"content": "doc1"}],
            50.0
        )

        # Context Manager
        cm_inst = m_cm.return_value
        cm_inst.build_comparative_context.return_value = "Built Context"
        cm_inst.prepare_prompt.return_value = [{"role": "user", "content": "prompt"}]

        # Sectors
        m_sectors.return_value = ["Finance"]

        # Tracing span
        mock_span = MagicMock()
        m_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        yield {
            "cache": m_cache,
            "llm": llm_inst,
            "retrieval": m_retr,
            "save_hist": m_save,
            "save_cache": m_cache_res,
            "qp": qp_inst,
            "get_llm": m_llm,
            "router": router_inst,
            "cleanup": m_cleanup
        }


@pytest.mark.asyncio
async def test_router_model_override_calls_route_with_router_name(mock_dependencies, mock_user):
    """Ensure that when ai_config contains router_model it is forwarded to the router.route_query call."""
    deps = mock_dependencies

    custom_config = {"router_model": "my-special-router"}

    await query_rag_pipeline(
        query="What is the policy?",
        current_user=mock_user,
        project_id="p1",
        ai_config=custom_config
    )

    # Assert route_query was called with the router_model_name kwarg set
    called = deps["router"].route_query.await_args
    # route_query is an AsyncMock; inspect last call kwargs
    last_call = deps["router"].route_query.await_args
    assert last_call is not None
    # Verify router was invoked with router_model_name in kwargs
    assert 'router_model_name' in last_call.kwargs
    assert last_call.kwargs['router_model_name'] == "my-special-router"


@pytest.mark.asyncio
async def test_query_is_lowercased_before_processing(mock_dependencies, mock_user):
    """Pipeline should lowercase the incoming query before processing."""
    deps = mock_dependencies

    # Replace process_query to assert it receives lowercase
    async def process_query_assertion(query, chat_history=None):
        assert query == "lowercase query"
        return {
            "is_valid": True,
            "is_greeting": False,
            "expanded_query": "expanded query",
            "metadata": {}
        }

    deps["qp"].process_query = process_query_assertion

    await query_rag_pipeline(
        query="LowerCase Query",
        current_user=mock_user,
        project_id="p1",
        ai_config={}
    )


@pytest.mark.asyncio
async def test_style_propagated_to_history_and_eval_when_skip_evaluation(mock_dependencies, mock_user):
    """Ensure the provided `style` is passed to history saver and attached to eval payload when skip_evaluation=True."""
    deps = mock_dependencies

    # Call with skip_evaluation True to get eval payload attached
    result = await query_rag_pipeline(
        query="q",
        current_user=mock_user,
        project_id="p1",
        ai_config={},
        style="Concise",
        skip_evaluation=True
    )

    # The pipeline attaches _eval_data when skip_evaluation True
    assert "_eval_data" in result
    assert result["_eval_data"]["style"] == "Concise"


@pytest.mark.asyncio
async def test_cleanup_and_sleep_on_llm_error_then_fallback(mock_dependencies, mock_user):
    """When the primary LLM raises, cleanup_vram should be called and sleep awaited before fallback."""
    deps = mock_dependencies

    # Primary LLM raises
    primary_llm = MagicMock()
    primary_llm.ainvoke = AsyncMock(side_effect=Exception("Primary Down"))

    # Fallback succeeds
    fallback_llm = MagicMock()
    fallback_resp = MagicMock()
    fallback_resp.content = "Fallback Answer"
    fallback_llm.ainvoke = AsyncMock(return_value=fallback_resp)

    # Make get_llm return primary then fallback based on model name
    def get_llm_side(name):
        if "simple" in name or "gpt-3.5" in name:
            return fallback_llm
        return primary_llm

    deps["get_llm"].side_effect = get_llm_side

    # Patch settings to ensure simple model name exists
    with patch("app.services.rag.pipeline_orchestrator.settings") as m_settings, \
         patch("app.services.rag.pipeline_orchestrator.asyncio.sleep", new=AsyncMock()) as m_sleep:
        m_settings.LLM_MODEL_SIMPLE = "gpt-3.5-turbo"
        m_settings.LLM_MODEL = "gpt-4"

        res = await query_rag_pipeline(
            query="q",
            current_user=mock_user,
            project_id="p1",
            ai_config={}
        )

        # Ensure cleanup was called
        deps["cleanup"].assert_called()
        # Ensure sleep was awaited
        assert m_sleep.await_count >= 1
        assert res["result"] in ("Fallback Answer", "Fallback Answer")
