import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import time

# Import the module under test
from app.services.rag.greeting_handler import _handle_greeting

# --- Fixtures ---

@pytest.fixture
def mock_dependencies():
    """Mocks external dependencies."""
    with patch("app.services.rag.greeting_handler._save_to_history") as m_save, \
         patch("app.services.rag.greeting_handler.tracer") as m_tracer:
        
        # Setup history save
        m_save.return_value = None  # Async function returns None
        
        yield {
            "save_hist": m_save,
            "tracer": m_tracer
        }

@pytest.fixture
def mock_user():
    """Creates a mock user object."""
    user = MagicMock()
    user.user_id = "test_user_123"
    return user

# --- Tests ---

@pytest.mark.asyncio
async def test_handle_greeting_pre_generated(mock_dependencies, mock_user):
    """Test handling when a greeting response is already provided."""
    query = "Hi there"
    chat_id = "chat_abc"
    proj = "proj_xyz"
    sectors = ["Finance"]
    
    # FIX: Set start time to 0.1s ago to ensure processing_time > 0
    start_time = time.time() - 0.1
    pre_gen_response = "Hello! Ready to help."

    result = await _handle_greeting(
        query=query,
        chat_id=chat_id,
        user=mock_user,
        proj=proj,
        sectors=sectors,
        start_time=start_time,
        greeting_response=pre_gen_response
    )

    # 1. Assert Response Content
    assert result["result"] == pre_gen_response
    assert result["is_greeting"] is True
    assert result["model_used"] == "router_0.5B_smart_greeting"
    assert result["source_documents"] == []
    
    # 2. Assert Processing Time (Should be approx 0.1s)
    assert result["processing_time"] > 0

    # 3. Assert History Saving
    mock_dependencies["save_hist"].assert_called_once_with(
        chat_id, 
        mock_user.user_id, 
        query, 
        proj, 
        sectors, 
        pre_gen_response
    )

@pytest.mark.asyncio
async def test_handle_greeting_fallback(mock_dependencies, mock_user):
    """Test handling when no greeting response is provided (fallback)."""
    query = "Hello"
    chat_id = "chat_fallback"
    proj = "proj_fallback"
    sectors = []
    
    # Ensure >0 duration
    start_time = time.time() - 0.1

    # greeting_response is None by default
    result = await _handle_greeting(
        query=query,
        chat_id=chat_id,
        user=mock_user,
        proj=proj,
        sectors=sectors,
        start_time=start_time
    )

    # 1. Assert Fallback Content
    expected_fallback = "Hello! I'm your Compliance Assistant. How can I help you with banking, compliance, or regulatory queries today?"
    assert result["result"] == expected_fallback
    assert result["is_greeting"] is True

    # 2. Assert History Saving with Fallback
    mock_dependencies["save_hist"].assert_called_once_with(
        chat_id, 
        mock_user.user_id, 
        query, 
        proj, 
        sectors, 
        expected_fallback
    )

@pytest.mark.asyncio
async def test_handle_greeting_processing_time(mock_dependencies, mock_user):
    """Test that processing time calculation is reasonable."""
    # Simulate start time 1 second ago
    start_time = time.time() - 1.0 
    
    result = await _handle_greeting(
        query="hi",
        chat_id="c1",
        user=mock_user,
        proj="p1",
        sectors=[],
        start_time=start_time,
        greeting_response="Hi"
    )
    
    # Should be at least 1 second
    assert result["processing_time"] >= 1.0
    # Should not be wildly incorrect (e.g. < 2s for this fast op)
    assert result["processing_time"] < 2.0