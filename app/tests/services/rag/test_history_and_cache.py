import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from app.services.rag.history_and_cache import (
    _get_chat_history,
    _save_to_history,
    _save_file_upload_to_history,
    _build_cache_key,
    _cache_result
)

# --- Fixtures ---

@pytest.fixture
def mock_collections():
    """Mock the MongoDB collections."""
    with patch("app.services.rag.history_and_cache.chat_history_collection") as mock_hist, \
         patch("app.services.rag.history_and_cache.cache_collection") as mock_cache:
        yield mock_hist, mock_cache

# --- Tests ---

@pytest.mark.asyncio
async def test_get_chat_history(mock_collections):
    """Test retrieving chat history."""
    mock_hist_coll, _ = mock_collections
    
    # Mock cursor behavior
    mock_cursor = MagicMock()
    # Chain: find() -> sort() -> limit() -> to_list()
    mock_hist_coll.find.return_value = mock_cursor
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.limit.return_value = mock_cursor
    
    # Mock data returned by to_list
    mock_data = [
        {"user_query": "q2", "llm_answer": "a2"},
        {"user_query": "q1", "llm_answer": "a1"}
    ]
    mock_cursor.to_list = AsyncMock(return_value=mock_data)

    result = await _get_chat_history("chat_123", limit=2)

    # Logic reverses the list before returning
    assert result[0]["user_query"] == "q1"
    assert result[1]["user_query"] == "q2"
    
    # Verify DB call structure
    mock_hist_coll.find.assert_called_with(
        {"chat_id": "chat_123", "message_type": "text"},
        projection={"user_query": 1, "llm_answer": 1, "sector": 1, "_id": 0}
    )
    mock_cursor.sort.assert_called_with("created_at", -1)
    mock_cursor.limit.assert_called_with(2)

@pytest.mark.asyncio
async def test_save_to_history(mock_collections):
    """Test saving a standard interaction to history."""
    mock_hist_coll, _ = mock_collections
    mock_hist_coll.insert_one = AsyncMock()

    await _save_to_history(
        chat_id="chat_1",
        user_id="user_1",
        query="query",
        project_id="proj_1",
        sectors=["sec1"],
        answer="answer",
        source_documents=[{"page_content": "doc1"}],
        retrieval_stats={"stat": 1},
        meta_data={"meta": 2}
    )

    # Verify insertion
    mock_hist_coll.insert_one.assert_called_once()
    call_args = mock_hist_coll.insert_one.call_args[0][0]
    
    assert call_args["chat_id"] == "chat_1"
    assert call_args["user_query"] == "query"
    assert call_args["llm_answer"] == "answer"
    assert call_args["source_documents"][0]["page_content"] == "doc1"
    assert call_args["retrieval_stats"] == {"stat": 1}
    assert call_args["meta"] == {"meta": 2}
    assert isinstance(call_args["created_at"], datetime)
    # Style wasn't provided in the call, should be None
    assert call_args.get("style") is None

@pytest.mark.asyncio
async def test_save_to_history_object_conversion(mock_collections):
    """Test converting object-like documents to dicts before saving."""
    mock_hist_coll, _ = mock_collections
    mock_hist_coll.insert_one = AsyncMock()

    # Create a dummy object mimicking a LangChain Document
    class MockDoc:
        page_content = "text"
        metadata = {"id": 1}
        relevance_score = 0.9
        rerank_score = None # Test None handling

    docs = [MockDoc()]

    await _save_to_history("c", "u", "q", "p", [], "a", source_documents=docs)

    call_args = mock_hist_coll.insert_one.call_args[0][0]
    saved_doc = call_args["source_documents"][0]
    
    assert saved_doc["page_content"] == "text"
    assert saved_doc["metadata"] == {"id": 1}
    assert saved_doc["relevance_score"] == 0.9
    assert saved_doc["rerank_score"] is None
    # Style defaults to None when not provided
    call_args = mock_hist_coll.insert_one.call_args[0][0]
    assert call_args.get("style") is None

@pytest.mark.asyncio
async def test_save_file_upload_to_history(mock_collections):
    """Test saving file upload event."""
    mock_hist_coll, _ = mock_collections
    mock_hist_coll.insert_one = AsyncMock()

    await _save_file_upload_to_history(
        "chat_1", "user_1", "proj_1", "sec_1", "file_id_1", "file.pdf"
    )

    mock_hist_coll.insert_one.assert_called_once()
    call_args = mock_hist_coll.insert_one.call_args[0][0]
    
    assert call_args["message_type"] == "file"
    assert call_args["file_id"] == "file_id_1"
    assert call_args["file_name"] == "file.pdf"
    assert call_args["user_query"] is None

def test_build_cache_key():
    """Test cache key generation logic."""
    key = _build_cache_key(
        user_id="u1",
        project_id="p1",
        query="  TEST Query  ",
        sectors=["B", "A"], # Should sort
        excluded_files=["Y", "X"] # Should sort
    )
    # style defaults to 'Detailed' and is included in the key
    expected = "user_u1|project_p1|sectors_A,B|excl_X,Y|style_Detailed|query_test query"
    assert key == expected

def test_build_cache_key_none_values():
    """Test cache key generation with optional Nones."""
    key = _build_cache_key("u1", "p1", "q", None, None)
    expected = "user_u1|project_p1|sectors_none|excl_none|style_Detailed|query_q"
    assert key == expected

@pytest.mark.asyncio
async def test_cache_result(mock_collections):
    """Test caching a result."""
    _, mock_cache_coll = mock_collections
    mock_cache_coll.insert_one = AsyncMock()

    await _cache_result("key_1", "u1", "q", "p1", ["s1"], "ans")

    mock_cache_coll.insert_one.assert_called_once()
    call_args = mock_cache_coll.insert_one.call_args[0][0]
    
    assert call_args["cache_key"] == "key_1"
    assert call_args["llm_answer"] == "ans"
    assert isinstance(call_args["created_at"], datetime)
    # Style wasn't provided; ensure it was stored as None
    assert call_args.get("style") is None

@pytest.mark.asyncio
async def test_db_error_handling(mock_collections):
    """Test that DB errors are logged but don't crash the app."""
    mock_hist_coll, _ = mock_collections
    
    # Simulate DB failure
    mock_hist_coll.insert_one.side_effect = Exception("DB Down")
    
    # Should run without raising exception
    await _save_to_history("c", "u", "q", "p", [], "a")
    
    # If we reached here, test passed (exception swallowed/logged)
    assert True