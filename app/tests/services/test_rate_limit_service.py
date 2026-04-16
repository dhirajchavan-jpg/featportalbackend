import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
from pymongo.errors import PyMongoError

# Import service functions
from app.services.rate_limit_service import (
    try_consume_uploads, 
    try_consume_chat_token
)

# --- Fixtures ---

@pytest.fixture
def mock_settings():
    """Mock config settings to have predictable limits."""
    with patch("app.services.rate_limit_service.settings") as settings:
        # Set small limits for easy testing
        settings.RATE_LIMIT_UPLOADS_PER_HOUR = 5
        settings.RATE_LIMIT_CHAT_PER_MINUTE = 2
        
        # We also need to patch the module-level constants that imported these settings
        with patch("app.services.rate_limit_service.UPLOAD_LIMIT", 5), \
             patch("app.services.rate_limit_service.CHAT_CAPACITY", 2), \
             patch("app.services.rate_limit_service.CHAT_LIMIT", 2):
            yield settings

@pytest.fixture
def mock_collections():
    """Mock the Mongo collections used in the service."""
    with patch("app.services.rate_limit_service.UPLOADS_COLL") as uploads, \
         patch("app.services.rate_limit_service.CHAT_COLL") as chat:
        yield uploads, chat

# ============================================================================
# TESTS: Upload Rate Limit (Fixed Window)
# ============================================================================

@pytest.mark.asyncio
async def test_consume_uploads_success(mock_settings, mock_collections):
    """Test successful upload when under the limit."""
    uploads_coll, _ = mock_collections
    
    # Mock behavior: find_one_and_update returns the updated document (count=1)
    # ReturnDocument.AFTER behavior
    uploads_coll.find_one_and_update = AsyncMock(return_value={"count": 1, "user_id": "u1"})

    allowed, remaining, reset_time = await try_consume_uploads("u1", count=1)

    assert allowed is True
    assert remaining == 4  # Limit 5 - 1 used
    assert reset_time > 0
    uploads_coll.find_one_and_update.assert_called_once()

@pytest.mark.asyncio
async def test_consume_uploads_limit_exceeded(mock_settings, mock_collections):
    """Test rejection when upload limit is reached."""
    uploads_coll, _ = mock_collections
    
    # 1. find_one_and_update returns None because the filter ($lt limit) failed
    uploads_coll.find_one_and_update = AsyncMock(return_value=None)
    
    # 2. Fallback find_one returns current state (count=5, i.e., limit reached)
    uploads_coll.find_one = AsyncMock(return_value={"count": 5})

    allowed, remaining, reset_time = await try_consume_uploads("u1", count=1)

    assert allowed is False
    assert remaining == 0
    assert reset_time > 0
    # Both methods should have been called
    uploads_coll.find_one_and_update.assert_called_once()
    uploads_coll.find_one.assert_called_once()

@pytest.mark.asyncio
async def test_consume_uploads_db_error(mock_settings, mock_collections):
    """Test fail-safe behavior on database error."""
    uploads_coll, _ = mock_collections
    
    # Simulate DB crashing
    uploads_coll.find_one_and_update.side_effect = PyMongoError("Connection lost")

    allowed, remaining, reset_time = await try_consume_uploads("u1", count=1)

    assert allowed is False
    assert remaining == 0
    # Should still calculate reset time based on logic, even if DB fails
    assert reset_time > 0

# ============================================================================
# TESTS: Chat Rate Limit (Sliding/Fixed Window)
# ============================================================================

@pytest.mark.asyncio
async def test_chat_first_request_creates_window(mock_settings, mock_collections):
    """Test that a new user creates a new time window."""
    _, chat_coll = mock_collections
    
    # User not found -> First request
    chat_coll.find_one = AsyncMock(return_value=None)
    chat_coll.insert_one = AsyncMock()

    allowed, remaining, retry_after = await try_consume_chat_token("u1")

    assert allowed is True
    assert remaining == 1  # Capacity 2 - 1 used
    assert retry_after == 0
    chat_coll.insert_one.assert_called_once()

@pytest.mark.asyncio
async def test_chat_within_window_success(mock_settings, mock_collections):
    """Test existing window with capacity remaining."""
    _, chat_coll = mock_collections
    
    # Existing window started 10 seconds ago, count is 0
    now = datetime.utcnow()
    chat_coll.find_one = AsyncMock(return_value={
        "window_start": now - timedelta(seconds=10),
        "count": 0
    })
    chat_coll.update_one = AsyncMock()

    allowed, remaining, retry_after = await try_consume_chat_token("u1")

    assert allowed is True
    assert remaining == 1 # Capacity 2 - 1 used
    chat_coll.update_one.assert_called_once()
    
    # Verify we incremented count
    call_args = chat_coll.update_one.call_args[0]
    update_doc = call_args[1]
    assert update_doc["$set"]["count"] == 1

@pytest.mark.asyncio
async def test_chat_limit_exceeded(mock_settings, mock_collections):
    """Test blocking when capacity is full within active window."""
    _, chat_coll = mock_collections
    
    # Active window, count is ALREADY 2 (Limit is 2)
    now = datetime.utcnow()
    window_start = now - timedelta(seconds=30) # Window is half gone
    
    chat_coll.find_one = AsyncMock(return_value={
        "window_start": window_start,
        "count": 2 
    })

    allowed, remaining, retry_after = await try_consume_chat_token("u1")

    assert allowed is False
    assert remaining == 0
    # Retry after should be roughly 30 seconds (60s total - 30s elapsed)
    assert 29 <= retry_after <= 31 
    chat_coll.update_one.assert_not_called()

@pytest.mark.asyncio
async def test_chat_window_expired_reset(mock_settings, mock_collections):
    """Test that expired window resets count."""
    _, chat_coll = mock_collections
    
    # Window started 70 seconds ago (expired > 60s)
    # Even though count was 2 (full), it should reset now
    now = datetime.utcnow()
    chat_coll.find_one = AsyncMock(return_value={
        "window_start": now - timedelta(seconds=70),
        "count": 2 
    })
    chat_coll.update_one = AsyncMock()

    allowed, remaining, retry_after = await try_consume_chat_token("u1")

    assert allowed is True
    assert remaining == 1 # Reset to 0, then consumed 1. Remaining = 2 - 1 = 1
    
    # Verify update logic sets window_start to NOW
    call_args = chat_coll.update_one.call_args[0]
    update_doc = call_args[1]
    assert update_doc["$set"]["count"] == 1
    # Check that new window start is roughly 'now'
    assert isinstance(update_doc["$set"]["window_start"], datetime)

@pytest.mark.asyncio
async def test_chat_db_error_handling(mock_settings, mock_collections):
    """Test safe failure on DB error."""
    _, chat_coll = mock_collections
    
    chat_coll.find_one.side_effect = Exception("DB Down")

    allowed, remaining, retry_after = await try_consume_chat_token("u1")

    assert allowed is False
    assert remaining == 0
    assert retry_after == 1 # Default fallback