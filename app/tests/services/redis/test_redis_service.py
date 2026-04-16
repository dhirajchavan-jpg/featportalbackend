import pytest
import json
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from redis.exceptions import ConnectionError

from app.services.redis.redis_service import RedisService
from app.config import settings

# --- FIXTURES ---

@pytest.fixture
def mock_redis_client():
    """
    Patches the low-level redis.from_url to return a fully AsyncMock client.
    This prevents actual connection attempts during test initialization.
    """
    with patch("app.services.redis.redis_service.redis.from_url") as mock_from_url:
        # Create a mock client that simulates AsyncRedis behavior
        mock_client = AsyncMock()
        mock_from_url.return_value = mock_client
        yield mock_client

@pytest.fixture
def redis_service(mock_redis_client):
    """
    Returns an instance of RedisService using the mocked client.
    """
    return RedisService()

# --- TESTS ---

@pytest.mark.asyncio
async def test_enqueue_job_success(redis_service, mock_redis_client):
    """
    Test standard job enqueueing:
    1. Pushes to Redis List (lpush)
    2. Sets initial status (setex)
    """
    # Setup
    mock_redis_client.lpush.return_value = 1 # Redis returns list length
    mock_redis_client.setex.return_value = True

    job_data = {"query": "test"}
    user_data = {"user_id": "u1"}

    # Act
    task_id = await redis_service.enqueue_job("query", job_data, user_data)

    # Assert
    assert task_id is not None
    
    # 1. Verify LPUSH (The Queue)
    mock_redis_client.lpush.assert_called_once()
    call_args = mock_redis_client.lpush.call_args
    assert call_args[0][0] == settings.REDIS_QUEUE_NAME
    payload = json.loads(call_args[0][1])
    assert payload["task_id"] == task_id
    assert payload["job_type"] == "query"
    assert payload["user_data"]["user_id"] == "u1"

    # 2. Verify SETEX (The Status Key)
    mock_redis_client.setex.assert_called_once()
    status_args = mock_redis_client.setex.call_args
    assert status_args[0][0] == f"job:{task_id}"
    assert status_args[0][1] == settings.REDIS_RESULT_TTL

@pytest.mark.asyncio
async def test_enqueue_job_retry_logic(redis_service, mock_redis_client):
    """
    Test that enqueue_job retries on ConnectionError (WinError 64 simulation).
    Scenario: Fail twice, succeed on third attempt.
    """
    # Setup: Fail twice, then succeed
    mock_redis_client.lpush.side_effect = [
        ConnectionError("WinError 64"), 
        ConnectionError("WinError 64"), 
        1 # Success
    ]

    # Act
    task_id = await redis_service.enqueue_job("query", {}, {})

    # Assert
    assert task_id is not None
    assert mock_redis_client.lpush.call_count == 3 # Verified retries

@pytest.mark.asyncio
async def test_enqueue_job_max_retries_exceeded(redis_service, mock_redis_client):
    """
    Test that enqueue_job raises exception after max retries exhausted.
    """
    # Setup: Always fail
    mock_redis_client.lpush.side_effect = ConnectionError("Fatal Error")

    # Act & Assert
    with pytest.raises(ConnectionError):
        await redis_service.enqueue_job("query", {}, {})
    
    # Assuming max_retries is 3
    assert mock_redis_client.lpush.call_count == 3

@pytest.mark.asyncio
async def test_get_job_status_found(redis_service, mock_redis_client):
    """Test retrieving existing job status."""
    task_id = "t1"
    mock_data = {"status": "processing", "task_id": task_id}
    mock_redis_client.get.return_value = json.dumps(mock_data)

    result = await redis_service.get_job_status(task_id)

    assert result["status"] == "processing"
    mock_redis_client.get.assert_called_with(f"job:{task_id}")

@pytest.mark.asyncio
async def test_get_job_status_not_found(redis_service, mock_redis_client):
    """Test retrieving non-existent job status."""
    mock_redis_client.get.return_value = None

    result = await redis_service.get_job_status("unknown")

    assert result is None

@pytest.mark.asyncio
async def test_update_job_result(redis_service, mock_redis_client):
    """Test updating job result (Worker side)."""
    task_id = "t1"
    result_data = {"answer": "42"}
    
    await redis_service.update_job_result(task_id, result_data, status="completed")

    mock_redis_client.setex.assert_called_once()
    args = mock_redis_client.setex.call_args[0]
    
    key = args[0]
    ttl = args[1]
    payload = json.loads(args[2])

    assert key == f"job:{task_id}"
    assert payload["status"] == "completed"
    assert payload["data"]["answer"] == "42"
    assert "completed_at" in payload

@pytest.mark.asyncio
async def test_cancel_job(redis_service, mock_redis_client):
    """Test setting the cancellation flag."""
    task_id = "cancel_me"
    
    await redis_service.cancel_job(task_id)
    
    mock_redis_client.setex.assert_called_with(
        f"job:{task_id}:cancel", 
        3600, 
        "true"
    )

@pytest.mark.asyncio
async def test_is_job_cancelled_true(redis_service, mock_redis_client):
    """Test checking cancellation flag (True)."""
    mock_redis_client.get.return_value = "true"
    
    is_cancelled = await redis_service.is_job_cancelled("t1")
    
    assert is_cancelled is True
    mock_redis_client.get.assert_called_with("job:t1:cancel")

@pytest.mark.asyncio
async def test_is_job_cancelled_false(redis_service, mock_redis_client):
    """Test checking cancellation flag (False/None)."""
    mock_redis_client.get.return_value = None
    
    is_cancelled = await redis_service.is_job_cancelled("t1")
    
    assert is_cancelled is False