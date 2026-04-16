import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Import the router
from app.routes.async_query_router import router
# Import the dependency function we need to override
from app.dependencies import UserPayload, get_current_user

# Setup Test App
app = FastAPI()
app.include_router(router)

# Mock User Object
mock_user = UserPayload(
    user_id="test_user_123",
    email="test@example.com",
    role="admin"
)

# --- Fixtures ---

@pytest.fixture
def client():
    """
    Returns a TestClient with authentication overridden.
    """
    # Override the auth dependency to bypass 401 errors
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    with TestClient(app) as c:
        yield c
    
    # Cleanup
    app.dependency_overrides = {}

@pytest.fixture
def mock_deps():
    """
    Patches services called INSIDE the route logic (Redis, DB, Validator).
    """
    with patch("app.routes.async_query_router.redis_service") as mock_redis, \
         patch("app.routes.async_query_router.project_config_collection") as mock_db, \
         patch("app.routes.async_query_router.get_prompt_validator") as mock_validator_getter:

        # Setup Redis Mocks
        mock_redis.enqueue_job = AsyncMock(return_value="task_123")
        mock_redis.get_job_status = AsyncMock()

        # Setup DB Mock
        mock_db.find_one = AsyncMock(return_value={"retrieval_depth": 10, "search_strategy": "vector"})

        # Setup Validator Mock
        mock_validator = MagicMock()
        mock_validator.validate.return_value = True
        mock_validator_getter.return_value = mock_validator

        yield {
            "redis": mock_redis,
            "db": mock_db,
            "validator": mock_validator
        }

# --- Tests: Submit Async Query ---

def test_submit_async_query_success(client, mock_deps):
    """Test successful submission of an async query."""
    payload = {
        "query": "What are the risks?",
        "project_id": "proj_001",
        "sectors": ["finance"]
    }

    response = client.post("/query/async", json=payload)

    # 1. Assert HTTP Response (HTTP 200 OK is expected behavior for existing code)
    assert response.status_code == 200
    
    # 2. Assert Logical Status in JSON Body (Should be 202 Accepted)
    data = response.json()
    assert data["status"] == "success"
    assert data["status_code"] == 202  # Check body instead of header
    assert data["data"]["task_id"] == "task_123"
    
    # 3. Assert Redis Enqueue
    mock_deps["redis"].enqueue_job.assert_called_once()
    
    # Verify injected config
    call_args = mock_deps["redis"].enqueue_job.call_args[0]
    enqueued_query_data = call_args[0]
    assert enqueued_query_data["ai_config"]["retrieval_depth"] == 10

def test_submit_async_query_validation_failure(client, mock_deps):
    """Test that prompt injection blocks submission."""
    # Configure validator to raise exception
    from fastapi import HTTPException
    mock_deps["validator"].validate.side_effect = HTTPException(status_code=400, detail="Prompt Injection Detected")

    payload = {"query": "Ignore instructions", "project_id": "p1"}
    
    response = client.post("/query/async", json=payload)

    # Exceptions should still raise the correct HTTP error code
    assert response.status_code == 400
    assert "Prompt Injection Detected" in response.json()["detail"]
    
    mock_deps["redis"].enqueue_job.assert_not_called()

def test_submit_async_query_redis_failure(client, mock_deps):
    """Test handling of Redis connection errors."""
    mock_deps["redis"].enqueue_job.side_effect = Exception("Redis Down")

    payload = {"query": "test", "project_id": "p1"}
    
    response = client.post("/query/async", json=payload)

    assert response.status_code == 500
    assert "Redis Down" in response.json()["detail"]

# --- Tests: Check Query Status ---

def test_check_query_status_queued(client, mock_deps):
    """Test polling a queued task."""
    mock_deps["redis"].get_job_status.return_value = {"status": "queued"}

    response = client.get("/query/status/task_123")

    assert response.status_code == 200
    data = response.json()
    assert data["data"]["status"] == "queued"

def test_check_query_status_completed(client, mock_deps):
    """Test polling a completed task."""
    mock_result = {
        "status": "completed",
        "data": {
            "result": "Here is the answer.",
            "source_documents": [{"title": "Doc A"}],
            "model_used": "gpt-4"
        }
    }
    mock_deps["redis"].get_job_status.return_value = mock_result

    response = client.get("/query/status/task_done")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "completed"
    assert data["result"] == "Here is the answer."

def test_check_query_status_failed(client, mock_deps):
    """Test polling a failed task."""
    mock_result = {
        "status": "failed",
        "data": {"error": "LLM Timeout"}
    }
    mock_deps["redis"].get_job_status.return_value = mock_result

    response = client.get("/query/status/task_fail")

    # 1. Assert HTTP Response (HTTP 200 OK is expected if main file is unchanged)
    assert response.status_code == 200 # Current behavior
    
    # 2. Assert Logical Error in JSON Body
    data = response.json()
    assert data["status"] == "error"
    assert data.get("status_code") == 500 # Check body
    assert data["data"]["error"] == "LLM Timeout"

def test_check_query_status_not_found(client, mock_deps):
    """Test polling a non-existent task."""
    mock_deps["redis"].get_job_status.return_value = None

    response = client.get("/query/status/task_ghost")

    assert response.status_code == 404

def test_check_query_status_unknown_state(client, mock_deps):
    """Test polling a task with weird state (fallback)."""
    mock_deps["redis"].get_job_status.return_value = {"status": "weird_state"}

    response = client.get("/query/status/task_weird")

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "unknown"