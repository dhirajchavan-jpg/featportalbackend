import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from bson import ObjectId

from app.main import app
from app.dependencies import UserPayload, get_current_user

# ------------------------------------------------------------------
# FIXTURES
# ------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def valid_user_id():
    """Generates a valid MongoDB ObjectId string."""
    return str(ObjectId())

@pytest.fixture
def test_user(valid_user_id):
    return UserPayload(
        user_id=valid_user_id,  # FIX: Use a valid ObjectId string
        email="user@test.com",
        role="admin"
    )

@pytest.fixture
def evaluation_doc(valid_user_id):
    return {
        "_id": ObjectId(),
        "user_id": valid_user_id,  # FIX: Use valid ID matching the user
        "query": "What is RBI?",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),

        "query_processing": {"expansion_quality_score": 0.9},

        "retrieval": {
            "estimated_hit_rate_at_3": 0.8
        },

        "reranking": {"reranking_effectiveness_score": 0.75},
        "context": {"context_quality_score": 0.7},

        "generation": {
            "overall_score": 80,
            "query_toxicity_score": 0.05,
            "response_toxicity_score": 0.02
        },

        "pipeline_health_score": 0.85
    }


@pytest.fixture
def mock_stats():
    return {
        "total_evaluated": 1,
        "avg_pipeline_health": 0.85,
        "avg_query_quality": 0.9,
        "avg_retrieval_hit_rate": 0.8,
        "avg_reranking_effectiveness": 0.75,
        "avg_context_quality": 0.7,
        "avg_generation_score": 80,
        "time_window_hours": 24
    }


@pytest.fixture
def mock_health():
    return {
        "overall_health": 0.85,
        "stage_health": {
            "query": 0.9,
            "retrieval": 0.8,
            "reranking": 0.75,
            "context": 0.7,
            "generation": 0.8
        },
        "issues": [],
        "recommendations": []
    }


# ------------------------------------------------------------------
# DEPENDENCY OVERRIDES
# ------------------------------------------------------------------

def override_user(user):
    app.dependency_overrides[get_current_user] = lambda: user


def clear_overrides():
    app.dependency_overrides.clear()


# ------------------------------------------------------------------
# TESTS
# ------------------------------------------------------------------

def test_get_evaluation_stats(client, test_user, mock_stats):
    override_user(test_user)

    with patch(
        "app.routes.evaluation.get_comprehensive_stats",
        new=AsyncMock(return_value=mock_stats)
    ):
        res = client.get("/api/evaluation/stats")

    clear_overrides()

    assert res.status_code == 200
    assert res.json()["total_evaluated"] == 1


def test_get_pipeline_health(client, test_user, mock_health):
    override_user(test_user)

    with patch(
        "app.routes.evaluation.get_pipeline_health",
        new=AsyncMock(return_value=mock_health)
    ):
        res = client.get("/api/evaluation/health")

    clear_overrides()

    assert res.status_code == 200
    assert res.json()["overall_health"] == 0.85


def test_get_evaluation_history(client, test_user, evaluation_doc):
    override_user(test_user)

    # Mock the Cursor
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.limit.return_value = cursor
    # to_list is async
    cursor.to_list = AsyncMock(return_value=[evaluation_doc])

    # Mock the Users Collection find_one
    users_collection = MagicMock()
    users_collection.find_one = AsyncMock(return_value={
        "_id": ObjectId(test_user.user_id),
        "name": "Test User",
        "role": "admin"
    })

    # Mock db dictionary access: db['collection_name']
    def collection_selector(name):
        if name == "comprehensive_evaluations":
            mock_col = MagicMock()
            mock_col.find.return_value = cursor
            return mock_col
        if name == "users":
            return users_collection
        return MagicMock()

    with patch("app.routes.evaluation.db") as mock_db:
        mock_db.__getitem__.side_effect = collection_selector
        
        res = client.get("/api/evaluation/history?limit=10&skip=0")

    clear_overrides()

    assert res.status_code == 200
    assert isinstance(res.json(), list)
    assert "user" in res.json()[0]


def test_get_stage_metrics_valid(client, test_user, evaluation_doc):
    override_user(test_user)

    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.limit.return_value = cursor
    cursor.to_list = AsyncMock(return_value=[evaluation_doc])

    mock_collection = MagicMock()
    mock_collection.find.return_value = cursor

    # FIX: Patch __getitem__ because code uses db['collection']
    with patch("app.routes.evaluation.db") as mock_db:
        mock_db.__getitem__.return_value = mock_collection
        
        res = client.get("/api/evaluation/metrics/by-stage/retrieval")

    clear_overrides()

    assert res.status_code == 200
    body = res.json()
    assert body["stage"] == "retrieval"
    assert body["total_evaluations"] == 1


def test_get_stage_metrics_invalid(client, test_user):
    override_user(test_user)

    res = client.get("/api/evaluation/metrics/by-stage/invalid_stage")

    clear_overrides()

    assert res.status_code == 400


def test_get_evaluation_trends(client, test_user, evaluation_doc):
    override_user(test_user)

    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.limit.return_value = cursor
    cursor.to_list = AsyncMock(return_value=[evaluation_doc])

    mock_collection = MagicMock()
    mock_collection.find.return_value = cursor

    # FIX: Patch __getitem__ because code uses db['collection']
    with patch("app.routes.evaluation.db") as mock_db:
        mock_db.__getitem__.return_value = mock_collection
        
        res = client.get("/api/evaluation/trends")

    clear_overrides()

    assert res.status_code == 200