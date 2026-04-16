import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

from app.main import app

client = TestClient(app)

# ---------------------------------------------------------
# BASIC SMOKE TESTS
# ---------------------------------------------------------

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"


def test_health_endpoint_cpu_only():
    with patch("torch.cuda.is_available", return_value=False):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["status"] == "healthy"
        assert data["cuda_available"] is False


# ---------------------------------------------------------
# ANALYTICS ENDPOINT
# ---------------------------------------------------------

def test_admin_analytics():
    fake_summary = [
        {"endpoint": "/query", "total_requests": 10, "error_count": 2},
        {"endpoint": "/upload", "total_requests": 5, "error_count": 0},
    ]

    with patch("app.main.get_analytics_summary", return_value=fake_summary):
        response = client.get("/admin/analytics")
        assert response.status_code == 200

        payload = response.json()
        assert payload["overview"]["total_requests"] == 15
        assert payload["overview"]["total_errors"] == 2
        assert payload["overview"]["active_endpoints"] == 2


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------
# @pytest.fixture
# def override_get_current_user(app):
#     def fake_user():
#         return MagicMock(user_id="u1", email="test@test.com")

#     app.dependency_overrides[get_current_user] = fake_user
#     yield
#     app.dependency_overrides.clear()

# @pytest.mark.asyncio
# async def test_query_history():
#     fake_history = [
#         {
#             "created_at": "2024-01-01",
#             "message_type": "query",
#             "user_query": "hello",
#             "llm_answer": "hi",
#             "file_id": None,
#             "file_name": None,
#             "sector": "rbi"
#         }
#     ]

#     mock_cursor = MagicMock()
#     mock_cursor.sort.return_value.limit.return_value.to_list = AsyncMock(
#         return_value=fake_history
#     )

#     with patch("app.main.chat_history_collection.find", return_value=mock_cursor), \
#          patch("app.main.project_config_collection.find_one", AsyncMock(return_value=None)), \
#          patch("app.main.get_current_user") as mock_user:

#         mock_user.return_value = MagicMock(user_id="u1")

#         response = client.get(
#             "/query/history/project1",
#             headers={"Authorization": "Bearer fake"}
#         )

#         assert response.status_code == 200
#         body = response.json()
#         assert len(body["data"]) == 1


# ---------------------------------------------------------
# REPORT ENDPOINTS
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_report_drift_queries():
    fake_data = {"items": [{"query": "q1"}]}

    with patch(
        "app.main.get_drift_queries",
        AsyncMock(return_value=fake_data)
    ):
        response = client.get("/reports/abc/drift-queries")
        assert response.status_code == 200
        assert response.json() == fake_data


# ---------------------------------------------------------
# EXCEL EXPORT (NO FILE SYSTEM)
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_download_drift_report_excel():
    fake_items = {
        "items": [
            {
                "evaluated_at": "2024-01-01",
                "drift_type": "semantic",
                "query": "hello",
                "score": 0.7
            }
        ]
    }

    with patch(
        "app.main.get_drift_queries",
        AsyncMock(return_value=fake_items)
    ):
        response = client.get("/reports/xyz/download-excel")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(
            "application/vnd.openxmlformats"
        )


# ---------------------------------------------------------
# DEBUG ROUTES (SAFE MOCK)
# ---------------------------------------------------------

def test_list_collections():
    fake_client = MagicMock()
    fake_client.get_collections.return_value = {"collections": []}

    with patch("app.main.get_qdrant_client", return_value=fake_client):
        response = client.get("/debug/collections")
        assert response.status_code == 200


# ---------------------------------------------------------
# LOG EXPORT (NO FILE REQUIRED)
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_download_logs_excel_no_file(tmp_path):
    with patch("os.path.exists", return_value=False):
        response = client.get(
            "/logs/download-excel",
            params={
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
                "log_type": "system"
            }
        )
        assert response.status_code == 200
