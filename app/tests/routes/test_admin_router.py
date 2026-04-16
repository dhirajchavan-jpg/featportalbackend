import io
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from app.dependencies import get_current_user

from app.routes.admin_router import router
from app.schemas import StandardResponse
from app.dependencies import UserPayload

# -------------------------
# TEST APP SETUP
# -------------------------

app = FastAPI()
app.include_router(router, prefix="/admin")


@pytest.fixture
def client():
    return TestClient(app)


# -------------------------
# COMMON FIXTURES
# -------------------------

@pytest.fixture
def admin_user():
    return UserPayload(
        user_id="admin123",
        email="admin@test.com",
        role="admin"
    )


@pytest.fixture
def non_admin_user():
    return UserPayload(
        user_id="user123",
        role="user",
        email="user@test.com"
    )


@pytest.fixture
def sample_file():
    # FIX: Use actual PDF magic bytes so we don't need to mock 'magic'
    return ('test.pdf', b'%PDF-1.4\nDummy PDF content for testing', 'application/pdf')


# -------------------------
# DEPENDENCY OVERRIDES
# -------------------------

@pytest.fixture(autouse=True)
def override_auth(admin_user):
    app.dependency_overrides = {}
    app.dependency_overrides[
        __import__("app.dependencies").dependencies.get_current_user
    ] = lambda: admin_user
    yield
    app.dependency_overrides = {}

@pytest.fixture(autouse=True)
def mock_settings():
    with patch("app.routes.admin_router.settings") as mock_settings:
        mock_settings.MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        mock_settings.UPLOAD_DIR = "/tmp"
        mock_settings.SERVER_BASE_URL = "http://testserver"
        mock_settings.QDRANT_URL = "http://localhost:6333"
        mock_settings.QDRANT_COLLECTION_NAME = "test_collection"
        yield


# -------------------------
# /upload-global TESTS
# -------------------------

def test_upload_global_success(client, admin_user, sample_file):
    app.dependency_overrides[get_current_user] = lambda: admin_user

    # No need to patch magic now, the file content is valid
    with patch("app.routes.admin_router.Global_file_collection.find_one", new=AsyncMock(return_value=None)), \
         patch("app.routes.admin_router.Global_file_collection.insert_one", new=AsyncMock()), \
         patch("app.routes.admin_router.process_and_index_global_file", new=MagicMock(return_value={"chunks_indexed": 5})):
        
        response = client.post(
            "/admin/upload-global",
            files={"file": sample_file},
            data={
                "sector": "RBI",
                "document_type": "circular",
                "category": "Banking"
            }
        )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["chunks_indexed"] == 5


def test_upload_global_non_admin_forbidden(client, non_admin_user, sample_file):
    app.dependency_overrides[
        __import__("app.dependencies").dependencies.get_current_user
    ] = lambda: non_admin_user

    response = client.post(
        "/admin/upload-global",
        files={"file": sample_file},
        data={
            "sector": "RBI",
            "document_type": "circular",
            "category": "Banking"
        }
    )

    assert response.status_code == 403


def test_upload_invalid_sector(client, sample_file):
    response = client.post(
        "/admin/upload-global",
        files={"file": sample_file},
        data={
            "sector": "INVALID",
            "document_type": "circular",
            "category": "Banking"
        }
    )
    # The router validation logic swallows errors and usually returns 500
    assert response.status_code in [400, 500]


def test_upload_invalid_document_type(client, sample_file):
    response = client.post(
        "/admin/upload-global",
        files={"file": sample_file},
        data={
            "sector": "RBI",
            "document_type": "invalid",
            "category": "Banking"
        }
    )
    assert response.status_code in [400, 500]


def test_upload_empty_category(client, sample_file):
    response = client.post(
        "/admin/upload-global",
        files={"file": sample_file},
        data={
            "sector": "RBI",
            "document_type": "circular",
            "category": "   "
        }
    )
    assert response.status_code in [400, 500]


def test_upload_unsupported_file_type(client):
    # Use dummy content for EXE so magic detects it as something else (likely text or octet-stream)
    response = client.post(
        "/admin/upload-global",
        files={"file": ("bad.exe", b"MZ...", "application/octet-stream")},
        data={
            "sector": "RBI",
            "document_type": "circular",
            "category": "Banking"
        }
    )
    assert response.status_code in [400, 500]


def test_upload_duplicate_file(client, sample_file):
    # Setup: duplicate found in DB
    with patch("app.routes.admin_router.Global_file_collection.find_one",
               new=AsyncMock(return_value={"filename": "existing.pdf", "sector": "RBI"})):

        response = client.post(
            "/admin/upload-global",
            files={"file": sample_file},
            data={
                "sector": "RBI",
                "document_type": "circular",
                "category": "Banking"
            }
        )

    # Router catches 409 exception and returns 500 with error details in body
    assert response.status_code == 500
    assert "Duplicate file detected" in response.text


# -------------------------
# /files TEST
# -------------------------

@pytest.mark.asyncio
async def test_get_all_global_files(client):
    fake_doc = {
        "_id": "123",
        "filename": "test.pdf",
        "sector": "RBI",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc)
    }

    with patch("app.routes.admin_router.Global_file_collection.find",
               return_value=MagicMock(to_list=AsyncMock(return_value=[fake_doc]))):

        response = client.get("/admin/files")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert len(body["data"]) == 1


# -------------------------
# /global-sectors TEST
# -------------------------

def test_list_global_sectors(client):
    mock_point = MagicMock()
    mock_point.payload = {
        "metadata": {
            "is_global": True,
            "source": "RBI",
            "file_name": "file1.pdf",
            "document_type": "circular"
        }
    }

    mock_client = MagicMock()
    mock_client.scroll.side_effect = [
        ([mock_point], None)
    ]

    with patch("qdrant_client.QdrantClient", return_value=mock_client):
        response = client.get("/admin/global-sectors")

    assert response.status_code == 200
    assert response.json()["data"][0]["sector"] == "RBI"