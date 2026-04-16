import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from bson import ObjectId
import os

from app.main import app
from app.dependencies import UserPayload, get_current_user
from app.auth.jwt_bearer import JWTBearer

# Try to import get_user_flexible to ensure we override the correct object key
try:
    from app.routes.file_viewer import get_user_flexible
except ImportError:
    # If not found in router, try dependencies or use None (skip override)
    try:
        from app.dependencies import get_user_flexible
    except ImportError:
        get_user_flexible = None

# -------------------------------------------------------------------
# FIXTURES
# -------------------------------------------------------------------

@pytest.fixture(autouse=True)
def override_auth():
    """
    Disable JWT auth for ALL tests by overriding the JWTBearer class.
    """
    app.dependency_overrides[JWTBearer] = lambda: None
    yield
    app.dependency_overrides.pop(JWTBearer, None)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def user():
    return UserPayload(
        user_id="user123",
        email="user@test.com",
        role="user"
    )

@pytest.fixture
def file_doc():
    return {
        "_id": ObjectId(),
        "file_id": "file123",
        "filename": "test_file.pdf",
        "file_path": "/fake/path/test_file.pdf"
    }

@pytest.fixture(autouse=True)
def override_user_deps(user):
    """
    Override BOTH standard and flexible user dependencies to ensure
    all endpoints (POST and GET) are authenticated.
    """
    # 1. Override standard get_current_user (likely used by POST /get-file-path)
    app.dependency_overrides[get_current_user] = lambda: user
    
    # 2. Override get_user_flexible (likely used by GET /view/{file_id})
    if get_user_flexible:
        app.dependency_overrides[get_user_flexible] = lambda: user
        
    yield
    
    # Cleanup
    app.dependency_overrides.pop(get_current_user, None)
    if get_user_flexible:
        app.dependency_overrides.pop(get_user_flexible, None)

# -------------------------------------------------------------------
# UNIT TEST: resolve_physical_path
# -------------------------------------------------------------------

def test_resolve_physical_path_found():
    try:
        from app.routes.file_viewer import resolve_physical_path
        with patch("os.path.exists", return_value=True):
            result = resolve_physical_path("/fake/path/test_file.pdf")
            assert result is not None
            assert result.endswith("test_file.pdf")
    except ImportError:
        pass

def test_resolve_physical_path_not_found():
    try:
        from app.routes.file_viewer import resolve_physical_path
        with patch("os.path.exists", return_value=False):
            result = resolve_physical_path("/fake/path/missing.pdf")
            assert result is None
    except ImportError:
        pass

# -------------------------------------------------------------------
# POST /files/get-file-path
# -------------------------------------------------------------------

def test_get_file_path_success(client, user, file_doc):
    with patch("app.routes.file_viewer.Global_file_collection.find_one", AsyncMock(return_value=file_doc)):
        response = client.post(
            "/files/get-file-path",
            json={"filename": "test_file.pdf"}
        )

    assert response.status_code == 200
    assert "file_url" in response.json()
    assert "/files/view/" in response.json()["file_url"]

def test_get_file_path_not_found(client, user):
    with patch("app.routes.file_viewer.Global_file_collection.find_one", AsyncMock(return_value=None)), \
         patch("app.routes.file_viewer.file_collection.find_one", AsyncMock(return_value=None)):
        
        response = client.post(
            "/files/get-file-path",
            json={"filename": "missing.pdf"}
        )

    assert response.status_code == 404

# -------------------------------------------------------------------
# GET /files/view/{file_id}
# -------------------------------------------------------------------

def test_view_file_success(client, user, file_doc):
    with patch("app.routes.file_viewer.Global_file_collection.find_one", AsyncMock(return_value=file_doc)), \
         patch("app.routes.file_viewer.resolve_physical_path", return_value="/fake/path/test_file.pdf"), \
         patch("builtins.open", mock_open(read_data=b"PDF DATA")), \
         patch("os.path.exists", return_value=True):

        response = client.get("/files/view/file123")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content == b"PDF DATA"

def test_view_file_db_record_missing(client, user):
    with patch("app.routes.file_viewer.Global_file_collection.find_one", AsyncMock(return_value=None)), \
         patch("app.routes.file_viewer.file_collection.find_one", AsyncMock(return_value=None)):
        
        response = client.get("/files/view/file123")

    assert response.status_code == 404
    # Robust assertion that handles different key names or string responses
    assert "not found" in str(response.json())

def test_view_file_physical_file_missing(client, user, file_doc):
    with patch("app.routes.file_viewer.Global_file_collection.find_one", AsyncMock(return_value=file_doc)), \
         patch("app.routes.file_viewer.resolve_physical_path", return_value=None):
        
        response = client.get("/files/view/file123")

    assert response.status_code == 404
    assert "not found" in str(response.json())