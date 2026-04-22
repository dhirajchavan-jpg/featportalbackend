import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, mock_open
from bson import ObjectId

from app.main import app
from app.dependencies import UserPayload, get_current_user
from app.auth.jwt_bearer import JWTBearer

try:
    from app.routes.file_viewer import get_user_flexible
except ImportError:
    get_user_flexible = None


@pytest.fixture(autouse=True)
def override_auth():
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
        role="user",
    )


@pytest.fixture
def file_doc():
    return {
        "_id": ObjectId(),
        "file_id": "file123",
        "filename": "test_file.pdf",
        "file_path": "/fake/path/test_file.pdf",
    }


@pytest.fixture(autouse=True)
def override_user_deps(user):
    app.dependency_overrides[get_current_user] = lambda: user
    if get_user_flexible:
        app.dependency_overrides[get_user_flexible] = lambda: user

    yield

    app.dependency_overrides.pop(get_current_user, None)
    if get_user_flexible:
        app.dependency_overrides.pop(get_user_flexible, None)


def test_resolve_physical_path_found():
    from app.routes.file_viewer import resolve_physical_path

    with patch("os.path.exists", return_value=True):
        result = resolve_physical_path("/fake/path/test_file.pdf")

    assert result is not None
    assert result.endswith("test_file.pdf")


def test_resolve_physical_path_not_found():
    from app.routes.file_viewer import resolve_physical_path

    with patch("os.path.exists", return_value=False), patch("os.path.isdir", return_value=False):
        result = resolve_physical_path("/fake/path/missing.pdf")

    assert result is None


def test_get_file_path_success(client, file_doc):
    with patch("app.routes.file_viewer.Global_file_collection.find_one", AsyncMock(return_value=file_doc)):
        response = client.post(
            "/files/get-file-path",
            json={"filename": "test_file.pdf"},
            headers={"Authorization": "Bearer token123"},
        )

    assert response.status_code == 200
    assert response.json()["file_url"] == "http://122.170.2.205:8072/files/view/file123?token=token123"
    assert response.json()["file_type"] == ".pdf"
    assert response.json()["filename"] == "test_file.pdf"


def test_get_file_path_not_found(client):
    with patch("app.routes.file_viewer.Global_file_collection.find_one", AsyncMock(return_value=None)), patch(
        "app.routes.file_viewer.file_collection.find_one", AsyncMock(return_value=None)
    ):
        response = client.post(
            "/files/get-file-path",
            json={"filename": "missing.pdf"},
            headers={"Authorization": "Bearer token123"},
        )

    assert response.status_code == 404


def test_view_file_success(client, file_doc):
    with patch("app.routes.file_viewer.Global_file_collection.find_one", AsyncMock(return_value=file_doc)), patch(
        "app.routes.file_viewer.resolve_physical_path", return_value="/fake/path/test_file.pdf"
    ), patch("builtins.open", mock_open(read_data=b"PDF DATA")):
        response = client.get("/files/view/file123?token=token123")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == 'inline; filename="test_file.pdf"'
    assert response.content == b"PDF DATA"


def test_view_file_docx_uses_docx_mime_and_attachment(client):
    file_doc = {
        "_id": ObjectId(),
        "file_id": "file456",
        "filename": "handbook.docx",
        "file_path": "/fake/path/handbook.docx",
    }

    with patch("app.routes.file_viewer.Global_file_collection.find_one", AsyncMock(return_value=file_doc)), patch(
        "app.routes.file_viewer.resolve_physical_path", return_value="/fake/path/handbook.docx"
    ), patch("builtins.open", mock_open(read_data=b"DOCX DATA")):
        response = client.get("/files/view/file456?token=token123")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert response.headers["content-disposition"] == 'attachment; filename="handbook.docx"'
    assert response.content == b"DOCX DATA"


def test_view_file_uses_path_extension_when_db_filename_has_no_extension(client):
    file_doc = {
        "_id": ObjectId(),
        "file_id": "file789",
        "filename": "handbook",
        "file_path": "/fake/path/handbook.docx",
    }

    with patch("app.routes.file_viewer.Global_file_collection.find_one", AsyncMock(return_value=file_doc)), patch(
        "app.routes.file_viewer.resolve_physical_path", return_value="/fake/path/handbook.docx"
    ), patch("builtins.open", mock_open(read_data=b"DOCX DATA")):
        response = client.get("/files/view/file789?token=token123")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert response.headers["content-disposition"] == 'attachment; filename="handbook.docx"'


def test_view_file_db_record_missing(client):
    with patch("app.routes.file_viewer.Global_file_collection.find_one", AsyncMock(return_value=None)), patch(
        "app.routes.file_viewer.file_collection.find_one", AsyncMock(return_value=None)
    ):
        response = client.get("/files/view/file123?token=token123")

    assert response.status_code == 404
    assert "not found" in str(response.json()).lower()


def test_view_file_physical_file_missing(client, file_doc):
    with patch("app.routes.file_viewer.Global_file_collection.find_one", AsyncMock(return_value=file_doc)), patch(
        "app.routes.file_viewer.resolve_physical_path", return_value=None
    ):
        response = client.get("/files/view/file123?token=token123")

    assert response.status_code == 404
    assert "not found" in str(response.json()).lower()
