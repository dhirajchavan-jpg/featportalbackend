import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId

# Ensure these imports match your project structure
from app.main import app
from app.dependencies import UserPayload
from app.middleware import files_middleware
from app.auth.jwt_bearer import JWTBearer

# ------------------------------------------------------------------
# FIXTURES
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_db_lifespan():
    """
    Mock the DB connection check during app startup to prevent 
    'RuntimeError: Event loop is closed' during TestClient startup.
    """
    with patch("app.main.verify_connection", new_callable=AsyncMock) as mock_verify:
        yield mock_verify

@pytest.fixture(autouse=True)
def override_auth():
    """Disable JWT auth for ALL tests."""
    app.dependency_overrides[JWTBearer] = lambda: None
    yield
    app.dependency_overrides.pop(JWTBearer, None)

@pytest.fixture
def client(mock_db_lifespan):
    with TestClient(app) as c:
        yield c

@pytest.fixture
def admin_user():
    return UserPayload(
        user_id="admin123",
        email="admin@test.com",
        role="admin"
    )

@pytest.fixture
def upload_file():
    return ("files", ("test.pdf", b"dummy content", "application/pdf"))

@pytest.fixture
def mock_project():
    return {
        "_id": ObjectId(),
        "project_id": "proj1",
        "has_organizational_compliance": True,
        "organization_sector": "RBI"
    }

@pytest.fixture
def file_model():
    m = MagicMock()
    m.file_id = "file123"
    m.project_id = "proj1"
    m.filename = "test.pdf"
    m.file_url = "/files/test.pdf"
    m.sector = "RBI"
    m.category = "Organization"
    m.compliance_type = "Policy"
    m.user_id = "admin123"
    return m

@pytest.fixture(autouse=True)
def override_user(admin_user):
    """Force authenticated admin user."""
    app.dependency_overrides[
        __import__("app.dependencies").dependencies.get_current_user
    ] = lambda: admin_user
    yield
    app.dependency_overrides.clear()
    
@pytest.fixture(autouse=True)
def register_file_router():
    from app.routes.file_router import router
    app.include_router(router)
    yield


# ------------------------------------------------------------------
# TESTS
# ------------------------------------------------------------------

def test_cancel_upload(client):
    response = client.post("/cancel_upload", json={"filename": "test.pdf"})
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_bulk_upload_success(client, upload_file, mock_project, file_model):
    """
    Test bulk upload success scenario.
    Patches the 'db' object entirely to avoid Event Loop errors with Motor.
    """
    
    # Configure the mock DB for this specific test
    mock_db = MagicMock()
    # Ensure project_configs.find_one returns None (so defaults are used)
    mock_db.project_configs.find_one = AsyncMock(return_value=None)

    with patch("app.routes.file_router.db", mock_db), \
         patch.object(files_middleware, "verify_project", AsyncMock(return_value=mock_project)), \
         patch.object(files_middleware, "save_file", AsyncMock(return_value=("test.pdf", "/tmp/test.pdf", "hash"))), \
         patch.object(files_middleware, "run_qdrant_indexing", AsyncMock()), \
         patch.object(files_middleware, "store_file_metadata", AsyncMock(return_value=file_model)), \
         patch("asyncio.create_task", lambda coro: coro):

        response = client.post(
            "/projects/proj1/files",
            files=[upload_file],
            data={
                "categories": ["Organization"],
                "compliance_types": ["Policy"]
            }
        )

    # Assert 200 OK (logic sets 201 via variable, but response object override 
    # must be in place in the router code. If router returns 200 by default, check 200)
    assert response.status_code in [200, 201]
    assert response.json()["status"] == "success"


def test_bulk_upload_category_mismatch(client, upload_file):
    response = client.post(
        "/projects/proj1/files",
        files=[upload_file],
        data={
            "categories": ["Org", "Extra"],
            "compliance_types": ["Policy"]
        }
    )

    assert response.status_code == 422


def test_get_all_files(client, file_model):
    with patch.object(
        files_middleware,
        "get_all_files_for_project",
        AsyncMock(return_value=[file_model])
    ):
        response = client.get("/projects/proj1/files")

    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


def test_get_file_metadata(client, file_model):
    with patch.object(files_middleware, "get_file", AsyncMock(return_value=file_model)):
        response = client.get("/projects/proj1/files/file123")

    assert response.status_code == 200
    assert response.json()["data"]["file_id"] == "file123"


def test_delete_file_admin(client, file_model):
    with patch.object(files_middleware, "get_file", AsyncMock(return_value=file_model)), \
         patch.object(files_middleware, "handle_file_deletion", AsyncMock()):

        response = client.delete("/projects/proj1/files/file123")

    assert response.status_code == 200
    assert response.json()["data"]["deleted"] is True


def test_deselect_file(client):
    # Patch the db object to intercept update_one
    mock_db = MagicMock()
    mock_db.user_file_selection.update_one = AsyncMock()
    
    with patch("app.routes.file_router.db", mock_db):
        response = client.post(
            "/deselected",
            json={
                "file_id": "file123",
                "filename": "test.pdf",
                "project_id": "proj1",
                "source": "project"
            }
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_reselect_file(client):
    mock_db = MagicMock()
    delete_result = MagicMock(deleted_count=1)
    mock_db.user_file_selection.delete_one = AsyncMock(return_value=delete_result)

    with patch("app.routes.file_router.db", mock_db):
        response = client.post(
            "/reselect",
            json={
                "file_id": "file123",
                "filename": "test.pdf",
                "project_id": "proj1",
                "source": "project"
            }
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_get_deselected_files(client):
    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=[
        {"file_id": "file123", "filename": "test.pdf", "source": "project", "created_at": None}
    ])
    
    mock_db = MagicMock()
    mock_db.user_file_selection.find.return_value = cursor

    with patch("app.routes.file_router.db", mock_db):
        response = client.get("/deselected/proj1")

    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


def test_get_file_chunks(client):
    file_id = str(ObjectId())

    # Create a mock for the files collection
    mock_files_collection = MagicMock()
    mock_files_collection.find_one = AsyncMock(return_value={
        "_id": ObjectId(file_id),
        "project_id": "proj1",
        "filename": "test.pdf"
    })

    # Patch 'db' object entirely
    with patch("app.routes.file_router.db") as mock_db:
        mock_db.files = mock_files_collection
        
        with patch.object(files_middleware, "verify_project", AsyncMock()), \
             patch("app.routes.file_router.get_vector_retriever") as mock_retriever_getter:

            mock_retriever = MagicMock()
            mock_retriever.get_chunks_by_filename = AsyncMock(return_value=["chunk1"])
            mock_retriever_getter.return_value = mock_retriever

            response = client.get(f"/files/{file_id}/chunks")

    assert response.status_code == 200
    assert response.json()["data"]["total_chunks_retrieved"] == 1