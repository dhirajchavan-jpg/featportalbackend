import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from bson import ObjectId

# Import your router and dependencies
from app.routes.User_router import router, User
from app.dependencies import UserPayload, get_current_user
from app.middleware.role_checker import require_roles

# -------------------------
# HELPER FOR ASYNC ITERATION
# -------------------------
class AsyncIterator:
    """Helper to mock MongoDB async cursors"""
    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        self.iter = iter(self.items)
        return self

    async def __anext__(self):
        try:
            return next(self.iter)
        except StopIteration:
            raise StopAsyncIteration

# -------------------------
# FIXTURES
# -------------------------

@pytest.fixture
def admin_user():
    return UserPayload(
        user_id=str(ObjectId()),
        email="admin@test.com",
        role="admin"
    )

@pytest.fixture
def app(admin_user):
    app = FastAPI()
    app.include_router(router)

    # Dependency overrides
    app.dependency_overrides[get_current_user] = lambda: admin_user
    # We mock the role checker to always allow the admin_user
    # Note: In real app, require_roles returns a callable dependency
    app.dependency_overrides[require_roles("admin")] = lambda: admin_user

    return app

@pytest.fixture
def client(app):
    return TestClient(app)

# -------------------------
# GLOBAL DB & SERVICE MOCKS
# -------------------------

@pytest.fixture(autouse=True)
def mock_db():
    """
    Patches ALL database collections and external services used in User_router.
    """
    with patch("app.routes.User_router.User_collection") as user_col, \
         patch("app.routes.User_router.project_collection") as project_col, \
         patch("app.routes.User_router.file_collection") as file_col, \
         patch("app.routes.User_router.cache_collection") as cache_col, \
         patch("app.routes.User_router.chat_history_collection") as chat_col, \
         patch("app.routes.User_router.user_file_selection_collection") as select_col, \
         patch("app.routes.User_router.get_qdrant_client") as qdrant_client_mock, \
         patch("app.routes.User_router.os") as mock_os, \
         patch("app.routes.User_router.create_access_token", return_value="fake-jwt-token"), \
         patch("app.routes.User_router.validate_name", return_value=None), \
         patch("app.routes.User_router.validate_email", return_value=None), \
         patch("app.routes.User_router.validate_password", return_value=None), \
         patch("app.routes.User_router.validate_role", return_value=None):

        # --- Setup Default Mock Behaviors ---
        
        # 1. Mongo Collections
        user_col.find_one = AsyncMock(return_value=None)
        user_col.insert_one = AsyncMock(return_value=MagicMock(inserted_id=ObjectId()))
        user_col.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
        
        # 2. Delete Many responses
        delete_response = MagicMock(deleted_count=1)
        project_col.delete_many = AsyncMock(return_value=delete_response)
        file_col.delete_many = AsyncMock(return_value=delete_response)
        cache_col.delete_many = AsyncMock(return_value=delete_response)
        chat_col.delete_many = AsyncMock(return_value=delete_response)
        select_col.delete_many = AsyncMock(return_value=delete_response)

        # 3. Find Cursors (Default to empty)
        user_col.find.return_value = AsyncIterator([])
        project_col.find.return_value = AsyncIterator([])
        file_col.find.return_value = AsyncIterator([])

        # 4. Qdrant
        qdrant_client_mock.return_value = MagicMock(delete=MagicMock())

        # 5. OS (Physical File Deletion)
        mock_os.path.exists.return_value = True
        mock_os.remove = MagicMock()

        yield

# -------------------------
# TESTS
# -------------------------

def test_register_user(client):
    user_data = {
        "name": "Test User",
        "email": "testuser@test.com",
        "password": "password123",
        "role": "user"
    }
    res = client.post("/register", json=user_data)
    assert res.status_code == 200
    assert res.json()["data"]["email"] == "testuser@test.com"

def test_login_user(client):
    # Mock finding a user
    with patch("app.routes.User_router.User_collection.find_one", new_callable=AsyncMock) as mock_find:
        mock_find.return_value = {
            "_id": ObjectId(),
            "email": "user@test.com",
            "password": "$2b$12$hashed",
            "role": "user",
            "name": "User"
        }
        with patch("app.routes.User_router.pwd_context.verify", return_value=True):
            res = client.post(
                "/login",
                json={"email": "user@test.com", "password": "password123"}
            )

    assert res.status_code == 200
    assert res.json()["data"]["access_token"] == "fake-jwt-token"

def test_refresh_token(client):
    res = client.post("/refresh")
    assert res.status_code == 200
    assert res.json()["data"]["access_token"] == "fake-jwt-token"

def test_delete_user_me(client, admin_user):
    """
    Tests the comprehensive deletion logic including physical files.
    """
    # 1. Mock Projects found
    project_id = str(ObjectId())
    
    # 2. Mock Files found
    file_doc = {"_id": ObjectId(), "project_id": project_id, "file_path": "/tmp/testfile.pdf"}

    # We need to explicitly patch the collections' .find() methods to return data
    # so the loop runs and triggers os.remove
    with patch("app.routes.User_router.project_collection.find", return_value=AsyncIterator([{"_id": project_id}])), \
         patch("app.routes.User_router.file_collection.find", return_value=AsyncIterator([file_doc])), \
         patch("app.routes.User_router.os.remove") as mock_remove:
         
        res = client.delete("/me")

        assert res.status_code == 200
        data = res.json()["data"]
        assert data["deleted_user_id"] == admin_user.user_id
        
        # Verify physical file deletion was attempted
        mock_remove.assert_called_with("/tmp/testfile.pdf")

def test_org_register(client, admin_user):
    """
    Test admin creating a user.
    """
    user_data = {
        "name": "Org Employee",
        "email": "employee@org.com",
        "password": "password123"
    }
    
    res = client.post("/org/register", json=user_data)
    
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["role"] == "user"
    assert data["created_by_admin"] == admin_user.user_id

def test_admin_delete_user(client, admin_user):
    """
    Test admin deleting a user they created.
    """
    target_user_id = str(ObjectId())
    
    # Mock finding the target user with the correct admin_id link
    target_user_doc = {
        "_id": ObjectId(target_user_id),
        "email": "employee@org.com",
        "role": "user",
        "admin_id": admin_user.user_id # Matches the logged in admin
    }

    with patch("app.routes.User_router.User_collection.find_one", new_callable=AsyncMock) as mock_find_one:
        mock_find_one.return_value = target_user_doc
        
        res = client.delete(f"/org/delete/{target_user_id}")

    assert res.status_code == 200
    assert res.json()["data"]["deleted_user_id"] == target_user_id

def test_admin_delete_user_forbidden(client, admin_user):
    """
    Test admin trying to delete a user they did NOT create.
    """
    target_user_id = str(ObjectId())
    
    # Mock finding a user created by SOMEONE ELSE
    target_user_doc = {
        "_id": ObjectId(target_user_id),
        "email": "other@org.com",
        "role": "user",
        "admin_id": "DIFFERENT_ADMIN_ID" 
    }

    with patch("app.routes.User_router.User_collection.find_one", new_callable=AsyncMock) as mock_find_one:
        mock_find_one.return_value = target_user_doc
        
        res = client.delete(f"/org/delete/{target_user_id}")

    assert res.status_code == 403
    assert "cannot delete users you did not create" in res.json()["detail"]["message"]