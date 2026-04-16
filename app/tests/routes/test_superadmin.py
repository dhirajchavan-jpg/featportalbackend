import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId
from io import BytesIO

# Router import
from app.routes.super_admin import router
from app.dependencies import UserPayload

# =========================================================
# APP SETUP & FIXTURES
# =========================================================

@pytest.fixture
def super_admin_user():
    """
    Returns a proper Pydantic model so it is JSON serializable
    (required for background tasks/Redis).
    """
    return UserPayload(
        user_id="super123",
        email="super@test.com",
        role="super_admin"
    )

@pytest.fixture
def app(super_admin_user):
    app = FastAPI()
    app.include_router(router) 

    # Import dependencies to override
    from app.routes.super_admin import get_current_user, verify_super_admin

    # 1. Override basic user injection
    app.dependency_overrides[get_current_user] = lambda: super_admin_user
    
    # 2. Override Super Admin check directly
    # This bypasses JWTBearer() logic entirely for endpoints requiring super admin
    app.dependency_overrides[verify_super_admin] = lambda: {
        "user_id": super_admin_user.user_id,
        "role": "super_admin",
        "sub": "superadmin"
    }

    yield app
    app.dependency_overrides = {}

@pytest.fixture
def client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

# =========================================================
# GLOBAL MOCKS
# =========================================================

@pytest.fixture(autouse=True)
def mock_globals():
    """
    Patches all external services used by the Super Admin router.
    Autouse ensures these are active for all tests in this file.
    """
    with patch("app.routes.super_admin.create_access_token", return_value="fake-token"), \
         patch("app.routes.super_admin.super_admin_collection") as sa_col, \
         patch("app.routes.super_admin.User_collection") as user_col, \
         patch("app.routes.super_admin.file_collection") as file_col, \
         patch("app.routes.super_admin.Global_file_collection") as global_file_col, \
         patch("app.routes.super_admin.magic.from_buffer", return_value="application/pdf"), \
         patch("app.routes.super_admin.process_and_index_global_file", return_value={"chunks_indexed": 5}), \
         patch("app.routes.super_admin.delete_global_document_by_source", new_callable=AsyncMock), \
         patch("app.routes.super_admin.redis_service") as mock_redis, \
         patch("app.routes.super_admin.get_ocr_engine") as mock_get_ocr:  # <--- Added Patch

        # 1. Setup Collection Counts
        sa_col.count_documents = AsyncMock(return_value=2)
        user_col.count_documents = AsyncMock(return_value=5)
        file_col.count_documents = AsyncMock(return_value=10)

        # 2. Setup Global File Collection
        global_file_col.find_one = AsyncMock(return_value=None)
        global_file_col.insert_one = AsyncMock()
        global_file_col.delete_one = AsyncMock()
        
        # 3. Setup Default Cursor for find()
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        global_file_col.find = MagicMock(return_value=mock_cursor)

        # 4. Redis Enqueue
        mock_redis.enqueue_job = AsyncMock(return_value="job_123")

        # 5. Mock OCR Engine to be available
        mock_ocr_instance = MagicMock()
        mock_ocr_instance.is_service_available.return_value = True
        mock_get_ocr.return_value = mock_ocr_instance

        yield {
            "global_col": global_file_col,
            "sa_col": sa_col,
            "cursor": mock_cursor
        }

# =========================================================
# TESTS
# =========================================================

@pytest.mark.skip(reason="Login endpoint is currently commented out in source code")
@pytest.mark.asyncio
async def test_super_admin_login(client, mock_globals):
    # Configure finding the admin user
    mock_globals["sa_col"].find_one = AsyncMock(return_value={
        "_id": ObjectId(),
        "username": "admin",
        "password": "$2b$12$hashed",
        "role": "super_admin"
    })

    # Mock password verification
    with patch("app.routes.super_admin.verify_password", return_value=True):
        res = await client.post("/login", json={
            "username": "admin",
            "password": "secret"
        })

    assert res.status_code == 200
    assert res.json()["access_token"] == "fake-token"

@pytest.mark.asyncio
async def test_dashboard_stats(client):
    res = await client.get("/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["total_admins"] == 2
    assert data["total_users"] == 5
    assert data["total_files"] == 10

@pytest.mark.asyncio
async def test_upload_global_file(client):
    # Using %PDF header to satisfy magic check if patch fails (fallback)
    fake_file = BytesIO(b"%PDF-1.4 fake content")
    fake_file.name = "test.pdf"

    res = await client.post(
        "/upload-global",
        data={
            "sector": "RBI",
            "document_type": "circular",
            "category": "banking"
        },
        files={"file": ("test.pdf", fake_file, "application/pdf")}
    )

    # Expect 202 Accepted as defined in your route code
    assert res.status_code == 202
    json_resp = res.json()
    assert json_resp["status"] == "success"
    # Ensure Redis task ID is returned
    assert json_resp["data"]["task_id"] == "job_123"

@pytest.mark.asyncio
async def test_get_all_global_files(client, mock_globals):
    # Setup cursor to return one specific file
    fake_doc = {
        "_id": ObjectId(), 
        "filename": "doc.pdf",
        "sector": "RBI",
        "created_at": None,
        "updated_at": None
    }
    
    # Configure the cursor mock from the global fixture
    mock_globals["cursor"].to_list.return_value = [fake_doc]

    res = await client.get("/files")

    assert res.status_code == 200
    assert res.json()["status"] == "success"
    assert len(res.json()["data"]) == 1
    assert res.json()["data"][0]["filename"] == "doc.pdf"

@pytest.mark.asyncio
async def test_delete_global_file(client, mock_globals):
    # Setup find_one to return a file so deletion proceeds
    mock_globals["global_col"].find_one = AsyncMock(return_value={
        "_id": ObjectId(),
        "filename": "doc.pdf",
        "sector": "RBI",
        "file_path": "/tmp/doc.pdf"
    })

    # Mock file system removal just in case
    with patch("os.remove") as mock_remove, \
         patch("os.path.exists", return_value=True):
        
        res = await client.delete("/global/RBI/doc.pdf")

    assert res.status_code == 200
    assert res.json()["status"] == "success"