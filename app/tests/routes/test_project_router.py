import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId

# Adjust import to match your file structure
from app.routes.project_router import router
from app.dependencies import get_current_user, UserPayload
from app.middleware.role_checker import require_roles
from app.auth.jwt_bearer import JWTBearer

# ------------------------------------------------------------------
# HELPER: Async Iterator for MongoDB Cursors
# ------------------------------------------------------------------
class AsyncIterator:
    """
    Helper to mock Motor/MongoDB async cursors.
    Allows 'async for' iteration over a list of items.
    """
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

# ------------------------------------------------------------------
# FIXTURES (UPDATED WITH VALID IDs)
# ------------------------------------------------------------------

@pytest.fixture
def admin_user():
    # FIXED: Generate a real 24-char hex ID
    return UserPayload(user_id=str(ObjectId()), email="admin@test.com", role="admin")

@pytest.fixture
def regular_user():
    # FIXED: Generate a real 24-char hex ID
    return UserPayload(user_id=str(ObjectId()), email="user@test.com", role="user")

@pytest.fixture
def app_instance(admin_user):
    """Create a fresh FastAPI app with overrides."""
    app = FastAPI()
    app.include_router(router)

    # Default override: Admin User
    app.dependency_overrides[get_current_user] = lambda: admin_user
    
    # Mock Role Checker to always accept
    def mock_require_roles(*roles):
        return lambda: admin_user
    
    app.dependency_overrides[require_roles] = mock_require_roles
    
    # Bypass JWT
    app.dependency_overrides[JWTBearer] = lambda: "fake_token"
    
    return app

@pytest.fixture
def client(app_instance):
    return TestClient(app_instance)

@pytest.fixture(autouse=True)
def mock_db(admin_user):
    """
    Patches DB collections globally.
    Returns a dict to access mocks inside tests.
    """
    with patch("app.routes.project_router.project_collection") as project_col, \
         patch("app.routes.project_router.project_config_collection") as config_col, \
         patch("app.routes.project_router.User_collection") as user_col:

        # 1. Setup Defaults
        project_col.insert_one = AsyncMock(return_value=MagicMock(inserted_id=ObjectId()))
        project_col.find_one = AsyncMock(return_value=None)
        project_col.update_one = AsyncMock(return_value=MagicMock(upserted_id=None, deleted_count=1))
        project_col.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
        
        # Default Cursor: Empty list
        project_col.find.return_value = AsyncIterator([])

        config_col.update_one = AsyncMock(return_value=MagicMock(upserted_id=None))
        config_col.find_one = AsyncMock(return_value=None)

        # Default User query returns the fixture admin ID
        user_col.find_one = AsyncMock(return_value={"admin_id": admin_user.user_id})

        yield {
            "project": project_col,
            "config": config_col,
            "user": user_col
        }

# ------------------------------------------------------------------
# TESTS
# ------------------------------------------------------------------

def test_create_project_success(client):
    payload = {
        "project_name": "Compliance Alpha",
        "description": "A valid description for testing.",
        "industry": "Finance",
        "regulatory_framework": "RBI",
        "third_party_framework": ["SOC2"],
        "has_organizational_compliance": False
    }

    # Mock validators to avoid Pydantic/Regex overhead in unit tests
    with patch("app.routes.project_router.validate_project_name"), \
         patch("app.routes.project_router.validate_description"), \
         patch("app.routes.project_router.validate_industry", return_value="FINANCE"), \
         patch("app.routes.project_router.validate_sectors", return_value=["RBI", "SOC2"]):

        res = client.post("/new_project", json=payload)

    assert res.status_code == 200
    assert res.json()["status"] == "success"
    data = res.json()["data"]
    assert "RBI" in data["sectors"]


def test_create_project_sector_merging_logic(client, mock_db):
    """
    Tests the logic that generates the specific 'Organization Sector ID'
    and merges it with Regulatory and Third Party.
    """
    payload = {
        "project_name": "Org Project",
        "description": "Desc...",
        "industry": "Tech",
        "regulatory_framework": "GDPR",
        "third_party_framework": ["ISO27001"],
        "has_organizational_compliance": True,
        "organization_sector": "MyCompany" 
    }

    with patch("app.routes.project_router.validate_project_name"), \
         patch("app.routes.project_router.validate_description"), \
         patch("app.routes.project_router.validate_industry", return_value="TECH"), \
         patch("app.routes.project_router.validate_sectors", return_value=["GDPR", "ISO27001"]):

        res = client.post("/new_project", json=payload)

    assert res.status_code == 200
    
    args, _ = mock_db["project"].insert_one.call_args
    inserted_data = args[0]
    
    sectors = inserted_data["sectors"]
    assert "GDPR" in sectors
    assert "ISO27001" in sectors
    
    org_sector = inserted_data["organization_sector"]
    assert org_sector.startswith("MYCOMPANY_")
    assert org_sector in sectors


def test_create_project_all_empty_fails(client):
    """
    Validation should fail if Reg, Third Party, and Org compliance are ALL empty.
    """
    payload = {
        "project_name": "Empty Project",
        "description": "Desc...",
        "industry": "Tech",
        "regulatory_framework": "None",
        "third_party_framework": [],
        "has_organizational_compliance": False
    }

    with patch("app.routes.project_router.validate_project_name"), \
         patch("app.routes.project_router.validate_description"), \
         patch("app.routes.project_router.validate_industry"):
         
        res = client.post("/new_project", json=payload)

    assert res.status_code == 400
    assert "must select at least one" in res.json()["detail"]


# -------------------- SETTINGS TESTS --------------------

def test_update_project_settings_patch(client, mock_db, admin_user):
    project_id = str(ObjectId())
    # Mock ownership check
    mock_db["project"].find_one.return_value = {"_id": ObjectId(project_id), "user_id": admin_user.user_id}

    res = client.patch(f"/{project_id}/settings", json={"retrieval_depth": 8})
    
    assert res.status_code == 200
    mock_db["config"].update_one.assert_called_once()


def test_update_settings_forbidden(client, mock_db):
    project_id = str(ObjectId())
    # Mock ownership check returning None (project doesn't belong to user)
    mock_db["project"].find_one.return_value = None

    res = client.patch(f"/{project_id}/settings", json={"retrieval_depth": 8})
    
    assert res.status_code == 403
    assert "Access denied" in res.json()["detail"]


# -------------------- GET PROJECTS TESTS --------------------

def test_get_my_projects_admin(client, mock_db, admin_user):
    """Admin should see their own projects."""
    projects = [{"_id": ObjectId(), "project_name": "Admin Proj", "user_id": admin_user.user_id}]
    mock_db["project"].find.return_value = AsyncIterator(projects)

    res = client.get("/my_projects")

    assert res.status_code == 200
    assert len(res.json()["data"]) == 1
    mock_db["project"].find.assert_called_with({"user_id": admin_user.user_id})


def test_get_my_projects_user(client, app_instance, mock_db, regular_user):
    """Regular User should see their Admin's projects."""
    
    # 1. Override dependency to be a regular user
    app_instance.dependency_overrides[get_current_user] = lambda: regular_user
    
    # 2. Create a fake admin ID (must be string for consistency, though ID format matters less here unless query uses it)
    fake_admin_id = str(ObjectId())

    # 3. Mock User Collection to return the linked Admin ID
    # CRITICAL: This ensures when router does User_collection.find_one, it gets this result
    mock_db["user"].find_one.return_value = {"_id": ObjectId(), "admin_id": fake_admin_id}

    # 4. Mock Project Collection to return projects belonging to "fake_admin_id"
    projects = [{"_id": ObjectId(), "project_name": "Boss Project", "user_id": fake_admin_id}]
    mock_db["project"].find.return_value = AsyncIterator(projects)

    res = client.get("/my_projects")

    assert res.status_code == 200
    assert res.json()["data"][0]["project_name"] == "Boss Project"
    
    # 5. Verify the query was made for the ADMIN's ID
    mock_db["project"].find.assert_called_with({"user_id": fake_admin_id})


# -------------------- SINGLE PROJECT TESTS --------------------

def test_get_project_success(client, mock_db, admin_user):
    project_id = str(ObjectId())
    mock_db["project"].find_one.return_value = {
        "_id": ObjectId(project_id),
        "user_id": admin_user.user_id,
        "project_name": "Alpha",
        "has_organizational_compliance": True
    }

    res = client.get(f"/project/{project_id}")
    
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["organizational_compliance"] == "Yes"


def test_delete_project_success(client, mock_db, admin_user):
    project_id = str(ObjectId())
    mock_db["project"].find_one.return_value = {"_id": ObjectId(project_id), "user_id": admin_user.user_id}
    
    res = client.delete(f"/project/{project_id}")
    
    assert res.status_code == 200
    mock_db["project"].delete_one.assert_called_once()