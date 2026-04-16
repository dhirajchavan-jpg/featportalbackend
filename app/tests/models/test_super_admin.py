# app/tests/models/test_super_admin.py
import pytest
from pydantic import ValidationError
import re
from app.models.super_admin import (
    SuperAdminLogin,
    OrgAdminCreate,
    OrgAdminUpdate,
    OrgAdminResponse,
    DashboardStats
)

# -------------------------------
# Tests for SuperAdminLogin
# -------------------------------

def test_super_admin_login_required_fields():
    """
    Summary:
        Validates that the SuperAdminLogin model requires both username and password.
    
    Explanation:
        - A login attempt without a password should raise a ValidationError.
        - When both username and password are provided, the model initializes correctly.
        - Ensures data integrity for login operations.
    """
    # Missing password should raise
    with pytest.raises(ValidationError):
        SuperAdminLogin(username="admin")  

    # Valid login should pass
    login = SuperAdminLogin(username="admin", password="pass123")
    assert login.username == "admin"


# -------------------------------
# Tests for OrgAdminCreate
# -------------------------------

def test_org_admin_create_email_validation():
    """
    Summary:
        Ensures that OrgAdminCreate enforces proper email format.
    
    Explanation:
        - Invalid email strings should raise a ValidationError.
        - Valid emails are accepted.
        - Uses a simple regex check to optionally validate email format.
    """
    # Invalid email should raise
    with pytest.raises(ValidationError):
        OrgAdminCreate(username="user", email="invalid", password="1234", organization_name="Org")
    
    # Valid email should work
    admin = OrgAdminCreate(username="user", email="user@test.com", password="1234", organization_name="Org")
    assert admin.email == "user@test.com"
    assert re.match(r"[^@]+@[^@]+\.[^@]+", admin.email)


# -------------------------------
# Tests for OrgAdminUpdate
# -------------------------------

def test_org_admin_update_optional_fields():
    """
    Summary:
        Tests that all fields in OrgAdminUpdate are optional.
    
    Explanation:
        - OrgAdminUpdate is used for partial updates, so fields should default to None.
        - Ensures the model can be instantiated with zero or only some fields.
    """
    update = OrgAdminUpdate()
    assert update.username is None

    update2 = OrgAdminUpdate(email="test@test.com")
    assert update2.email == "test@test.com"


# -------------------------------
# Tests for OrgAdminResponse
# -------------------------------

def test_org_admin_response_defaults_and_alias():
    """
    Summary:
        Checks default values and alias handling in OrgAdminResponse.
    
    Explanation:
        - _id field should map to id attribute.
        - organization_name defaults to "N/A" if not provided.
        - is_active defaults and type correctness are validated.
    """
    response = OrgAdminResponse(_id="123", username="u", email="a@b.com", role="admin", is_active=True)
    assert response.id == "123"
    assert response.organization_name == "N/A"
    assert response.is_active is True


# -------------------------------
# Tests for DashboardStats
# -------------------------------

def test_dashboard_stats():
    """
    Summary:
        Validates that DashboardStats correctly assigns numeric statistics.
    
    Explanation:
        - total_admins, total_users, total_files are required fields.
        - Ensures that stats object is correctly initialized for dashboard display.
    """
    stats = DashboardStats(total_admins=5, total_users=10, total_files=20)
    assert stats.total_admins == 5
    assert stats.total_users == 10
    assert stats.total_files == 20
