# app/tests/models/test_user.py
import pytest
from pydantic import ValidationError
from app.models.user import User  # adjust import path as needed


# -------------------------------
# Test required fields in User model
# -------------------------------
def test_user_required_fields():
    """
    Summary:
        Ensures that all required fields (name, email, password) are enforced in the User model.
    
    Explanation:
        - Omitting any required field should raise a Pydantic ValidationError.
        - Verifies that the default role is 'user'.
        - Checks that optional field admin_id defaults to None.
    """
    # Missing name
    with pytest.raises(ValidationError):
        User(email="test@example.com", password="pass123")
    
    # Missing email
    with pytest.raises(ValidationError):
        User(name="John", password="pass123")
    
    # Missing password
    with pytest.raises(ValidationError):
        User(name="John", email="test@example.com")
    
    # All required fields provided
    user = User(name="John", email="test@example.com", password="pass123")
    assert user.name == "John"
    assert user.email == "test@example.com"
    assert user.password == "pass123"
    assert user.role == "user"  # default role
    assert user.admin_id is None


# -------------------------------
# Test custom role assignment
# -------------------------------
def test_user_role_custom_value():
    """
    Summary:
        Validates that the User model allows overriding the default role.
    
    Explanation:
        - By default, role is 'user'.
        - If a custom role (e.g., 'admin') is provided, it should be correctly assigned.
    """
    user = User(name="Alice", email="alice@test.com", password="password", role="admin")
    assert user.role == "admin"


# -------------------------------
# Test optional admin_id field
# -------------------------------
def test_user_admin_id_optional():
    """
    Summary:
        Ensures that admin_id is optional and can be set if needed.
    
    Explanation:
        - admin_id can link the user to a specific admin account.
        - If provided, it should be correctly stored.
    """
    user = User(name="Bob", email="bob@test.com", password="secret", admin_id="admin123")
    assert user.admin_id == "admin123"


# -------------------------------
# Test email validation
# -------------------------------
def test_user_email_validation():
    """
    Summary:
        Validates that the User model enforces proper email format.
    
    Explanation:
        - An invalid email string should raise a Pydantic ValidationError.
        - Ensures email data integrity for user accounts.
    """
    with pytest.raises(ValidationError):
        User(name="Jane", email="invalid-email", password="password")
