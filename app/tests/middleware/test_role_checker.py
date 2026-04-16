import pytest
from fastapi import HTTPException, status

from app.middleware.role_checker import require_roles

# -------------------------
# Role Checker Middleware Tests
# -------------------------

class DummyUser:
    """
    Dummy user object to simulate `current_user` in role checks.
    """
    def __init__(self, role: str):
        self.role = role


def test_require_roles_allows_valid_role():
    """
    Summary:
        Tests that a user with an allowed role passes the role check.

    Explanation:
        The `require_roles` function returns a checker function that validates
        the current user's role against allowed roles. This test ensures that
        a user with a role included in the allowed list is correctly returned
        without errors.
    """
    checker = require_roles("admin", "superadmin")
    user = DummyUser(role="admin")

    result = checker(current_user=user)

    assert result == user


def test_require_roles_case_insensitive():
    """
    Summary:
        Validates that role checking is case-insensitive.

    Explanation:
        Ensures that role names like "ADMIN" or "admin" are treated equally.
        This prevents accidental authorization failures due to casing mismatches.
    """
    checker = require_roles("admin")
    user = DummyUser(role="ADMIN")

    result = checker(current_user=user)

    assert result == user


def test_require_roles_denies_invalid_role():
    """
    Summary:
        Confirms that a user with an invalid role is denied access.

    Explanation:
        Users whose roles are not included in the allowed roles list
        should trigger an HTTPException with status 403 (Forbidden).
        This test ensures the proper exception is raised with correct status.
    """
    checker = require_roles("admin", "superadmin")
    user = DummyUser(role="user")

    with pytest.raises(HTTPException) as exc:
        checker(current_user=user)

    assert exc.value.status_code == status.HTTP_403_FORBIDDEN
    assert "Access forbidden" in exc.value.detail


def test_require_roles_single_role():
    """
    Summary:
        Validates that role checking works when only one role is allowed.

    Explanation:
        Ensures that a single-role whitelist functions correctly.
        The user with the exact allowed role should be returned successfully.
    """
    checker = require_roles("user")
    user = DummyUser(role="user")

    assert checker(current_user=user) == user


def test_require_roles_error_message_contains_allowed_roles():
    """
    Summary:
        Ensures the error message shows all allowed roles.

    Explanation:
        When a user fails the role check, the exception detail should
        provide information about the allowed roles to help with debugging
        or client messaging. This test verifies that the allowed roles
        appear in the exception message.
    """
    checker = require_roles("admin", "manager")
    user = DummyUser(role="guest")

    with pytest.raises(HTTPException) as exc:
        checker(current_user=user)

    assert "('admin', 'manager')" in exc.value.detail
