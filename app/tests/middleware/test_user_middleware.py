import pytest
from fastapi import HTTPException
from fastapi import FastAPI

from app.middleware.user_middleware import (
    validate_name,
    validate_email,
    validate_password,
    validate_role,
    setup_cors,
)

# -------------------------
# User Middleware Tests
# -------------------------

def test_validate_name_valid():
    """
    Summary:
        Ensures that valid names pass validation without raising exceptions.

    Explanation:
        The validate_name function checks that names contain only letters
        and spaces and are of reasonable length. Names like 'John Doe'
        should pass cleanly.
    """
    validate_name("John Doe")  # should not raise


def test_validate_name_too_short():
    """
    Summary:
        Ensures that names shorter than 2 characters are rejected.

    Explanation:
        Short names like "J" are invalid. The function should raise an
        HTTPException with status 400 and a descriptive message.
    """
    with pytest.raises(HTTPException) as exc:
        validate_name("J")

    assert exc.value.status_code == 400
    assert "at least 2 characters" in exc.value.detail


def test_validate_name_invalid_characters():
    """
    Summary:
        Ensures that names containing digits or symbols are rejected.

    Explanation:
        Names must contain only letters and spaces. "John123" should
        trigger an HTTPException indicating invalid characters.
    """
    with pytest.raises(HTTPException) as exc:
        validate_name("John123")

    assert exc.value.status_code == 400
    assert "only letters and spaces" in exc.value.detail


def test_validate_name_too_long():
    """
    Summary:
        Ensures that names longer than 50 characters are rejected.

    Explanation:
        Names exceeding reasonable length should be disallowed
        to prevent malformed or malicious input.
    """
    with pytest.raises(HTTPException):
        validate_name("A" * 51)


def test_validate_email_valid(monkeypatch):
    """
    Summary:
        Ensures that valid emails with resolvable domains pass validation.

    Explanation:
        The validate_email function checks both format and domain.
        DNS resolution is mocked here to simulate a valid domain.
    """
    def mock_resolve(domain, record):
        return True

    monkeypatch.setattr("dns.resolver.resolve", mock_resolve)

    validate_email("test@example.com")


def test_validate_email_invalid_format():
    """
    Summary:
        Ensures that emails with invalid format are rejected.

    Explanation:
        Email must follow standard format (user@domain). "invalid-email"
        triggers HTTPException with status 400.
    """
    with pytest.raises(HTTPException) as exc:
        validate_email("invalid-email")

    assert exc.value.status_code == 400
    assert "Invalid email format" in exc.value.detail


def test_validate_email_invalid_domain(monkeypatch):
    """
    Summary:
        Ensures that emails with unresolvable domains are rejected.

    Explanation:
        DNS resolution is mocked to raise an error. validate_email
        should catch this and raise HTTPException indicating the domain
        does not accept mail.
    """
    def mock_resolve(domain, record):
        raise Exception("DNS error")

    monkeypatch.setattr("dns.resolver.resolve", mock_resolve)

    with pytest.raises(HTTPException) as exc:
        validate_email("test@invaliddomain.com")

    assert exc.value.status_code == 400
    assert "does not accept mail" in exc.value.detail


def test_validate_password_valid():
    """
    Summary:
        Ensures that a strong password passes validation.

    Explanation:
        Passwords must meet complexity requirements (uppercase, lowercase,
        digits, special chars). "Strong@123" satisfies these rules.
    """
    validate_password("Strong@123")


def test_validate_password_invalid():
    """
    Summary:
        Ensures that weak passwords are rejected.

    Explanation:
        Passwords that do not meet complexity requirements trigger
        HTTPException with descriptive message.
    """
    with pytest.raises(HTTPException) as exc:
        validate_password("weakpass")

    assert exc.value.status_code == 400
    assert "Password must include" in exc.value.detail


def test_validate_role_valid_user():
    """
    Summary:
        Validates that 'user' is accepted as a valid role.

    Explanation:
        The validate_role function enforces allowed roles. This test ensures
        the lowercase 'user' is valid.
    """
    validate_role("user")


def test_validate_role_valid_admin():
    """
    Summary:
        Validates that 'ADMIN' is accepted as a valid role (case-insensitive).

    Explanation:
        Role checking is case-insensitive, so 'ADMIN' should be allowed.
    """
    validate_role("ADMIN")


def test_validate_role_invalid():
    """
    Summary:
        Ensures that invalid roles are rejected.

    Explanation:
        Roles not included in the allowed list (like 'superadmin') trigger
        HTTPException with status 400.
    """
    with pytest.raises(HTTPException) as exc:
        validate_role("superadmin")

    assert exc.value.status_code == 400
    assert "Invalid role" in exc.value.detail


def test_setup_cors_adds_middleware():
    """
    Summary:
        Ensures that setup_cors adds the CORSMiddleware to a FastAPI app.

    Explanation:
        The setup_cors function adds middleware for Cross-Origin Resource Sharing.
        This test verifies that the middleware list includes CORSMiddleware.
    """
    app = FastAPI()

    setup_cors(app)

    middleware_classes = [mw.cls.__name__ for mw in app.user_middleware]
    assert "CORSMiddleware" in middleware_classes
