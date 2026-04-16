import pytest
from fastapi import HTTPException

from app.dependencies import get_current_user, UserPayload


# ---------------------------------------------------------
# SUCCESS CASE
# ---------------------------------------------------------

def test_get_current_user_success(mocker):
    """
    Valid token payload → UserPayload returned
    """
    mocker.patch(
        "app.dependencies.decode_access_token",
        return_value={
            "user_id": "user123",
            "sub": "user@example.com",
            "role": "admin",
        },
    )

    result = get_current_user(token="fake-token")

    assert isinstance(result, UserPayload)
    assert result.user_id == "user123"
    assert result.email == "user@example.com"
    assert result.role == "admin"


# ---------------------------------------------------------
# TOKEN DECODE FAILURE
# ---------------------------------------------------------

def test_get_current_user_invalid_token(mocker):
    """
    decode_access_token returns None → 403
    """
    mocker.patch(
        "app.dependencies.decode_access_token",
        return_value=None,
    )

    with pytest.raises(HTTPException) as exc:
        get_current_user(token="invalid-token")

    assert exc.value.status_code == 403
    assert "Invalid or expired token" in exc.value.detail


# ---------------------------------------------------------
# MISSING user_id
# ---------------------------------------------------------

def test_get_current_user_missing_user_id(mocker):
    """
    Missing user_id → 403
    """
    mocker.patch(
        "app.dependencies.decode_access_token",
        return_value={
            "sub": "user@example.com",
            "role": "user",
        },
    )

    with pytest.raises(HTTPException) as exc:
        get_current_user(token="fake-token")

    assert exc.value.status_code == 403
    assert "Token payload is invalid" in exc.value.detail


# ---------------------------------------------------------
# MISSING email (sub)
# ---------------------------------------------------------

def test_get_current_user_missing_email(mocker):
    """
    Missing email (sub) → 403
    """
    mocker.patch(
        "app.dependencies.decode_access_token",
        return_value={
            "user_id": "user123",
            "role": "user",
        },
    )

    with pytest.raises(HTTPException) as exc:
        get_current_user(token="fake-token")

    assert exc.value.status_code == 403
    assert "Token payload is invalid" in exc.value.detail
