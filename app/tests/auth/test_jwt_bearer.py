import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends
from unittest.mock import patch

from app.auth.jwt_bearer import JWTBearer, validate_token


app = FastAPI()

@app.get("/protected")
async def protected(token: str = Depends(JWTBearer())):
    return {"token": token}

client = TestClient(app)


def test_missing_authorization_header():
    """Summary: Request without Authorization header must be rejected with 401."""
    response = client.get("/protected")
    assert response.status_code == 401
    detail = response.json()["detail"]
    assert detail["status"] == "error"
    assert detail["status_code"] == 401
    assert detail["message"] == "Authentication credentials were not provided."



def test_invalid_auth_scheme():
    """Summary: Non-Bearer authentication schemes must be rejected."""
    response = client.get(
        "/protected",
        headers={"Authorization": "Basic abcdef"}
    )

    body = response.json()["detail"]

    assert response.status_code == 401
    assert body["status_code"] == 401
    assert body["message"] == "Authentication credentials were not provided."
    assert body["errors"][0]["field"] == "Authorization"



@patch("app.auth.jwt_bearer.decode_access_token")
def test_invalid_token(mock_decode):
    """Summary: Invalid or expired JWT must return 403 Forbidden."""
    mock_decode.return_value = None

    response = client.get(
        "/protected",
        headers={"Authorization": "Bearer invalidtoken"}
    )

    body = response.json()["detail"]

    assert response.status_code == 403
    assert body["status_code"] == 403
    assert body["message"] == "Invalid or expired token."


@patch("app.auth.jwt_bearer.decode_access_token")
def test_valid_token(mock_decode):
    """Summary: Valid JWT must allow access to protected endpoint."""
    mock_decode.return_value = {"user_id": "123", "sub": "user@test.com"}

    response = client.get(
        "/protected",
        headers={"Authorization": "Bearer validtoken"}
    )

    assert response.status_code == 200

@patch("app.auth.jwt_bearer.decode_access_token")
@pytest.mark.asyncio
async def test_validate_token_invalid(mock_decode):
    """Summary: Token validation must fail when JWT decoding returns None."""
    mock_decode.return_value = None

    with pytest.raises(HTTPException) as exc:
        await validate_token("badtoken")

    assert exc.value.status_code == 401


@patch("app.auth.jwt_bearer.decode_access_token")
@pytest.mark.asyncio
async def test_validate_token_missing_fields(mock_decode):
    """Summary: Token payload missing required fields must raise 400 error."""
    mock_decode.return_value = {"sub": "user@test.com"}

    with pytest.raises(HTTPException) as exc:
        await validate_token("token")

    assert exc.value.status_code == 400

@patch("app.auth.jwt_bearer.decode_access_token")
@pytest.mark.asyncio
async def test_validate_token_success(mock_decode):
    """Summary: Valid token payload must be returned after successful validation."""
    mock_decode.return_value = {"user_id": "123", "sub": "user@test.com"}

    payload = await validate_token("token")

    assert payload["user_id"] == "123"
