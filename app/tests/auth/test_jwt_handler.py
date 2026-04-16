import pytest
from datetime import timedelta

from app.auth.jwt_handler import create_access_token, decode_access_token


def test_create_access_token_returns_string():
    """Summary: Access token creation must return a JWT string."""
    token = create_access_token({"user_id": "123", "sub": "user@test.com"})
    assert isinstance(token, str)


def test_decode_access_token_valid():
    """Summary: Valid JWT must decode successfully and return original payload."""
    data = {"user_id": "123", "sub": "user@test.com"}
    token = create_access_token(data)
    
    payload = decode_access_token(token)

    assert payload["user_id"] == "123"
    assert payload["sub"] == "user@test.com"


def test_decode_access_token_invalid():
    """Summary: Invalid or malformed JWT must return None."""
    payload = decode_access_token("invalid.token.value")
    assert payload is None
