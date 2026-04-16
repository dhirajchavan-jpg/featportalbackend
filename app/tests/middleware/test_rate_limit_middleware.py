import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.middleware.rate_limit_middleware import RateLimitMiddleware

# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def app():
    """
    Summary:
        Provides a FastAPI app instance with RateLimitMiddleware applied.

    Explanation:
        Each test receives a fresh app instance to ensure isolation.
        Endpoints for upload, bulk file upload, and query simulate
        real routes protected by rate limiting middleware.
    """
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)

    @app.post("/upload")
    async def upload():
        return {"ok": True}

    @app.post("/files/projects/123")
    async def bulk_upload():
        return {"ok": True}

    @app.post("/query")
    async def query():
        return {"ok": True}

    return app


@pytest.fixture
def client(app):
    """
    Summary:
        Provides a TestClient for the FastAPI app.

    Explanation:
        Allows sending test HTTP requests to routes with middleware applied.
    """
    return TestClient(app)


@pytest.fixture
def valid_token():
    """
    Summary:
        Returns a mock valid JWT token string.

    Explanation:
        Used to simulate authenticated requests in tests.
    """
    return "Bearer valid.jwt.token"


@pytest.fixture
def jwt_payload():
    """
    Summary:
        Returns a dictionary representing decoded JWT payload.

    Explanation:
        Simulates authenticated user identity for rate limit checks.
    """
    return {"user_id": "user123"}


# -------------------------
# Upload Endpoint Tests
# -------------------------

def test_upload_unauthenticated(client):
    """
    Summary:
        Unauthenticated upload requests are rejected.

    Explanation:
        Requests without Authorization header receive 401 Unauthorized
        to ensure rate limit checks require authentication.
    """
    response = client.post("/upload")

    assert response.status_code == 401
    assert response.json()["message"] == "Invalid or missing token."


@patch("app.middleware.rate_limit_middleware.decode_access_token")
@patch("app.middleware.rate_limit_middleware.try_consume_uploads")
def test_upload_rate_limited(mock_consume, mock_decode, client, valid_token, jwt_payload):
    """
    Summary:
        Authenticated uploads exceeding rate limits are rejected.

    Explanation:
        Simulates token decode and rate limit consumption failure.
        Returns 429 Too Many Requests with Retry-After header.
    """
    mock_decode.return_value = jwt_payload
    mock_consume.return_value = (False, 0, 60)

    response = client.post(
        "/upload",
        headers={"Authorization": valid_token},
    )

    assert response.status_code == 429
    body = response.json()

    assert body["status"] == "error"
    assert body["status_code"] == 429
    assert "Rate limit exceeded for upload" in body["message"]
    assert response.headers["Retry-After"] == "60"


@patch("app.middleware.rate_limit_middleware.decode_access_token")
@patch("app.middleware.rate_limit_middleware.try_consume_uploads")
def test_upload_success(mock_consume, mock_decode, client, valid_token, jwt_payload):
    """
    Summary:
        Authenticated uploads within rate limits succeed.

    Explanation:
        Middleware correctly allows requests and returns rate limit headers
        for client monitoring.
    """
    mock_decode.return_value = jwt_payload
    mock_consume.return_value = (True, 9, 120)

    response = client.post(
        "/upload",
        headers={"Authorization": valid_token},
        json={"foo": "bar"},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True

    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers
    assert "X-RateLimit-Reset" in response.headers


@patch("app.middleware.rate_limit_middleware.decode_access_token")
@patch("app.middleware.rate_limit_middleware.try_consume_uploads")
def test_multipart_upload_not_consumed(mock_consume, mock_decode, client, valid_token, jwt_payload):
    """
    Summary:
        Multipart file uploads bypass standard rate limit consumption.

    Explanation:
        Middleware detects multipart content type and allows upload without
        decrementing the upload quota, ensuring bulk file uploads are handled safely.
    """
    mock_decode.return_value = jwt_payload
    mock_consume.return_value = (True, 9, 60)

    response = client.post(
        "/files/projects/123",
        headers={
            "Authorization": valid_token,
            "Content-Type": "multipart/form-data"
        },
        files={"file": ("test.txt", b"hello")}
    )

    assert response.status_code == 200


# -------------------------
# Chat Endpoint Tests
# -------------------------

def test_chat_unauthenticated(client):
    """
    Summary:
        Unauthenticated chat requests are rejected.

    Explanation:
        Ensures chat endpoints require JWT for rate limiting and auditing.
    """
    response = client.post("/query")

    assert response.status_code == 401
    assert response.json()["message"] == "Invalid or missing token."


@patch("app.middleware.rate_limit_middleware.decode_access_token")
@patch("app.middleware.rate_limit_middleware.try_consume_chat_token")
def test_chat_rate_limited(mock_consume, mock_decode, client, valid_token, jwt_payload):
    """
    Summary:
        Chat requests exceeding rate limits are rejected.

    Explanation:
        Middleware returns 429 Too Many Requests and retry_after_seconds header
        to indicate when the client can retry.
    """
    mock_decode.return_value = jwt_payload
    mock_consume.return_value = (False, 0, 30)

    response = client.post(
        "/query",
        headers={"Authorization": valid_token},
    )

    assert response.status_code == 429
    assert response.json()["retry_after_seconds"] == 30


@patch("app.middleware.rate_limit_middleware.decode_access_token")
@patch("app.middleware.rate_limit_middleware.try_consume_chat_token")
def test_chat_success(mock_consume, mock_decode, client, valid_token, jwt_payload):
    """
    Summary:
        Chat requests within rate limits succeed.

    Explanation:
        Middleware decrements chat token allowance and returns rate limit headers.
    """
    mock_decode.return_value = jwt_payload
    mock_consume.return_value = (True, 19, 60)

    response = client.post(
        "/query",
        headers={"Authorization": valid_token},
        json={"query": "Hello"},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True

    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers


# -------------------------
# Non-limited route tests
# -------------------------

def test_non_limited_route_passes_through(app):
    """
    Summary:
        Routes not subject to rate limiting function normally.

    Explanation:
        Ensures middleware does not interfere with endpoints
        that are exempt from rate limits.
    """
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
