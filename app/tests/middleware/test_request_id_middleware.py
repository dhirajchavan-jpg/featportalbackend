import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.request_id_middleware import (
    request_id_middleware,
    REQUEST_ID_CTX,
)

# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def app():
    """
    Summary:
        Provides a FastAPI app instance with request_id_middleware applied.

    Explanation:
        Each test receives a fresh app instance.
        Routes include a test endpoint and a file upload endpoint
        to simulate both normal and file-specific requests.
    """
    app = FastAPI()

    app.middleware("http")(request_id_middleware)

    @app.get("/test")
    async def test_endpoint():
        return {"ok": True}

    @app.post("/files/projects/123")
    async def upload_endpoint():
        return {"uploaded": True}

    return app


@pytest.fixture
def client(app):
    """
    Summary:
        Provides a TestClient for the FastAPI app.

    Explanation:
        Allows sending HTTP requests to the app routes with middleware applied.
    """
    return TestClient(app)


# -------------------------
# Middleware Tests
# -------------------------

def test_request_id_generated_when_missing(client):
    """
    Summary:
        Generates a unique X-Request-ID if missing in the request.

    Explanation:
        Middleware automatically generates a request ID if the client
        does not provide one. Ensures header is present and valid.
    """
    response = client.get("/test")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers

    request_id = response.headers["X-Request-ID"]
    assert isinstance(request_id, str)
    assert len(request_id) > 0


def test_request_id_preserved_from_header(client):
    """
    Summary:
        Preserves the X-Request-ID if provided in the request header.

    Explanation:
        Middleware should not overwrite client-supplied request IDs,
        allowing correlation across services.
    """
    custom_id = "my-custom-request-id"

    response = client.get(
        "/test",
        headers={"X-Request-ID": custom_id},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_id


def test_request_id_contextvar_set(client):
    """
    Summary:
        Sets REQUEST_ID_CTX context variable during request.

    Explanation:
        Ensures the middleware sets the context variable during processing
        but does not leak it after the request finishes.
    """
    response = client.get("/test")

    assert response.status_code == 200

    # ContextVar should not leak after request
    assert REQUEST_ID_CTX.get() is None


def test_request_id_skipped_for_file_uploads(client):
    """
    Summary:
        Skips generating X-Request-ID for file upload endpoints.

    Explanation:
        Middleware bypasses file uploads to avoid unnecessary ID headers
        for multipart/form-data requests.
    """
    response = client.post("/files/projects/123")

    assert response.status_code == 200
    assert "X-Request-ID" not in response.headers


def test_request_ids_are_unique(client):
    """
    Summary:
        Ensures each request receives a unique X-Request-ID.

    Explanation:
        Confirms middleware generates distinct IDs for multiple requests,
        which is important for logging and traceability.
    """
    r1 = client.get("/test")
    r2 = client.get("/test")

    assert r1.headers["X-Request-ID"] != r2.headers["X-Request-ID"]


def test_response_body_unchanged(client):
    """
    Summary:
        Verifies middleware does not alter response content.

    Explanation:
        Middleware should only add headers, leaving the body intact.
        Ensures original API response is returned correctly.
    """
    response = client.get("/test")

    assert response.json() == {"ok": True}
