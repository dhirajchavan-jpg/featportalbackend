import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.middleware.custom_expection import CustomExceptionMiddleware


@pytest.fixture
def app():
    """
    Summary:
        Provide a FastAPI application configured with custom exception middleware.

    Explanation:
        This fixture sets up a minimal FastAPI application with the
        CustomExceptionMiddleware applied, allowing middleware behavior
        to be tested in isolation using test endpoints.
    """
    app = FastAPI()
    app.add_middleware(CustomExceptionMiddleware)

    @app.get("/success")
    async def success():
        return {"status": "success", "data": {"msg": "ok"}}

    @app.get("/error")
    async def error():
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "bad request"}
        )

    return app


def test_process_time_added_on_success(app):
    """
    Summary:
        Successful responses must include processing time metadata.

    Explanation:
        This test verifies that the CustomExceptionMiddleware augments
        successful responses by adding process timing information,
        supporting observability and performance monitoring.
    """
    client = TestClient(app)
    response = client.get("/success")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "success"


def test_process_time_not_added_on_error(app):
    """
    Summary:
        Error responses must not include processing time metadata.

    Explanation:
        This ensures that the middleware does not alter or decorate
        error responses, preserving original error semantics and payloads.
    """
    client = TestClient(app)
    response = client.get("/error")

    body = response.json()
    assert response.status_code == 400
    assert "process_time" not in body
