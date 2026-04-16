import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.datastructures import Headers

from app.middleware.exception_handler import (
    http_exception_handler,
    general_exception_handler
)


class DummyRequest(Request):
    """
    Summary:
        Minimal HTTP request object used for testing exception handlers.

    Explanation:
        This dummy request simulates a real Starlette Request instance
        without requiring an active ASGI server. It provides the minimal
        scope needed to invoke exception handlers directly in unit tests.
    """
    def __init__(self):
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": Headers({}).raw,
        }
        super().__init__(scope)


@pytest.mark.asyncio
async def test_http_exception_with_dict_detail():
    """
    Summary:
        HTTPException with dictionary-based detail must be handled correctly.

    Explanation:
        This test verifies that when an HTTPException contains a structured
        dictionary as its detail, the exception handler correctly extracts
        and serializes the error information into the HTTP response body
        without losing the original status code or message.
    """
    detail = {
        "status": "error",
        "status_code": 404,
        "message": "Not found"
    }
    exc = HTTPException(status_code=404, detail=detail)

    response = await http_exception_handler(DummyRequest(), exc)

    assert response.status_code == 404
    assert response.body
    assert b"Not found" in response.body


@pytest.mark.asyncio
async def test_http_exception_with_string_detail():
    """
    Summary:
        HTTPException with string-based detail must be normalized into an error response.

    Explanation:
        This test ensures that when an HTTPException contains a plain string
        as its detail, the exception handler wraps it into a consistent
        error response format, including a default error status.
    """
    exc = HTTPException(status_code=401, detail="Unauthorized")

    response = await http_exception_handler(DummyRequest(), exc)

    assert response.status_code == 401
    assert b"Unauthorized" in response.body
    assert b'"status":"error"' in response.body


@pytest.mark.asyncio
async def test_general_exception_handler():
    """
    Summary:
        Unhandled exceptions must return a standardized internal server error response.

    Explanation:
        This test validates that unexpected runtime exceptions are caught
        by the general exception handler and converted into a safe,
        user-friendly error response with HTTP 500 status, preventing
        internal stack traces from leaking to clients.
    """
    exc = RuntimeError("boom")

    response = await general_exception_handler(DummyRequest(), exc)

    assert response.status_code == 500
    assert b"An unexpected error occurred" in response.body
    assert b'"status":"error"' in response.body
