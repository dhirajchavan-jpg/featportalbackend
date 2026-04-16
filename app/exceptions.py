# app/exceptions.py

import traceback
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.config import settings
from app.schemas import StandardResponse, ErrorDetail

async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handler for FastAPI's built-in HTTPException (like 403, 404, 422).
    This will catch the errors you are raising in your endpoints.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=StandardResponse(
            status="error",
            status_code=exc.status_code,
            message=exc.detail,
            errors=[ErrorDetail(message=exc.detail)],
            process_time=request.headers.get("X-Process-Time", "0.0s")
        ).model_dump(exclude_none=True)
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handler for Pydantic's RequestValidationError (422).
    This catches invalid request bodies.
    """
    errors = []
    for error in exc.errors():
        # Get the field name, remove 'body' prefix
        field = " -> ".join(map(str, error.get("loc", [])))
        if field.startswith("body -> "):
            field = field[7:]
            
        errors.append(ErrorDetail(
            message=error.get("msg"),
            field=field,
            rejected_value=error.get("input")
        ))

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=StandardResponse(
            status="error",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="Input validation failed",
            errors=errors,
            process_time=request.headers.get("X-Process-Time", "0.0s")
        ).model_dump(exclude_none=True)
    )

async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handler for all other unexpected 500-level exceptions (crashes).
    Returns a detailed stack trace in DEBUG mode.
    """
    meta = None
    if settings.DEBUG:
        meta = {"traceback": traceback.format_exc().splitlines()}

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=StandardResponse(
            status="error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="An unexpected internal server error occurred.",
            errors=[ErrorDetail(message=str(exc))],
            meta=meta,
            process_time=request.headers.get("X-Process-Time", "0.0s")
        ).model_dump(exclude_none=True)
    )