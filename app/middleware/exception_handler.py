# app/middleware/exception_handler.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from app.middleware.response_helper_middleware import error_response_dict
import logging

logger = logging.getLogger(__name__)

async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Global handler for HTTPException that converts all exceptions
    to your structured GlobalResponse format automatically.
    """
    # If detail is already a dict (from error_response_dict), use it directly
    if isinstance(exc.detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    
    # Otherwise, wrap plain string errors in GlobalResponse format
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response_dict(
            message=str(exc.detail),
            code="HTTP_ERROR",
            status_code=exc.status_code
        )
    )

async def general_exception_handler(request: Request, exc: Exception):
    """
    Catches any unhandled exceptions and returns them in GlobalResponse format.
    """
    logger.exception(f"Unhandled exception during {request.method} {request.url}")
    
    return JSONResponse(
        status_code=500,
        content=error_response_dict(
            message="An unexpected error occurred",
            code="INTERNAL_SERVER_ERROR",
            status_code=500
        )
    )
