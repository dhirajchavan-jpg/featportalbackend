import uuid
from contextvars import ContextVar
from fastapi import Request, Response

# This ContextVar is still useful to have, even if logging isn't set up yet.
REQUEST_ID_CTX = ContextVar("request_id", default=None)

async def request_id_middleware(request: Request, call_next):
    """
    Middleware to add a unique X-Request-ID to every request.
    """
    # Skip this middleware for file uploads (multipart)
    if request.url.path.startswith("/files/projects/"):
        return await call_next(request)

    # Get ID from header or generate a new one
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())

    # Store it in the ContextVar
    token = REQUEST_ID_CTX.set(request_id)

    try:
        # Process the request
        response: Response = await call_next(request)
        
        # Add the ID to the response header
        response.headers["X-Request-ID"] = request_id
        
        return response

    finally:
        # Clean up the ContextVar
        REQUEST_ID_CTX.reset(token)