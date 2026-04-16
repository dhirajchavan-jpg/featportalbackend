# app/middleware/custom_expection.py

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import traceback
import time
import json
from datetime import datetime

# --- Local Imports ---
from app.schemas import StandardResponse
from app.config import settings

class CustomExceptionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip for large uploads to avoid breaking streaming
        if request.url.path.startswith("/files/projects/"):
             return await call_next(request)

        start_time = time.time()
        
        # Run the request. All exceptions will be caught by handlers, not this.
        response = await call_next(request)
        
        process_time = f"{(time.time() - start_time):.4f}s"
        
        # Set the process time as a header. This is how the exception handlers
        # will get the time.
        response.headers["X-Process-Time"] = process_time

        # Inject process_time into SUCCESSFUL JSON responses
        if 200 <= response.status_code < 300 and response.media_type == "application/json":
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            try:
                data = json.loads(response_body.decode())
                if isinstance(data, dict) and data.get("status") == "success":
                    data["process_time"] = process_time
                    return JSONResponse(
                        content=data, 
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass # Not a JSON body we can modify

            # Recreate the iterator if we consumed it but didn't modify it
            response.body_iterator = iter([response_body])
            
        return response