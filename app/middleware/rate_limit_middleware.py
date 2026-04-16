# app/middleware/rate_limit_middleware.py
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from datetime import datetime

from app.auth.jwt_handler import decode_access_token
from app.config import settings
from app.services.rate_limit_service import try_consume_uploads, try_consume_chat_token


UPLOAD_LIMIT = settings.RATE_LIMIT_UPLOADS_PER_HOUR
CHAT_LIMIT = settings.RATE_LIMIT_CHAT_PER_MINUTE


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

        path = request.url.path
        method = request.method.upper()

        # ----------------------------------------------------
        # CASE 1: FILE UPLOAD ENDPOINTS (bulk + single upload)
        # ----------------------------------------------------
        if (path == "/upload" and method == "POST") or (
            path.startswith("/files/projects") and method == "POST"
        ):
            payload = self._get_payload(request)
            if not payload:
                return self._unauthenticated_response()

            # Ensure body is stored for FastAPI to re-read
            upload_count = await self._determine_upload_count(request)

            allowed, remaining, reset_seconds = await try_consume_uploads(
                payload["user_id"], upload_count
            )

            if not allowed:
                return self._rate_limited_response(
                    limit=UPLOAD_LIMIT,
                    remaining=remaining,
                    retry_after=reset_seconds,
                    reason="upload",
                )

            response = await call_next(request)
            self._attach_headers(
                response,
                UPLOAD_LIMIT,
                remaining,
                int(datetime.utcnow().timestamp() + reset_seconds),
            )
            return response

        # ----------------------------------------------------
        # CASE 2: CHAT QUERY RATE LIMIT
        # ----------------------------------------------------
        if path.startswith("/query") and method == "POST":


            payload = self._get_payload(request)
            if not payload:
                return self._unauthenticated_response()

            allowed, remaining, retry_after = await try_consume_chat_token(
                payload["user_id"]
            )

            if not allowed:
                return self._rate_limited_response(
                    limit=CHAT_LIMIT,
                    remaining=remaining,
                    retry_after=retry_after,
                    reason="chat",
                )

            response = await call_next(request)
            self._attach_headers(
                response,
                CHAT_LIMIT,
                remaining,
                int(datetime.utcnow().timestamp() + retry_after),
            )
            return response

        return await call_next(request)

    # --------------------------------------------------------
    # Extract JWT Payload
    # --------------------------------------------------------
    def _get_payload(self, request: Request):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return None
        token = auth.split(" ")[1]
        return decode_access_token(token)

    # --------------------------------------------------------
    # Ensure form-data can be read multiple times
    # --------------------------------------------------------
    async def _determine_upload_count(self, request: Request) -> int:
        """
        Do NOT consume multipart/form-data.
        Only handle non-multipart bodies.
        """

        content_type = request.headers.get("Content-Type", "")

        # If body is multipart/form-data → DO NOT TOUCH IT
        if "multipart/form-data" in content_type:
            return 1  # Let FastAPI parse normally

        # For non-multipart → safe to read
        body = await request.body()
        request._body = body

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request._receive = receive

        return 1


    # --------------------------------------------------------
    # Rate Limit Response
    # --------------------------------------------------------
    def _rate_limited_response(self, limit, remaining, retry_after, reason):
        return JSONResponse(
            status_code=429,
            content={
                "status": "error",
                "status_code": 429,
                "message": f"Rate limit exceeded for {reason}. You can only upload 10 documents in 1 hour",
                "retry_after_seconds": retry_after,
                "limit": limit,
                "remaining": remaining,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(
                    int(datetime.utcnow().timestamp() + max(1, retry_after))

                ),
                "Access-Control-Allow-Origin": "*", 
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*"
            },
        )

    # --------------------------------------------------------
    # Success Response Headers
    # --------------------------------------------------------
    def _attach_headers(self, response, limit, remaining, reset_ts):
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_ts)

    # --------------------------------------------------------
    # Unauthenticated
    # --------------------------------------------------------
    def _unauthenticated_response(self):
        return JSONResponse(
            status_code=401,
            content={
                "status": "error",
                "status_code": 401,
                "message": "Invalid or missing token.",
            },
        )
