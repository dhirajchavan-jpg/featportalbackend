# app/auth/jwt_bearer.py
from fastapi import Request, HTTPException, logger
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import ExpiredSignatureError, JWTError
from .jwt_handler import decode_access_token
from fastapi import HTTPException
from app.auth.jwt_handler import decode_access_token
from app.middleware.response_helper_middleware import error_response_dict

class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = False):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)

        if not credentials:
            raise HTTPException(
                status_code=401,
                detail=error_response_dict(
                    message="Authentication credentials were not provided.",
                    code="AUTHENTICATION_REQUIRED",
                    status_code=401,
                    errors=[{"field": "Authorization", "message": "Missing credentials"}]
                )
            )

        if credentials.scheme != "Bearer":
            raise HTTPException(
                status_code=403,
                detail=error_response_dict(
                    message="Invalid authentication scheme.",
                    code="INVALID_AUTH_SCHEME",
                    status_code=403,
                    errors=[{"field": "Authorization", "message": "Scheme must be Bearer", "rejected_value": credentials.scheme}]
                )
            )

        if not self.verify_jwt(credentials.credentials):
            raise HTTPException(
                status_code=403,
                detail=error_response_dict(
                    message="Invalid or expired token.",
                    code="INVALID_OR_EXPIRED_TOKEN",
                    status_code=403,
                    errors=[{"field": "Authorization", "message": "Token invalid or expired"}]
                )
            )

        return credentials.credentials

    def verify_jwt(self, jwtoken: str):
        payload = decode_access_token(jwtoken)
        return payload is not None


        

async def validate_token(token: str):
    """
    Validates JWT token and ensures it contains required claims.
    Raises HTTPException with consistent error response if invalid or expired.
    """
    try:
        payload = decode_access_token(token)

        # If decoding failed or returned None
        if not payload:
            raise HTTPException(
                status_code=401,
                detail={
                    "status": "error",
                    "status_code": 401,
                    "code": "INVALID_OR_EXPIRED_TOKEN",
                    "message": "Invalid or expired token. Please login again.",
                    "errors": [
                        {
                            "field": "Authorization",
                            "message": "Token decoding failed or expired",
                            "rejected_value": token
                        }
                    ],
                    "data": None
                }
            )

        # Check mandatory fields in payload
        if not payload.get("user_id") or not payload.get("sub"):
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "status_code": 400,
                    "code": "TOKEN_PAYLOAD_MISSING",
                    "message": "User ID or email missing in token.",
                    "errors": [
                        {"field": "token", "message": "Payload missing 'user_id' or 'sub'"}
                    ],
                    "data": None
                }
            )

        return payload

    except HTTPException:
        raise  # already properly formatted
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "status_code": 401,
                "code": "INVALID_OR_EXPIRED_TOKEN",
                "message": "Invalid or expired token. Please login again.",
                "errors": [
                    {
                        "field": "Authorization",
                        "message": str(e),
                        "rejected_value": token
                    }
                ],
                "data": None
            }
        )
