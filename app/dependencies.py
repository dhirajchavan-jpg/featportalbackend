# app/dependencies.py
from fastapi import Depends, HTTPException
from pydantic import BaseModel, EmailStr
from app.auth.jwt_bearer import JWTBearer
from app.auth.jwt_handler import decode_access_token

class UserPayload(BaseModel):
    user_id: str
    email: EmailStr
    role: str

def get_current_user(token: str = Depends(JWTBearer())) -> UserPayload:
    """
    Dependency to decode the JWT token, validate its payload,
    and return the user information as a Pydantic model.
    """
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(status_code=403, detail="Invalid or expired token.")

    user_id = payload.get("user_id")
    user_email = payload.get("sub")
    user_role = payload.get("role")
    
    if user_id is None or user_email is None:
        raise HTTPException(status_code=403, detail="Token payload is invalid.")
    
    return UserPayload(user_id=user_id, email=user_email, role=user_role)