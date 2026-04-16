# app/utils/role_checker.py
from fastapi import Depends, HTTPException, status
from app.dependencies import get_current_user, UserPayload

def require_roles(*allowed_roles):
    def role_checker(current_user: UserPayload = Depends(get_current_user)):
        user_role = current_user.role.lower()

        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden. Allowed roles {allowed_roles}"
            )
        return current_user
    return role_checker
