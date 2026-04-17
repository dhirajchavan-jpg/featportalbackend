from fastapi import Depends, HTTPException, status
from app.dependencies import get_current_user, UserPayload


ROLE_ALIASES = {
    "hr": "admin",
    "administrator": "admin",
    "superadmin": "super_admin",
    "ceo": "super_admin",
}


def _normalize_role(role: str) -> str:
    value = (role or "").strip().lower()
    return ROLE_ALIASES.get(value, value)


def require_roles(*allowed_roles):
    normalized_allowed = {_normalize_role(r) for r in allowed_roles}

    def role_checker(current_user: UserPayload = Depends(get_current_user)):
        user_role = _normalize_role(current_user.role)

        if user_role not in normalized_allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden. Allowed roles {tuple(sorted(normalized_allowed))}"
            )
        return current_user

    return role_checker
