import re
import dns.resolver
from fastapi import HTTPException

def validate_name(name: str):
    if not name or len(name.strip()) < 2:
        raise HTTPException(status_code=400, detail="Name must be at least 2 characters long")
    if not re.match(r"^[A-Za-z\s]+$", name):
        raise HTTPException(status_code=400, detail="Name must contain only letters and spaces")
    if len(name) > 50:
        raise HTTPException(status_code=400, detail="Name too long (max 50 characters)")

def validate_email(email: str):
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    if not re.match(pattern, email):
        raise HTTPException(status_code=400, detail="Invalid email format")
    try:
        domain = email.split('@')[1]
        dns.resolver.resolve(domain, 'MX')
    except Exception:
        raise HTTPException(status_code=400, detail="Email domain is not valid or does not accept mail.")

def validate_password(password: str):
    if len(password) < 8 or not re.search(r"[A-Z]", password) or not re.search(r"[a-z]", password) or not re.search(r"\d", password) or not re.search(r"[@$!%*?&#]", password):
        raise HTTPException(status_code=400, detail="Password must include at least one number, one lowercase letter, a special character and upper case letter")

def validate_role(role: str):
    allowed_roles = ["user", "admin"]
    if role.lower() not in allowed_roles:
        raise HTTPException(status_code=400, detail=f"Invalid role. Allowed roles: {allowed_roles}")

def setup_cors(app):
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
