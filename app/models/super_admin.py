from pydantic import BaseModel, EmailStr, Field, BeforeValidator
from typing import Optional, List, Annotated
from datetime import datetime
from bson import ObjectId

# --- ObjectId Helper for Pydantic v2 ---
PyObjectId = Annotated[str, BeforeValidator(str)]

# --- Super Admin Login Schema ---
class SuperAdminLogin(BaseModel):
    username: str
    password: str

# --- Schema for Creating a Tenant/Org Admin ---
class OrgAdminCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    organization_name: str
    phone_number: Optional[str] = None

# --- Schema for Updating an Admin ---
class OrgAdminUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    organization_name: Optional[str] = None
    phone_number: Optional[str] = None
    is_active: Optional[bool] = None  # To ban/unban

# --- Schema for Response (Dashboard View) ---
class OrgAdminResponse(BaseModel):
    id: PyObjectId = Field(alias="_id")
    username: str
    email: EmailStr
    organization_name: Optional[str] = "N/A"
    role: str
    is_active: bool
    created_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}

# --- Stats Schema ---
class DashboardStats(BaseModel):
    total_admins: int
    total_users: int
    total_files: int