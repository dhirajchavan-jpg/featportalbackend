# app/global_response.py
from typing import Any, List, Optional
from pydantic import BaseModel, Field

class ErrorItem(BaseModel):
    message: str
    field: Optional[str] = None
    rejected_value: Optional[Any] = None

class Meta(BaseModel):
    page: Optional[int] = None
    per_page: Optional[int] = None
    total: Optional[int] = None

class GlobalResponse(BaseModel):
    status: str = Field(..., pattern="^(success|error)$")
    status_code: int
    message: str
    data: Optional[Any] = None
    errors: Optional[List[ErrorItem]] = None
    meta: Optional[Meta] = None
    process_time: Optional[str] = None
