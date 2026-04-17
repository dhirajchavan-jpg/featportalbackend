from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone


class ProjectModel(BaseModel):
    project_name: str
    has_organizational_compliance: bool = True
    organization_sector: str
    industry: str
    description: str
    # Kept for compatibility; backend enforces a single sector entry.
    sectors: Optional[List[str]] = []
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProjectUpdateSchema(BaseModel):
    project_name: Optional[str] = None
    description: Optional[str] = None
    industry: Optional[str] = None
