from pydantic import BaseModel, Field
from typing import List
from typing import Optional
from datetime import datetime, timezone

class ProjectModel(BaseModel):
    
    project_name:str
    regulatory_framework: Optional[str] = None
    third_party_framework: List[str] = []
    has_organizational_compliance: bool = False
    organization_sector: Optional[str] = None
    industry: str
    description:str
    sectors: Optional[List[str]] = []  # New field: list of selected sectors for this project
    user_id: Optional[str] = None  
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ProjectUpdateSchema(BaseModel):
    
    project_name: Optional[str] = None
    description: Optional[str] = None
    industry: Optional[str] = None