from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class UserFileSelectionModel(BaseModel):
    file_id: str
    filename: str
    project_id: Optional[str] = None
    source: str                     # "admin", "project", etc.
    deselected: bool = True
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat() if dt else None,
        }
