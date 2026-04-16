# app/models/Files.py
from typing import Optional
from datetime import datetime
from pydantic import BaseModel,Field
from bson import ObjectId


class FileModel(BaseModel):
    file_id: Optional[str] = None   # Unique ID for the file (from MongoDB)
    project_id: str                 # Reference to the Project ID
    filename: str                   # Original or stored filename (without extension)
    file_url: str                   # File path on your server
    sector: str
    category: str = Field(
        ..., 
        description="The main section: 'Regulatory', 'Third Party', or 'Organization'"
    )
    compliance_type: str = Field(
        ..., 
        description="The sub-section: 'Circular' or 'Guidelines'"
    )
    
    file_hash: Optional[str] = Field(
        None, description="SHA-256 hash of the file content"
    )

    user_id: Optional[str] = Field(
        None, description="ID of the user who uploaded the file"
    )

    created_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of upload",
    )

    class Config:
        from_attributes=True      # Replaces 'orm_mode'
        populate_by_name=True
        json_encoders = {
            ObjectId: str,
            datetime: lambda dt: dt.isoformat(),
        }




