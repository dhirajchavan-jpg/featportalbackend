# app/models/Admin_file.py

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime



class GlobalFileUploadForm(BaseModel):
    """Form model for uploading global files"""
    sector: str = Field(..., description="Sector name (e.g., RBI, GDPR, HIPAA)")
    document_type: str = Field(..., description="Document type: 'circular' or 'guidelines'")
    category: str = Field(..., min_length=1, description="Category (e.g., 'Cybersecurity', 'Compliance')")
    description: Optional[str] = Field(None, description="Optional file description")
    effective_date: Optional[str] = Field(None, description="Effective date (YYYY-MM-DD)")
    version: Optional[str] = Field(None, description="Document version")

class GlobalFileDB(BaseModel):
    # --- Fields from Form Inputs ---
    sector: str
    document_type: str
    category: str
    description: Optional[str] = None
    effective_date: Optional[str] = None
    version: Optional[str] = None

    # --- Fields from File & User ---
    filename: str
    uploaded_by: str
    file_hash: str

    # --- Fields from System Processing ---
    file_id: str
    file_path: str
    file_url: Optional[str] = None

   
    
    # --- Timestamps ---
    created_at: datetime
    updated_at: datetime
    is_active: bool = True