# app/middleware/project_middleware.py
import re
from fastapi import HTTPException
from typing import List

# Define allowed sectors (should match your global sectors)
ALLOWED_SECTORS = ["RBI", "SEBI", "IRDAI", "GDPR", "HIPAA", "SOC2", "ISO27001", "PCI-DSS", "CCPA"]


# -------------------- Validate Project Name --------------------
def validate_project_name(name: str):
    """
    Validates project name:
    - Minimum 3 characters
    - Maximum 100 characters
    - Only letters, numbers, spaces, and basic punctuation
    """
    if not name or len(name.strip()) < 3:
        raise HTTPException(status_code=422, detail="Project name must be at least 3 characters long.")
    if len(name) > 100:
        raise HTTPException(status_code=422, detail="Project name too long (max 100 characters).")
    if not re.match(r"^[A-Za-z0-9\s\-_,.&()]+$", name):
        raise HTTPException(status_code=422, detail="Project name contains invalid characters. Only letters, numbers, spaces, and basic punctuation are allowed.")


# -------------------- Validate Industry --------------------
def validate_industry(industry: str) -> str:
    """
    Validates industry and returns a cleaned, lowercase version.
    - Minimum 2 characters
    - Maximum 50 characters
    - Only letters, spaces, and &
    """
    if not industry:
         raise HTTPException(status_code=422, detail="Industry cannot be empty.")
     
    cleaned = industry.strip().lower()
    
    if not cleaned:
        raise HTTPException(status_code=422, detail="Industry cannot be empty.")

    # --- MODIFIED ---
    # 1. Convert to lowercase and strip whitespace
    lowercase_industry = industry.lower().strip()
    
    # 2. Perform checks on the cleaned string
    if len(lowercase_industry) < 2:
        raise HTTPException(status_code=422, detail="Industry must be at least 2 characters long.")
    if len(lowercase_industry) > 50:
        raise HTTPException(status_code=422, detail="Industry too long (max 50 characters).")
    if not re.match(r"^[a-z\s&]+$", lowercase_industry): # Updated regex to match lowercase
        raise HTTPException(status_code=422, detail="Industry must contain only letters, spaces, and &.")
        
    # 3. Return the cleaned, lowercase string
    return lowercase_industry


# -------------------- Validate Description --------------------
def validate_description(desc: str):
    """
    Validates description:
    - Minimum 10 characters
    - Maximum 1000 characters
    """
    if not desc or len(desc.strip()) < 10:
        raise HTTPException(status_code=422, detail="Project description must be at least 10 characters long.")
    if len(desc) > 1000:
        raise HTTPException(status_code=422, detail="Project description too long (max 1000 characters).")


# -------------------- Validate Sectors (NEW) --------------------
def validate_sectors(sectors: List[str]) -> List[str]:
    """
    Validates and normalizes sector list:
    - Converts to uppercase
    - Removes duplicates
    - Checks against allowed sectors
    - Returns cleaned list
    """
    if not sectors:
        return []  # Empty list is valid
    
    if not isinstance(sectors, list):
        raise HTTPException(status_code=422, detail="Sectors must be a list.")
    
    # Normalize: uppercase and strip whitespace
    normalized_sectors = [sector.strip().upper() for sector in sectors if sector and sector.strip()]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sectors = []
    for sector in normalized_sectors:
        if sector not in seen:
            seen.add(sector)
            unique_sectors.append(sector)
    
    # Validate each sector against allowed list
    invalid_sectors = [s for s in unique_sectors if s not in ALLOWED_SECTORS]
    if invalid_sectors:
        raise HTTPException(
            status_code=422, 
            detail=f"Invalid sectors: {', '.join(invalid_sectors)}. Allowed sectors: {', '.join(ALLOWED_SECTORS)}"
        )
    
    return unique_sectors