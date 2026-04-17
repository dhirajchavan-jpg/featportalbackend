# app/middleware/project_middleware.py
import re
from fastapi import HTTPException
from typing import List


# -------------------- Validate Project Name --------------------
def validate_project_name(name: str):
    if not name or len(name.strip()) < 3:
        raise HTTPException(status_code=422, detail="Project name must be at least 3 characters long.")
    if len(name) > 100:
        raise HTTPException(status_code=422, detail="Project name too long (max 100 characters).")
    if not re.match(r"^[A-Za-z0-9\s\-_,.&()]+$", name):
        raise HTTPException(status_code=422, detail="Project name contains invalid characters. Only letters, numbers, spaces, and basic punctuation are allowed.")


# -------------------- Validate Industry --------------------
def validate_industry(industry: str) -> str:
    if not industry:
        raise HTTPException(status_code=422, detail="Industry cannot be empty.")

    lowercase_industry = industry.lower().strip()

    if len(lowercase_industry) < 2:
        raise HTTPException(status_code=422, detail="Industry must be at least 2 characters long.")
    if len(lowercase_industry) > 50:
        raise HTTPException(status_code=422, detail="Industry too long (max 50 characters).")
    if not re.match(r"^[a-z\s&]+$", lowercase_industry):
        raise HTTPException(status_code=422, detail="Industry must contain only letters, spaces, and &.")

    return lowercase_industry


# -------------------- Validate Description --------------------
def validate_description(desc: str):
    if not desc or len(desc.strip()) < 10:
        raise HTTPException(status_code=422, detail="Project description must be at least 10 characters long.")
    if len(desc) > 1000:
        raise HTTPException(status_code=422, detail="Project description too long (max 1000 characters).")


# -------------------- Validate Sectors --------------------
def validate_sectors(sectors: List[str]) -> List[str]:
    """
    Normalizes sectors and enforces non-empty values.
    This middleware no longer validates against a fixed regulatory list.
    """
    if not isinstance(sectors, list):
        raise HTTPException(status_code=422, detail="Sectors must be a list.")

    normalized_sectors = [sector.strip().upper() for sector in sectors if sector and sector.strip()]

    seen = set()
    unique_sectors = []
    for sector in normalized_sectors:
        if sector not in seen:
            seen.add(sector)
            unique_sectors.append(sector)

    return unique_sectors
