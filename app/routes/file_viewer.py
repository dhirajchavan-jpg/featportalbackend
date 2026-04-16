from fastapi import APIRouter, Depends, HTTPException, Body, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
from bson import ObjectId
import os
from pydantic import BaseModel
from app.config import settings

from app.auth.jwt_bearer import JWTBearer
from app.dependencies import get_current_user, UserPayload
from app.database import Global_file_collection, file_collection
from app.middleware.files_middleware import verify_project
from app.utils.logger import setup_logger
logger = setup_logger()

router = APIRouter(tags=["File Viewer"])

# ---------------------------------------------------------
#  1. SMART PATH RESOLVER (Handles Spaces vs Underscores)
# ---------------------------------------------------------
# --- 1. ROBUST PATH RESOLVER (Fixes Double Dots & Folders) ---
def resolve_physical_path(raw_path: str):
    """
    Finds the file on disk by:
    1. Checking the exact path.
    2. Fixing extension typos (..pdf -> .pdf).
    3. Checking spaces vs underscores.
    4. Searching in 'docs', 'uploads', and 'root' folders.
    """
    if not raw_path:
        return None

    # Normalize separators for Windows
    raw_path = raw_path.replace("\\", os.sep).replace("/", os.sep)
    project_root = os.getcwd()
    
    # Extract the filename from the DB path
    filename = os.path.basename(raw_path)
    
    # -------------------------------------------------
    # A. GENERATE FILENAME VARIATIONS
    # -------------------------------------------------
    name_candidates = [filename]

    # Fix common "Double Dot" typo (e.g., file..pdf -> file.pdf)
    if "..pdf" in filename:
        name_candidates.append(filename.replace("..pdf", ".pdf"))
    
    # Add Space/Underscore variations for ALL candidates so far
    final_names = []
    for name in name_candidates:
        final_names.append(name)
        final_names.append(name.replace("_", " ")) # Try spaces
        final_names.append(name.replace(" ", "_")) # Try underscores
    
    # Remove duplicates
    final_names = list(dict.fromkeys(final_names))

    # -------------------------------------------------
    # B. GENERATE FOLDER LOCATIONS
    # -------------------------------------------------
    # We will look for the file in these folders
    folders_to_check = [
        os.path.dirname(raw_path),       # 1. The folder listed in the DB (e.g., E:\...\docs)
        os.path.join(project_root, "docs"),    # 2. Explicit 'docs' folder
        os.path.join(project_root, "uploads"), # 3. Explicit 'uploads' folder
        project_root                     # 4. Project root
    ]

    # -------------------------------------------------
    # C. CHECK ALL COMBINATIONS
    # -------------------------------------------------
    for folder in folders_to_check:
        # Skip empty folders (like if dirname returned empty)
        if not folder: continue
        
        for name in final_names:
            candidate_path = os.path.join(folder, name)
            if os.path.exists(candidate_path):
                return candidate_path
            
    return None


class FileNameRequest(BaseModel):
    filename: str

# ---------------------------------------------------------
#  2. GET FILE PATH (Fuzzy DB Search)
# ---------------------------------------------------------
@router.post("/files/get-file-path")
async def get_file_path(
    request: Request,
    payload: FileNameRequest,
    current_user: UserPayload = Depends(get_current_user)
):
    original_name = payload.filename
    # print(f">>> LOOKING UP: {original_name} <<<")

    # Helper to search both collections
    async def find_in_db(name_to_search):
        doc = await Global_file_collection.find_one({"filename": name_to_search})
        if not doc:
            doc = await file_collection.find_one({"filename": name_to_search})
        return doc

    # Generate Candidate Names for DB Search
    candidates = [original_name]

    # 1. Handle Extensions and Trailing Dots (e.g., "File..pdf" -> "File")
    if "." in original_name:
        no_ext = original_name.rsplit(".", 1)[0]
        candidates.append(no_ext)
        candidates.append(no_ext.rstrip(".")) 

    # 2. Handle Space/Underscore Variations
    final_candidates = []
    for name in candidates:
        final_candidates.append(name)
        final_candidates.append(name.replace(" ", "_")) # Try underscores
        final_candidates.append(name.replace("_", " ")) # Try spaces

    # Clean duplicates
    final_candidates = list(dict.fromkeys(final_candidates))

    # Search DB loop
    file_doc = None
    for candidate in final_candidates:
        file_doc = await find_in_db(candidate)
        if file_doc:
            # print(f">>> Match found for: {candidate} <<<")
            break

    if not file_doc:
        logger.warning(">>> File not found in DB after fuzzy search <<<")
        return JSONResponse(status_code=404, content={"message": "File not found"})

    # Extract ID
    file_id = str(file_doc.get("file_id") or file_doc.get("_id"))

    # Extract Token
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "") if "Bearer " in auth_header else ""
    
    base_url = settings.BACKEND_PUBLIC_URL.rstrip("/")
    # Build URL (Dynamic Base URL)
    # base_url = str(request.base_url).rstrip("/")
    # secure_url = f"{base_url}/files/view/{file_id}?token={token}"

    # Use this instead:
    base_url = settings.BACKEND_PUBLIC_URL.rstrip("/")
    secure_url = f"{base_url}/files/view/{file_id}?token={token}"
    logger.info(f"Generated secure file URL: {secure_url}")


    
    return {"file_url": secure_url}


# ---------------------------------------------------------
#  3. FLEXIBLE AUTH HELPER
# ---------------------------------------------------------
async def get_user_flexible(
    request: Request,
    token: str = Query(None)
):
    if "Authorization" in request.headers:
        token_str = await JWTBearer()(request)
        return get_current_user(token=token_str)
    
    if token:
        return get_current_user(token=token)
        
    raise HTTPException(status_code=401, detail="Not authenticated")


# ---------------------------------------------------------
#  4. VIEW FILE ENDPOINT (Uses Smart Resolver)
# ---------------------------------------------------------
@router.get("/files/view/{file_id}")
async def view_file(
    file_id: str,
    current_user: UserPayload = Depends(get_user_flexible), 
):
    # print(f">>> VIEWING ID: {file_id} <<<")
    file_doc = None
    
    # 1. Find Record (Try FileID string, then ObjectId)
    try:
        # Check Global
        file_doc = await Global_file_collection.find_one({"file_id": file_id})
        if not file_doc:
            file_doc = await Global_file_collection.find_one({"_id": ObjectId(file_id)})
        
        # Check Project
        if not file_doc:
            file_doc = await file_collection.find_one({"file_id": file_id})
        if not file_doc:
            file_doc = await file_collection.find_one({"_id": ObjectId(file_id)})
            
    except Exception:
        pass

    if not file_doc:
        raise HTTPException(status_code=404, detail="File record not found")

    # 2. Get Raw Path (Check both potential fields)
    raw_path = file_doc.get("file_path") or file_doc.get("file_url")

    if not raw_path:
        raise HTTPException(status_code=404, detail="Path missing in database record")

    # 3. Resolve Physical Path (The Fix)
    # ... inside view_file ...

    # 3. Resolve Physical Path (The Fix)
    final_path = resolve_physical_path(raw_path)  # <--- MUST CALL THE NEW FUNCTION

    if not final_path:
        logger.error(f"ERROR: File '{raw_path}' not found on disk (Checked variations).")
        raise HTTPException(status_code=404, detail="File content not found on server")

    # print(f"-> Streaming from: {final_path}")

    # ... return StreamingResponse ...

    def iter_file():
        with open(final_path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk

    return StreamingResponse(
        iter_file(),
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"}
    )