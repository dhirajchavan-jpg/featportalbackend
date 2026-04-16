# app/middleware/files_middleware.py

from fastapi import UploadFile, HTTPException
import os, re
import asyncio
import magic
from bson import ObjectId
import aiofiles  # <--- ADDED: For async file IO
import shutil    # <--- ADDED: For moving files
import tempfile
import hashlib 
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Local Application Imports ---
from app.auth.jwt_handler import decode_access_token
from app.models.Files import FileModel
from app.database import file_collection, project_collection,User_collection
# from app.services import rag_service
from app.services.rag.file_indexing import process_and_index_file, delete_document_by_source
from app.dependencies import UserPayload
from app.config import settings
from app.services.redis.redis_service import redis_service # <--- ADDED: Redis Service Import
import mimetypes

# Directory to store uploaded files
DOC_FOLDER = os.path.join(os.getcwd(), "docs")
os.makedirs(DOC_FOLDER, exist_ok=True)

ALLOWED_MIME_TYPES = list(settings.ALLOWED_MIME_TYPES.values())
MAX_FILE_SIZE = settings.MAX_FILE_SIZE


#  Token Validation (unchanged)
async def validate_token(token: str) -> Dict[str, Any]:
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_id = payload.get("user_id")
    if not user_id or not payload.get("sub"):
        raise HTTPException(status_code=400, detail="User ID or email missing in token")
    return payload


#  Project Ownership Validation (unchanged)
async def verify_project(project_id: str, current_user: UserPayload):
    user_id = current_user.user_id
    project_id = project_id.lower().strip()
    try:
        project_obj_id = ObjectId(project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project ID format")
    
    project = await project_collection.find_one({"_id": project_obj_id})
    if not project:
        raise HTTPException(status_code=403, detail="The User is Not authorized for this project")
    return project


#  Filename Sanitization & Hashing (unchanged)
def sanitize_filename(filename: str) -> str:
    filename = re.sub(r"[^\w\s-]", "", filename)
    filename = re.sub(r"\s+", "_", filename)
    return filename

def calculate_sha256(content: bytes) -> str:
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content)
    return sha256_hash.hexdigest()

async def stream_to_temp_file(uploaded_file: UploadFile) -> str:
    """
    Streams UploadFile to a temporary file on disk using standard shutil.
    Runs in a thread to avoid blocking the event loop.
    """
    # 1. Create a temp file (not deleted automatically on close)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_path = temp_file.name
    
    try:
        # 2. Reset file cursor to the beginning (Crucial!)
        await uploaded_file.seek(0)
        
        # 3. Copy file data safely in a thread
        # uploaded_file.file is the underlying python file object
        await asyncio.to_thread(
            shutil.copyfileobj, 
            uploaded_file.file, 
            temp_file
        )
    except Exception as e:
        # Cleanup on failure
        temp_file.close()
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"File streaming failed: {str(e)}")
    finally:
        # Always close the file handle
        temp_file.close()
        
    return temp_path

# --- MODIFIED: Hash Calculation (Standard IO) ---
async def calculate_file_hash(file_path: str) -> str:
    """Reads file from disk in chunks to calculate SHA256"""
    sha256_hash = hashlib.sha256()
    
    def _read_and_hash():
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(8192), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    return await asyncio.to_thread(_read_and_hash)

#  File Saving with Validation (unchanged)
async def save_file(uploaded_file: UploadFile, project_id: str) -> (str, str, str):
    """
    Saves file safely by streaming to temp storage first.
    """
    original_filename = uploaded_file.filename
    # ... (Sanitization logic from before) ...
    from app.middleware.files_middleware import sanitize_filename # Ensure you have this
    name_part, ext_part = os.path.splitext(original_filename)
    filename_only = sanitize_filename(name_part)

    # 1. Stream to Temp File (Robust Method)
    temp_file_path = await stream_to_temp_file(uploaded_file)
    
    try:
        # 2. Validate Size
        file_size = os.path.getsize(temp_file_path)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size exceeds limit")
        
        # 3. Calculate Hash
        file_hash = await calculate_file_hash(temp_file_path)

        # 4. Check Duplicates (Hash)
        existing_hash = await file_collection.find_one({
            "project_id": project_id, 
            "file_hash": file_hash
        })
        if existing_hash:
            raise HTTPException(
                status_code=409, 
                detail=f"Duplicate file content detected. (File: {existing_hash.get('filename')})."
            )

        # 5. Validate MIME (from disk)
        mime_type = await asyncio.to_thread(magic.from_file, temp_file_path, mime=True)
        
        if mime_type not in ALLOWED_MIME_TYPES:
             raise HTTPException(status_code=400, detail=f"File MIME '{mime_type}' is not allowed.")

        # 6. Determine Extension & Final Path
        ext = mimetypes.guess_extension(mime_type)
        if not ext: ext = ext_part

        final_filename_with_ext = filename_only + ext
        final_file_path = os.path.join(DOC_FOLDER, final_filename_with_ext)

        # 7. Check Duplicate Filename
        existing_name = await file_collection.find_one({
            "project_id": project_id, 
            "filename": filename_only 
        })
        if existing_name:
            raise HTTPException(
                status_code=409, 
                detail=f"File with name '{filename_only}' already exists."
            )

        # 8. Move Temp to Final
        await asyncio.to_thread(shutil.move, temp_file_path, final_file_path)
        
        return filename_only, final_file_path, file_hash

    except Exception as e:
        # Cleanup
        if os.path.exists(temp_file_path):
            try: os.unlink(temp_file_path)
            except: pass
        
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")


#  MODIFIED: Metadata Storage (Now saves Category and Type)
async def store_file_metadata(
    project_id: str, 
    filename: str, 
    file_path: str, 
    file_hash: str,
    sector: str = "N/A", 
    category: str = "General",       
    compliance_type: str = "General",
    user_id: str = None  # <-- Added user_id parameter
) -> FileModel:

    file_doc = FileModel(
        project_id=project_id,
        filename=filename,
        file_url=file_path,
        file_hash=file_hash,
        sector=sector,
        category=category,
        compliance_type=compliance_type,
        user_id=user_id  # <-- Store the user ID
    )
    
    try:
        result = await file_collection.insert_one(file_doc.model_dump(exclude={"file_id"}))
        file_doc.file_id = str(result.inserted_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving metadata: {str(e)}")
    return file_doc


#  MODIFIED: Fetch File Metadata (Reads Category and Type)
async def get_file(
    project_id: str, 
    file_id: str, 
    current_user: UserPayload,
    check_physical_file: bool = True
) -> FileModel:
    try:
        file_obj_id = ObjectId(file_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file ID format")
    
    project_id = project_id.lower().strip()
    file_doc = await file_collection.find_one({"_id": file_obj_id, "project_id": project_id})
    
    if not file_doc:
        raise HTTPException(status_code=404, detail="File not found")

    await verify_project(project_id, current_user)

    file_path = file_doc.get("file_url")
    if check_physical_file:
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found on server")
            
    return FileModel(
        file_id=str(file_doc["_id"]),
        project_id=file_doc["project_id"],
        filename=file_doc["filename"],
        file_url=file_path,
        sector=file_doc.get("sector") or "N/A",
        category=file_doc.get("category", "General"),
        compliance_type=file_doc.get("compliance_type", "General"),
        user_id=file_doc.get("user_id") # <-- Ensure user_id is returned for permission checking
    )


#  MODIFIED: Qdrant Indexing (Passes new fields AND generates ID if needed)
async def run_qdrant_indexing(
    file_path: str, 
    project_id: str, 
    sector: str, 
    current_user: UserPayload,
    category: str = "General",      
    compliance_type: str = "General",
    original_filename: str = None, 
    ocr_engine: str = "paddleocr",
    file_id: str = None, # If None, we generate it here
    file_hash: str = None # <--- ADDED: Needed for DB save in worker
):
    """
    Enqueues the file indexing job to Redis.
    The Worker will pick this up, process it, and THEN save metadata to MongoDB.
    """
    try:
        logging.info(f"INFO: Enqueuing indexing job for: {original_filename}")
        
        # 1. Generate a File ID if one wasn't provided (Crucial for the new flow)
        if not file_id:
            file_id = str(ObjectId())

        # Prepare User Data for Worker reconstruction
        user_data = {
            "user_id": current_user.user_id,
            "email": current_user.email,
            "role": current_user.role
        }
        
        # Prepare Job Data
        job_data = {
            "file_path": file_path,
            "project_id": project_id,
            "sector": sector,
            "category": category,
            "compliance_type": compliance_type,
            "original_filename": original_filename,
            "ocr_engine": ocr_engine,
            "file_id": file_id,    # Pass the ID (either passed in or generated here)
            "file_hash": file_hash, # Pass the hash so worker can save it
            "doc_type": "general"
        }
        
        # Enqueue
        task_id = await redis_service.enqueue_job(
            job_type="file_upload",
            job_data=job_data,
            user_data=user_data
        )

        logging.info(f"INFO: Job Enqueued | Task ID: {task_id} | Future File ID: {file_id}")
        return task_id
        
    except Exception as e:
        logging.error(f"ERROR: Failed to enqueue indexing job: {e}")
        # In production, you might want to raise here or handle it gracefully.
        # Since this is critical for the user to get their file processed, we raise.
        raise HTTPException(status_code=500, detail=f"Failed to enqueue indexing job: {e}")


#  MODIFIED: Fetch all files (Returns Category and Type for UI)
async def get_all_files_for_project(project_id: str, current_user: UserPayload) -> List[FileModel]:
    project_id = project_id.lower().strip()
    await verify_project(project_id, current_user)


    # Base query
    query = {"project_id": project_id}


    # FILTER: If user is NOT admin
    if current_user.role != "admin":
        # Start by allowing the user to see their own files
        allowed_user_ids = [current_user.user_id]


        try:
            # Fetch the user's profile to check for an 'admin_id'
            user_obj_id = ObjectId(current_user.user_id)
            user_doc = await User_collection.find_one({"_id": user_obj_id})


            # If the user has an admin_id, add it to the allowed list
            # This allows the user to see files uploaded by their Admin
            if user_doc and user_doc.get("admin_id"):
                allowed_user_ids.append(str(user_doc["admin_id"]))
       
        except Exception as e:
            # If fetching fails, log it and fail safe (show only own files)
            logging.error(f"Error fetching admin_id for visibility: {e}")


        # Modify query to find files where user_id is EITHER the user's OR their admin's
        query["user_id"] = {"$in": allowed_user_ids}


    files_cursor = file_collection.find(query)
   
    file_list = []
    async for file_doc in files_cursor:
        file_list.append(
            FileModel(
                file_id=str(file_doc["_id"]),
                project_id=file_doc["project_id"],
                filename=file_doc["filename"],
                file_url=file_doc.get("file_url"),
                sector=file_doc.get("sector") or "N/A",
                category=file_doc.get("category", "General"),
                compliance_type=file_doc.get("compliance_type", "General"),
                user_id=file_doc.get("user_id")
            )
        )
   
    return file_list



#  Deletion Logic (unchanged)
# --- MODIFIED: Complete Deletion (Qdrant + Disk + MongoDB) ---
async def handle_file_deletion(file_doc: FileModel, project_id: str, user: UserPayload):
    """
    Deletes a file from:
    1. Qdrant (Vectors & Chat History Reference)
    2. MongoDB Metadata
    
    PRESERVES: The physical file on disk (it will remain in the 'docs/' folder).
    """
    logging.info(f"INFO: Starting soft-deletion for file: {file_doc.filename}")

    # 1. Delete from Qdrant (Vectors & Chat History)
    try:
        logging.info(f"INFO: Deleting vectors from Qdrant for project: {project_id}, file: {file_doc.filename}")
        qdrant_result = await delete_document_by_source(
            filename=file_doc.filename, 
            project_id=project_id, 
            current_user=user
        )
        logging.info(f"INFO: Qdrant deletion result: {qdrant_result}")
    except Exception as e:
        logging.error(f"ERROR: Failed to delete from Qdrant (Proceeding with DB delete): {e}")

    # 2. (SKIPPED) Delete Physical File from Disk
    # We purposefully comment this out to keep the file for audit/backup.
    # if file_doc.file_url and os.path.exists(file_doc.file_url):
    #     try:
    #         os.remove(file_doc.file_url)
    #         print(f"INFO: Physical file deleted: {file_doc.file_url}")
    #     except Exception as e:
    #         print(f"ERROR: Failed to delete physical file: {e}")
    logging.info(f"INFO: Skipping physical file deletion. File remains at: {file_doc.file_url}")

    # 3. Delete Metadata from MongoDB
    try:
        file_obj_id = ObjectId(file_doc.file_id)
        result = await file_collection.delete_one({"_id": file_obj_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="File metadata not found in database.")

        logging.info(f"INFO: Successfully deleted metadata for file_id: {file_doc.file_id}")

    except Exception as e:
        logging.error(f"ERROR: Failed to delete metadata for {file_doc.file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file metadata: {e}")