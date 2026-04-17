# app/routes/super_admin.py

from fastapi import APIRouter, HTTPException, status, Depends, Form, File, UploadFile
import tempfile
import hashlib
import uuid
import magic
from app.services.document_processing.ocr_engine import get_ocr_engine
from app.dependencies import UserPayload, get_current_user
from app.database import super_admin_collection, User_collection, file_collection, Global_file_collection
from app.models.super_admin import (
    SuperAdminLogin,
    OrgAdminCreate,
    OrgAdminResponse,
    OrgAdminUpdate,
    DashboardStats
)
from app.auth.jwt_handler import create_access_token, decode_access_token
from app.auth.jwt_bearer import JWTBearer
from passlib.context import CryptContext
from datetime import datetime
from bson import ObjectId
from typing import List, Optional
# from app.services import rag_service
from app.services.rag.file_indexing import delete_global_document_by_source, process_and_index_global_file
from app.utils.logger import setup_logger
from app.schemas import StandardResponse
from app.config import settings
from datetime import datetime, timezone
from app.models.Admin_file import GlobalFileDB
import asyncio
from app.services.redis.redis_service import redis_service # <--- ADDED: Redis Service Import

import os

logger = setup_logger()
# Setup Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter()

ALLOWED_DOCUMENT_TYPES = [
    "circular",
    "guidelines"
]

# Define allowed global sectors
ALLOWED_GLOBAL_SECTORS = [
    "RBI",      # Reserve Bank of India
    "SEBI",     # Securities and Exchange Board of India
    "GDPR",     # General Data Protection Regulation
    "HIPAA",    # Health Insurance Portability and Accountability Act
    "CCPA",     # California Consumer Privacy Act
    "SOX",      # Sarbanes-Oxley Act
    "PCI_DSS",  # Payment Card Industry Data Security Standard
    "NIST",     # National Institute of Standards and Technology
    "ISO_27001" # ISO/IEC 27001
]


# --- Helpers ---
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

async def verify_super_admin(token: str = Depends(JWTBearer())):
    """
    Dependency to ensure the requester is a Super Admin.
    """
    payload = decode_access_token(token)
    if not payload or payload.get("role") != "super_admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Super Admin privileges required."
        )
    return payload

# ==========================================
# 1. Super Admin Login
# ==========================================
# @router.post("/login")
# async def super_admin_login(credentials: SuperAdminLogin):
#     """
#     Login for Super Admin. Checks 'super_admin_collection'.
#     """
#     admin = await super_admin_collection.find_one({"username": credentials.username})
    
#     if not admin or not verify_password(credentials.password, admin["password"]):
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid Super Admin credentials"
#         )
    
#     if admin.get("role") != "super_admin":
#          raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Unauthorized. Only Super Admins can access this dashboard."
#         )
    
#     token_data = {
#         "user_id": str(admin["_id"]),
#         "sub": admin["username"],
#         "role": "super_admin"
#     }
#     access_token = create_access_token(token_data)
    
#     return {
#         "access_token": access_token,
#         "token_type": "bearer",
#         "role": "super_admin",
#         "username": admin["username"]
#     }

# ==========================================
# 6. Dashboard Statistics
# ==========================================
@router.get("/stats", response_model=DashboardStats, dependencies=[Depends(verify_super_admin)])
async def get_dashboard_stats():
    # Count Admins (excluding super_admin)
    total_admins = await super_admin_collection.count_documents({"role": "admin"})
    total_users = await User_collection.count_documents({"role": "user"})
    total_files = await file_collection.count_documents({})

    return {
        "total_admins": total_admins,
        "total_users": total_users,
        "total_files": total_files
    }


@router.post("/upload-global")
async def upload_global_sector_file(
    sector: str = Form(..., description="Sector name (e.g., RBI, GDPR, HIPAA)"),
    document_type: str = Form(..., description="Document type: 'circular' or 'guidelines'"),
    category: str = Form(..., min_length=1, description="Category is required"),
    description: Optional[str] = Form(None, description="Optional file description"),
    effective_date: Optional[str] = Form(None, description="Effective date (YYYY-MM-DD)"),
    version: Optional[str] = Form(None, description="Document version"),
    file: UploadFile = File(...),
    current_user: UserPayload = Depends(get_current_user),
):
    """
    Upload a global sector file (Admin Only) with Duplicate Detection
    """

    # 1. VERIFY ADMIN PERMISSIONS
    if current_user.role != "super_admin":
        logger.warning(f"[AUTH ALERT] User '{current_user.user_id}' tried to upload global files without permission.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Super Admin privileges required."
         )

    # 2. NORMALIZE SECTOR (organization source)
    normalized_sector = " ".join((sector or "").strip().split()).upper()
    if not normalized_sector:
        raise HTTPException(status_code=422, detail="Organization sector is required")

    # 3. VALIDATE DOCUMENT TYPE
    normalized_document_type = document_type.strip().lower()
    if normalized_document_type not in ALLOWED_DOCUMENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document type. Allowed types: {', '.join(ALLOWED_DOCUMENT_TYPES)}"
        )
    
    # Normalize Category (Required)
    clean_category = (category or "Organization").strip()
    if not clean_category:
         raise HTTPException(status_code=400, detail="Category cannot be empty or just whitespace.")
    
    logger.info(f"[ADMIN] Uploading global file: {normalized_sector} | Type: {normalized_document_type} | Cat: {clean_category}")
    
    # 4. VALIDATE FILE TYPE (Updated for PDF, DOCS, TXT)
    file_extension = os.path.splitext(file.filename)[1].lower()
    ALLOWED_FILES = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".txt": "text/plain"
    }
    
    if file_extension not in ALLOWED_FILES.keys():
        raise HTTPException(status_code=400, detail=f"File type not supported. Allowed: {list(ALLOWED_FILES.keys())}")
    
    # 5. READ CONTENT & VALIDATE SIZE
    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File size exceeds limit")
    
    
    # 6. DUPLICATE CHECK (HASHING)
    
    # Calculate SHA-256 hash
    file_hash = hashlib.sha256(content).hexdigest()

    # Check MongoDB for existing hash
    existing_file = await Global_file_collection.find_one({"file_hash": file_hash})
    
    if existing_file:
        existing_filename = existing_file.get('filename', 'Unknown')
        existing_sector = existing_file.get('sector', 'Unknown')
        logger.warning(f"[ADMIN] Duplicate upload attempt. Hash: {file_hash} matches {existing_filename}")
        
        raise HTTPException(
            status_code=409, # Conflict
            detail={
                "error": "Duplicate file detected",
                "message": f"This file already exists in the system (Sector: {existing_sector}).",
                "existing_filename": existing_filename
            }
        )
    

    # 7. VALIDATE MIME TYPE
    try:
        mime_type = await asyncio.to_thread(magic.from_buffer, content, mime=True)
        
        # Special handling for .docx which often appears as application/zip
        if file_extension == ".docx" and mime_type == "application/zip":
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            
        expected_mime = ALLOWED_FILES.get(file_extension)
        if not expected_mime or mime_type != expected_mime:
            logger.warning(f"MIME mismatch for {file.filename}: Got {mime_type}, expected {expected_mime}")
            raise HTTPException(status_code=400, detail=f"File content does not match extension (Got {mime_type})")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {e}")
    
    # 8. GENERATE FILE ID AND PATHS
    file_id = str(uuid.uuid4())
    safe_filename = file.filename.replace(" ", "_").replace("/", "")
    unique_filename = f"{file_id}-{safe_filename}"
    
    # Directory structure: global/SECTOR/type/
    global_upload_dir = os.path.join(
        settings.UPLOAD_DIR, "global", normalized_sector, normalized_document_type
    )
    os.makedirs(global_upload_dir, exist_ok=True)
    permanent_file_path = os.path.join(global_upload_dir, unique_filename)
    
    # -------------------------------------------------------------
    # 9. SAVE TO PERMANENT STORAGE (SYNC for Worker Access)
    # -------------------------------------------------------------
    try:
        def write_permanent_file():
            with open(permanent_file_path, "wb") as buffer:
                buffer.write(content)
        await asyncio.to_thread(write_permanent_file)
    except Exception as e:
        logger.error(f"[ADMIN] Failed to write file to disk: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file to server storage")
    
    # 10. CHECK OCR AVAILABILITY
    selected_engine = "paddleocr" 
    
    if selected_engine == "paddleocr":
        ocr_engine = get_ocr_engine()
        if not ocr_engine.is_service_available():
            # Cleanup file if OCR is dead
            try:
                os.unlink(permanent_file_path)
            except:
                pass
            
            raise HTTPException(
                status_code=503,
                detail="PaddleOCR is not available. Please try again later."
            )
    
    # -------------------------------------------------------------
    # 11. SAVE METADATA (MONGO) & ENQUEUE (REDIS)
    # -------------------------------------------------------------
    try:
        relative_path = f"global/{normalized_sector}/{normalized_document_type}/{unique_filename}"
        file_url = f"{settings.SERVER_BASE_URL}/admin-files/{relative_path}"
        
        extra_metadata = {
            "file_name": file.filename,
            "document_type": normalized_document_type,
            "category": clean_category,
            "description": description,
            "effective_date": effective_date,
            "version": version,
            "uploaded_by": current_user.user_id,
            "is_global": True,
            "file_hash": file_hash 
        }
        # Filter out None values for metadata
        extra_metadata = {k: v for k, v in extra_metadata.items() if v is not None}
        
        # SAVE METADATA TO MONGODB
        file_record = GlobalFileDB(
            sector=normalized_sector,
            document_type=normalized_document_type,
            category=clean_category,
            description=description,
            effective_date=effective_date,
            version=version,
            filename=file.filename,
            uploaded_by=current_user.user_id,
            file_hash=file_hash, # <--- Saving Hash
            file_id=file_id,
            file_path=permanent_file_path,
            file_url=file_url,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            is_active=True
        )
        file_record.file_url = file_url
        
        await Global_file_collection.insert_one(file_record.model_dump())
        logger.info(f"[ADMIN] Metadata saved to MongoDB for {file_id}")
        
        # ENQUEUE JOB TO REDIS
        user_data = {
            "user_id": current_user.user_id,
            "email": current_user.email,
            "role": current_user.role
        }

        job_data = {
            "file_path": permanent_file_path, # Worker uses this path
            "sector": normalized_sector,
            "file_id": file_id,
            "original_filename": file.filename,
            "extra_metadata": extra_metadata,
            "ocr_engine": selected_engine 
        }

        task_id = await redis_service.enqueue_job(
            job_type="global_file_upload",
            job_data=job_data,
            user_data=user_data
        )
        logger.info(f"[ADMIN] Enqueued Global Indexing Job: {task_id}")
        
        # 13. RESPONSE (Accepted)
        return StandardResponse(
            status="success",
            status_code=202,
            message=f"Global file uploaded and queued for processing",
            data={
                "file_id": file_id,
                "task_id": task_id,
                "sector": normalized_sector,
                "filename": file.filename,
                "file_hash": file_hash,
                "status": "processing",
                "file_data": file_record.model_dump()
            }
        )
    
    except Exception as e:
        logger.error(f"[ADMIN] Global upload failed: {e}")
        
        # ROLLBACK
        try:
            await Global_file_collection.delete_one({"file_id": file_id})
        except Exception:
            pass
            
        try:
            if os.path.exists(permanent_file_path):
                os.unlink(permanent_file_path)
        except Exception:
            pass
            
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    finally:
        await file.close()

@router.get("/files")
async def get_all_global_files(
    current_user: UserPayload = Depends(get_current_user)
):
    # verify_admin(current_user)

    try:
        docs = await Global_file_collection.find({}).to_list(None)
        

        # Convert ObjectId and datetime fields
        files_list = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])

            if "created_at" in doc:
                doc["created_at"] = doc["created_at"].isoformat() if hasattr(doc["created_at"], "isoformat") else doc["created_at"]
            if "updated_at" in doc:
                doc["updated_at"] = doc["updated_at"].isoformat() if hasattr(doc["updated_at"], "isoformat") else doc["updated_at"]

            files_list.append(doc)

        return StandardResponse(
            status="success",
            status_code=200,
            message=f"Retrieved {len(files_list)} global files",
            data=files_list
        )

    except Exception as e:
        logger.error(f"[ADMIN] ERROR in GET /files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    

@router.delete("/global/{sector}/{filename}")
async def delete_global_file(
    sector: str,
    filename: str,
    current_user: UserPayload = Depends(get_current_user)
):
    """
    Delete a global sector file.
    STRICT ACCESS: Only 'super_admin' role is allowed.
    """

    # --- 1. STRICT SUPER ADMIN CHECK ---
    if current_user.role != "super_admin":
        logger.warning(
            f"[AUTH ALERT] User '{current_user.user_id}' (Role: {current_user.role}) "
            f"tried to delete global files."
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access Denied: Only Super Admins can perform this action."
        )

    normalized_sector = sector.strip().upper()

    # --- 2. Find file metadata ---
    file_doc = await Global_file_collection.find_one({
        "filename": filename,
        "sector": normalized_sector
    })

    if not file_doc:
        raise HTTPException(status_code=404, detail="File not found")

    logger.info(
        f"[ADMIN] Super Admin '{current_user.user_id}' deleting "
        f"{filename} | Sector: {normalized_sector}"
    )

    try:
        # A. Delete from Qdrant (RAG chunks)
        # FIX: Added 'await' because this is likely an async DB operation
        await delete_global_document_by_source(
            filename=filename,
            sector=normalized_sector
        )
        
        # B. Delete physical file
        file_path = file_doc.get("file_path")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"[ADMIN] Deleted file from disk: {file_path}")

        # C. Delete MongoDB metadata
        await Global_file_collection.delete_one({"_id": file_doc["_id"]})

        return StandardResponse(
            status="success",
            status_code=200,
            message=f"Successfully deleted {filename}",
            data=None
        )

    except Exception as e:
        logger.error(f"[ADMIN] Global delete failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete global file: {str(e)}"
        )





