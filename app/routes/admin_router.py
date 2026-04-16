# app/routes/admin_router.py

"""
Admin Router - Global Sector File Management
Handles uploading and managing global regulatory files (RBI, GDPR, HIPAA, etc.)
"""

import os
import uuid
import asyncio
import aiofiles  # <--- ADDED for streaming
import shutil
import tempfile
import hashlib
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from app.config import settings
from app.dependencies import UserPayload, get_current_user
from app.schemas import StandardResponse
from app.models.Admin_file import GlobalFileDB
# from app.services import rag_service
from app.services.rag.file_indexing import process_and_index_global_file, delete_global_document_by_source

from app.utils.logger import setup_logger
from app.database import Global_file_collection
import magic

logger = setup_logger()

router = APIRouter()

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

# Define allowed document types
ALLOWED_DOCUMENT_TYPES = [
    "circular",
    "guidelines"
]


async def stream_to_disk(upload_file: UploadFile, dest_path: str):
    """
    Stream file directly to disk using shutil.copyfileobj in a thread.
    This avoids high memory usage and issues with aiofiles/event loops.
    """
    try:
        # Reset file cursor to the beginning (Crucial!)
        await upload_file.seek(0)
        
        # Run blocking I/O in a separate thread
        with open(dest_path, "wb") as buffer:
            await asyncio.to_thread(shutil.copyfileobj, upload_file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File streaming failed: {str(e)}")

async def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA256 hash from file on disk (memory safe).
    Reads in 8KB chunks to avoid loading large files into RAM.
    """
    def _hash_file():
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(8192), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    return await asyncio.to_thread(_hash_file)

# -------------------------------------------

def verify_admin(current_user: UserPayload):
    """Verify user has admin privileges."""
    if current_user.role.lower() != "admin":
        raise HTTPException(
            status_code=403,
            detail="Access denied. This endpoint requires administrator privileges."
        )

@router.post("/upload-global", tags=["Admin"])
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
    Upload a global sector file (Admin Only) with Memory-Safe Streaming.
    """
    
    # 1. VERIFY ADMIN PERMISSIONS
    verify_admin(current_user)
    
    # 2. VALIDATE SECTOR
    normalized_sector = sector.strip().upper()
    if normalized_sector not in ALLOWED_GLOBAL_SECTORS:
        raise HTTPException(status_code=400, detail=f"Invalid sector. Allowed: {', '.join(ALLOWED_GLOBAL_SECTORS)}")
    
    # 3. VALIDATE DOCUMENT TYPE
    normalized_document_type = document_type.strip().lower()
    if normalized_document_type not in ALLOWED_DOCUMENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid document type. Allowed: {', '.join(ALLOWED_DOCUMENT_TYPES)}")
    
    clean_category = category.strip()
    if not clean_category:
         raise HTTPException(status_code=400, detail="Category cannot be empty.")
    
    logger.info(f"[ADMIN] Uploading: {normalized_sector} | Type: {normalized_document_type} | Cat: {clean_category}")
    
    # 4. VALIDATE EXTENSION
    file_extension = os.path.splitext(file.filename)[1].lower()
    ALLOWED_FILES = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".txt": "text/plain"
    }
    
    if file_extension not in ALLOWED_FILES.keys():
        raise HTTPException(status_code=400, detail=f"File type not supported. Allowed: {list(ALLOWED_FILES.keys())}")
    
    # --- STREAMING LOGIC START ---
    
    # 5. CREATE TEMP FILE (Empty container)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close() # Close immediately so other functions can access it
    temp_file_path = temp_file.name

    try:
        # 6. STREAM TO DISK (Replaces unsafe file.read())
        await stream_to_disk(file, temp_file_path)

        # 7. VALIDATE SIZE (From Disk)
        file_size = os.path.getsize(temp_file_path)
        if file_size > settings.MAX_FILE_SIZE:
             raise HTTPException(status_code=400, detail=f"File size exceeds limit")

        # 8. DUPLICATE CHECK (HASHING FROM DISK)
        file_hash = await calculate_file_hash(temp_file_path)

        existing_file = await Global_file_collection.find_one({"file_hash": file_hash})
        if existing_file:
            existing_filename = existing_file.get('filename', 'Unknown')
            existing_sector = existing_file.get('sector', 'Unknown')
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "Duplicate file detected",
                    "message": f"This file already exists in the system (Sector: {existing_sector}).",
                    "existing_filename": existing_filename
                }
            )
        
        # 9. VALIDATE MIME TYPE (FROM DISK)
        try:
            mime_type = await asyncio.to_thread(magic.from_file, temp_file_path, mime=True)
            
            # Special handling for .docx
            if file_extension == ".docx" and mime_type == "application/zip":
                mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                
            expected_mime = ALLOWED_FILES.get(file_extension)
            if not expected_mime or mime_type != expected_mime:
                logger.warning(f"MIME mismatch for {file.filename}: Got {mime_type}")
                raise HTTPException(status_code=400, detail=f"File content does not match extension (Got {mime_type})")
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Validation error: {e}")
        
        # 10. GENERATE PATHS
        file_id = str(uuid.uuid4())
        safe_filename = file.filename.replace(" ", "_").replace("/", "")
        unique_filename = f"{file_id}-{safe_filename}"
        
        global_upload_dir = os.path.join(
            settings.UPLOAD_DIR, "global", normalized_sector, normalized_document_type
        )
        os.makedirs(global_upload_dir, exist_ok=True)
        permanent_file_path = os.path.join(global_upload_dir, unique_filename)
        
        # 11. PROCESS AND INDEX (Using Temp File)
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
        # Filter None values
        extra_metadata = {k: v for k, v in extra_metadata.items() if v is not None}
        
        result = await asyncio.to_thread(
            process_and_index_global_file,
            file_path=temp_file_path,
            sector=normalized_sector,
            file_id=file_id,
            original_filename=file.filename,
            extra_metadata=extra_metadata
        )
        
        # 12. SAVE TO PERMANENT STORAGE (Copy from Temp)
        # shutil.copy is thread-safe for files
        await asyncio.to_thread(shutil.copy, temp_file_path, permanent_file_path)

        relative_path = f"global/{normalized_sector}/{normalized_document_type}/{unique_filename}"
        file_url = f"{settings.SERVER_BASE_URL}/admin-files/{relative_path}"
        
        # 13. SAVE METADATA TO MONGODB
        file_record = GlobalFileDB(
            sector=normalized_sector,
            document_type=normalized_document_type,
            category=clean_category,
            description=description,
            effective_date=effective_date,
            version=version,
            filename=file.filename,
            uploaded_by=current_user.user_id,
            file_hash=file_hash, 
            file_id=file_id,
            file_path=permanent_file_path,
            file_url=file_url,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            is_active=True
        )
        
        await Global_file_collection.insert_one(file_record.model_dump())
        logger.info(f"[ADMIN] Metadata saved for {file_id}")
        
        # 14. RESPONSE
        return StandardResponse(
            status="success",
            status_code=200,
            message=f"Global file uploaded successfully",
            data={
                "file_id": file_id,
                "sector": normalized_sector,
                "filename": file.filename,
                "chunks_indexed": result.get('chunks_indexed', 0),
                "file_url": file_url
            }
        )
    
    except Exception as e:
        logger.error(f"[ADMIN] Global upload failed: {e}")
        
        # ROLLBACK
        try: await delete_global_document_by_source(filename=file.filename, sector=normalized_sector)
        except: pass

        try: await Global_file_collection.delete_one({"file_id": file_id})
        except: pass
            
        if 'permanent_file_path' in locals() and os.path.exists(permanent_file_path):
            try: os.remove(permanent_file_path)
            except: pass

        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    finally:
        # CLEANUP TEMP FILE
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try: os.unlink(temp_file_path)
            except: pass
        await file.close()


@router.get("/files", tags=["Admin"])
async def get_all_global_files(
    current_user: UserPayload = Depends(get_current_user)
):
    verify_admin(current_user)

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



@router.get("/global-sectors", tags=["Admin"])
async def list_global_sectors(
    current_user: UserPayload = Depends(get_current_user)
):
    """
    List all available global sectors with statistics (Admin Only)
    """
    verify_admin(current_user)
    
    from qdrant_client import QdrantClient, models
    
    client = QdrantClient(url=settings.QDRANT_URL)
    sectors = {}
    
    offset = None
    while True:
        try:
            results, offset = client.scroll(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="metadata.is_global", match=models.MatchValue(value=True))]
                ),
                limit=100,
                offset=offset,
                with_payload=True
            )
            
            for point in results:
                metadata = point.payload.get('metadata', {})
                source = metadata.get('source')
                filename = metadata.get('file_name')
                doc_type = metadata.get('document_type')
                
                if source not in sectors:
                    sectors[source] = {
                        "chunk_count": 0,
                        "files": set(),
                        "circular_count": 0,
                        "guidelines_count": 0
                    }
                
                sectors[source]["chunk_count"] += 1
                sectors[source]["files"].add(filename)
                
                if doc_type == "circular":
                    sectors[source]["circular_count"] += 1
                elif doc_type == "guidelines":
                    sectors[source]["guidelines_count"] += 1
            
            if offset is None:
                break
        except Exception as e:
            logger.error(f"[ADMIN] Error listing sectors: {e}")
            break
    
    sector_list = [
        {
            "sector": sector,
            "chunk_count": data["chunk_count"],
            "file_count": len(data["files"]),
            "circular_chunks": data["circular_count"],
            "guidelines_chunks": data["guidelines_count"],
            "files": sorted(list(data["files"]))
        }
        for sector, data in sorted(sectors.items())
    ]
    
    return StandardResponse(
        status="success",
        status_code=200,
        message=f"Found {len(sector_list)} global sectors",
        data=sector_list
    )


# @router.delete("/global/{sector}/{filename}", tags=["Admin"])
# async def delete_global_file(
#     sector: str,
#     filename: str,
#     current_user: UserPayload = Depends(get_current_user)
# ):
#     """
#     Delete a global sector file (Admin Only)
#     """
#     verify_admin(current_user)
    
#     normalized_sector = sector.strip().upper()
    
#     logger.info(f"[ADMIN] Deleting {filename} from {normalized_sector}")
    
#     try:
#         result = await rag_service.delete_global_document_by_source(
#             filename=filename,
#             sector=normalized_sector
#         )
        
#         db_result = await Global_file_collection.delete_one({
#             "filename": filename,
#             "sector": normalized_sector
#         })
        
#         msg = f"Deleted {filename} from {normalized_sector} (Vectors deleted, DB Record deleted: {db_result.deleted_count})"
        
#         return StandardResponse(
#             status="success",
#             status_code=200,
#             message=msg,
#             data=result
#         )
    
#     except Exception as e:
#         logger.error(f"[ADMIN] Delete failed: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Delete failed: {str(e)}"
#         )


# @router.get("/allowed-sectors", tags=["Admin"])
# async def get_allowed_sectors():
#     """
#     Get list of allowed global sectors and document types
#     """
#     return StandardResponse(
#         status="success",
#         status_code=200,
#         message="List of allowed global sectors and document types",
#         data={
#             "sectors": ALLOWED_GLOBAL_SECTORS,
#             "document_types": ALLOWED_DOCUMENT_TYPES,
#             "sector_count": len(ALLOWED_GLOBAL_SECTORS),
#             "document_type_count": len(ALLOWED_DOCUMENT_TYPES)
#         }
#     )