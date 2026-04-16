# app/routes/file_router.py

from fastapi import APIRouter, Depends, UploadFile, File as FastAPIFile, HTTPException, Form
from typing import List, Optional,Dict,Any
from app.database import project_collection
from bson import ObjectId
import logging
import os
from app.services.document_processing.ocr_engine import get_ocr_engine
from collections import defaultdict
import asyncio
from bson import ObjectId
from fastapi import Request
from app.services.retrieval.vector_retriever import get_vector_retriever
from app.database import db
from datetime import datetime
from app.models.UserFileSelection import UserFileSelectionModel
from app.middleware.role_checker import require_roles
from app.database import db, project_config_collection
# --- Local Application Imports ---
from app.auth.jwt_bearer import JWTBearer
from app.middleware import files_middleware
from app.models.Files import FileModel
from app.services.redis.redis_service import redis_service # <--- ADDED

logger = logging.getLogger(__name__)

# --- Use standard dependency and response models ---
from app.dependencies import get_current_user, UserPayload
from app.schemas import (
    StandardResponse, 
    FileUploadResponseData, 
    BulkUploadResponseData, 
    FileUploadErrorData
)
from pydantic import BaseModel


router = APIRouter()

# --- MODIFIED: Cancellation now targets Task ID for Redis ---
class CancelRequest(BaseModel):
    task_id: str 

@router.post("/cancel_upload")
async def cancel_upload_signal(req: CancelRequest):
    """
    Marks the job as cancelled in Redis.
    The worker will see this flag and perform the rollback (delete data).
    """
    if not req.task_id:
        raise HTTPException(status_code=400, detail="Task ID is required")
        
    await redis_service.cancel_job(req.task_id)
    return {"status": "cancelled", "task_id": req.task_id}


# --- FIXED: Task Status Endpoint for Frontend Polling ---
@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Check the status of a background processing task.
    Returns: 'pending', 'processing', 'completed', 'failed', or 'cancelled'
    """
    # Fetch the full job data object from Redis (contains both status and result/data)
    job_info = await redis_service.get_job_status(task_id)
    
    if not job_info:
        return StandardResponse(
            status="error",
            status_code=404,
            message="Task ID not found or expired.",
            data=None
        )

    # RedisService saves the result in the 'data' key
    return StandardResponse(
        status="success",
        status_code=200,
        message="Task status retrieved",
        data={
            "task_id": task_id,
            "status": job_info.get("status"),
            "result": job_info.get("data") 
        }
    )


# --- MODIFIED: This endpoint now handles BULK uploads with CATEGORIES ---
# NOTE: Changed response_model to Any or generic StandardResponse because we return task_ids now
@router.post("/projects/{project_id}/files", response_model=StandardResponse[Dict[str, Any]])
async def upload_files_bulk(
    project_id: str,
    request: Request,   # <-- ADD THIS HERE
    files: List[UploadFile] = FastAPIFile(...), 
    categories: List[str] = Form(...),       
    compliance_types: List[str] = Form(...),
    # sectors: List[str] = Form(default=[]),             
    current_user: UserPayload = Depends(require_roles("admin", "user"))
):
    """
    Handles BULK file upload.
    - Admin: Can upload to any Category.
    - User: Can ONLY upload to 'Organization' category.
    """

    # --- Initial validation ---
    if not files:
        raise HTTPException(status_code=422, detail="No files were uploaded.")
        
    if len(files) != len(categories) or len(files) != len(compliance_types):
        raise HTTPException(
            status_code=422, 
            detail="Count mismatch between files, categories, or compliance types."
        )
    
    # --- ROLE CHECK: Enforce Category Restrictions ---
    if current_user.role != "admin":
        for cat in categories:
            if cat.strip() != "Organization":
                raise HTTPException(
                    status_code=403,
                    detail="Restricted: Users can only upload files to the 'Organization' category."
                )

    # --- Normalize project_id once ---
    # project_id = project_id.lower().strip()

    # 1. Verify project ownership/access
    project = await files_middleware.verify_project(project_id, current_user)

    # ---------------------------------------------------------------------
    # --- NEW: DETERMINE OCR ENGINE STRATEGY (Project Config Check) ---
    # ---------------------------------------------------------------------
    selected_engine = "paddleocr" # Default per requirements
    try:
        # Fetch the project configuration
        # Assuming collection name is 'project_configs' based on your schema usage
        project_config = await db.project_configs.find_one({"project_id": project_id})
        
        if project_config:
            config_engine = project_config.get("ocr_engine")
            
            # Explicitly check for 'easyocr', otherwise default to paddle
            if config_engine == "paddleocr":
                selected_engine = "paddleocr"
            elif config_engine == "easyocr":
                selected_engine = "easyocr"
            else:
                selected_engine = "paddleocr"
                
            logger.info("[Upload] Project Config Found | project_id=%s | ocr_engine=%s",project_id,selected_engine)

        else:
            logger.warning(
    "[Upload] No Project Config Found | project_id=%s | default_ocr_engine=%s",
    project_id,
    selected_engine
)


    except Exception as e:
        logger.exception(
    "[Upload] Failed to fetch project config | project_id=%s | Defaulting OCR to paddleocr",
    project_id
)

        selected_engine = "paddleocr"
    # ---------------------------------------------------------------------
    if selected_engine == "paddleocr":
        ocr_instance = get_ocr_engine()
        if not ocr_instance.is_service_available():
            raise HTTPException(
                status_code=503, # Service Unavailable
                detail="PaddleOCR is not available. If you want to upload, change setting PaddleOCR to EasyOCR or try again later."
            )
    
    # MODIFIED: Changed type hint to allow returning Task IDs
    successful_uploads: List[Dict[str, Any]] = []
    failed_uploads: List[FileUploadErrorData] = []

    # --- Loop through each file ---
    for i, uploaded_file in enumerate(files):

        # --- ADD CANCELLATION HANDLING (REPLACE OLD BLOCK) ---
        if await request.is_disconnected():
            logger.warning(
    "[Upload] Client disconnected | project_id=%s | Upload aborted",
    project_id
)
            return StandardResponse(
                status="error",
                status_code=499,
                message="Client cancelled upload.",
                data={"successful_uploads": [], "failed_uploads": []}
            )
        # -----------------------------------------------------


        file_path = None
        try:
            current_category = categories[i].strip()
            current_compliance_type = compliance_types[i].strip()
            
            if current_category.lower() == "organization":
                # Organization sector ALWAYS comes from project creation
                if not project.get("has_organizational_compliance"):
                    raise HTTPException(
                        status_code=400,
                        detail="Organization compliance is not enabled for this project"
                    )

                if not project.get("organization_sector"):
                    raise HTTPException(
                        status_code=500,
                        detail="Organization sector missing in project configuration"
                    )

                current_sector = project["organization_sector"].strip().upper()


            else:
                # Regulatory / Third-party (super admin flow)
                # Keep your existing logic (sector from upload)
                current_sector = "N/A"


            # 2. Save physical file
            filename, file_path, file_hash = await files_middleware.save_file(
                uploaded_file, 
                project_id
            )

            # ---------------------------------------------------------------------
            # MODIFIED FLOW: DO NOT Save Metadata Here -> Pass to Job Directly
            # ---------------------------------------------------------------------
            
            # [REMOVED] Metadata storage in DB is now handled by the Worker after processing
            # file_doc = await files_middleware.store_file_metadata(...) 

            # 4. ENQUEUE JOB (Pass all metadata so Worker can create the DB entry later)
            # This calls Redis via the middleware
            task_id = await files_middleware.run_qdrant_indexing(
                file_path=file_path,
                project_id=project_id,
                sector=current_sector,
                current_user=current_user,
                category=current_category,      
                compliance_type=current_compliance_type,
                original_filename=uploaded_file.filename,
                ocr_engine=selected_engine,
                file_id=None,  # Worker will generate the ID now
                file_hash=file_hash
            )

            # 5. Add success result (Return Task ID)
            successful_uploads.append({
                "filename": uploaded_file.filename,
                "task_id": task_id,
                "status": "processing",
                "message": "File queued for background processing."
            })

        except Exception as e:
            # Cleanup
            if file_path and os.path.exists(file_path):
                try: 
                    os.remove(file_path)
                except: 
                    pass 

            error_msg = e.detail if isinstance(e, HTTPException) else str(e)
            failed_uploads.append(
                FileUploadErrorData(
                    filename=uploaded_file.filename,
                    error=error_msg
                )
            )

    # 6. Create response
    # We return a dict because the strict BulkUploadResponseData might require 'file_id'
    response_data = {
        "successful_uploads": successful_uploads,
        "failed_uploads": failed_uploads
    }
    
    status_code = 200 
    message = ""

    if successful_uploads and not failed_uploads:
        message = "All files uploaded and queued for processing."
        status_code = 202 # Accepted/Processing
    elif successful_uploads and failed_uploads:
        message = "Partial success. Some files queued, others failed."
        status_code = 207 
    elif not successful_uploads and failed_uploads:
        message = "All files failed to upload."
        status_code = 422 
    
    return StandardResponse(
        status="success" if successful_uploads else "error",
        status_code=status_code,
        message=message,
        data=response_data
    )

# --- MODIFIED: Returns Category and Type for UI filtering ---
@router.get("/projects/{project_id}/files", response_model=StandardResponse[List[FileUploadResponseData]])
async def get_all_files(
    project_id: str,
    # Allow both admins and users to access this endpoint
    current_user: UserPayload = Depends(require_roles("admin", "user"))
):
    """
    Retrieves metadata.
    - Admin: Gets all files.
    - User: Gets ONLY files uploaded by themselves.
    """
    # The filtering logic is handled inside this middleware function based on user role
    file_models_list = await files_middleware.get_all_files_for_project(project_id, current_user)
    
    response_data_list = [
        FileUploadResponseData(
            file_id=file_doc.file_id,
            project_id=file_doc.project_id,
            filename=file_doc.filename,
            file_url=file_doc.file_url,
            sector=file_doc.sector or "N/A",
            category=file_doc.category, 
            compliance_type=file_doc.compliance_type
        ) for file_doc in file_models_list
    ]
    
    return StandardResponse(
        status="success",
        status_code=200,
        message="Files retrieved successfully.",
        data=response_data_list
    )

# --- MODIFIED: Single file retrieval ---
@router.get("/projects/{project_id}/files/{file_id}", response_model=StandardResponse[FileUploadResponseData])
async def get_file_metadata(
    project_id: str, 
    file_id: str, 
    current_user: UserPayload = Depends(get_current_user)
):
    # Note: You might want to add ownership checks here too if strict privacy is needed,
    # currently implies standard project access verification.
    file_doc = await files_middleware.get_file(project_id, file_id, current_user)

    response_data = FileUploadResponseData(
        file_id=file_doc.file_id,
        project_id=file_doc.project_id,
        filename=file_doc.filename,
        file_url=file_doc.file_url,
        sector=file_doc.sector or "N/A",
        category=file_doc.category,
        compliance_type=file_doc.compliance_type
    )
    
    return StandardResponse(
        status="success",
        status_code=200,
        message="File metadata retrieved successfully.",
        data=response_data
    )

# --- DELETE file (unchanged) ---
@router.delete("/projects/{project_id}/files/{file_id}", response_model=StandardResponse[dict])
async def delete_file(
    project_id: str,
    file_id: str,
    # MODIFIED: Allow 'user' role to access, validation happens inside
    current_user: UserPayload = Depends(require_roles("admin", "user"))
):
    # Get file metadata first (don't check physical existence yet, purely for ownership check)
    file_doc = await files_middleware.get_file(project_id, file_id, current_user, check_physical_file=False)
    
    # --- OWNERSHIP CHECK ---
    # 1. Admin can delete anything.
    # 2. User can ONLY delete their own file.
    if current_user.role != "admin":
        # Ensure file_doc has user_id, and it matches current user
        if not file_doc.user_id or file_doc.user_id != current_user.user_id:
            raise HTTPException(
                status_code=403, 
                detail="Permission denied. You can only delete files you uploaded."
            )

    try:
        await files_middleware.handle_file_deletion(
            file_doc=file_doc,
            project_id=project_id,
            user=current_user
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

    return StandardResponse(
        status="success",
        status_code=200,
        message="File metadata deleted successfully.",
        data={"file_id": file_id, "project_id": project_id, "deleted": True}
    )

@router.post("/deselected")
async def deselect_file(
    deselection: UserFileSelectionModel,
    current_user: UserPayload = Depends(get_current_user)
):
    """
    Mark a file as deselected (hidden) for this user.
    """
    obj = deselection.dict()
    # Ensure these are written/overwritten from backend context
    obj["user_id"] = current_user.user_id
    obj["created_at"] = datetime.utcnow()
    obj["deselected"] = True

    await db.user_file_selection.update_one(
        {"user_id": obj["user_id"], "file_id": obj["file_id"], "source": obj["source"]},
        {"$set": obj},
        upsert=True
    )
    return {"status": "success", "deselected": {"file_id": obj["file_id"], "filename": obj["filename"]}}


# --- NEW: RESTORE (UNHIDE) FILE ENDPOINT ---
@router.post("/reselect")
async def reselect_file(
    selection: UserFileSelectionModel,
    current_user: UserPayload = Depends(get_current_user)
):
    """
    Restores a file by removing it from the user's deselected list.
    """
    # We simply delete the entry that marked it as 'deselected'
    result = await db.user_file_selection.delete_one({
        "user_id": current_user.user_id,
        "file_id": selection.file_id,
        "project_id": selection.project_id
    })
    
    if result.deleted_count == 0:
        # It wasn't hidden, or didn't exist, but that's fine—it's visible now.
        return {"status": "info", "message": "File was not hidden."}

    return {"status": "success", "message": "File restored successfully."}


# ---  NEW: GET DESELECTED FILES ---
@router.get("/deselected/{project_id}")
async def get_deselected_files(
    project_id: str,
    current_user: UserPayload = Depends(get_current_user)
):
    """
    Get list of files the user has deselected for this project.
    Used by frontend to filter the main list.
    """
    try:
        # Query for files hidden by THIS user for THIS project
        # We specifically look for 'deselected: True'
        cursor = db.user_file_selection.find({
            "user_id": current_user.user_id,
            "project_id": project_id,
            "deselected": True
        })
        
        # Convert cursor to list
        deselected_docs = await cursor.to_list(length=1000)

        # Format the response (convert ObjectId to string if needed, or just return key fields)
        results = []
        for doc in deselected_docs:
            results.append({
                "file_id": doc.get("file_id"),
                "filename": doc.get("filename"),
                "source": doc.get("source"),
                "deselected_at": doc.get("created_at")
            })

        return StandardResponse(
            status="success",
            status_code=200,
            message="Deselected files retrieved successfully.",
            data=results
        )
    except Exception as e:
        # Log error if needed
        logger.error(f"Error fetching deselected files: {str(e)}")
        return StandardResponse(
            status="error",
            status_code=500,
            message="Failed to fetch deselected files.",
            data=[]
        )

from app.database import Global_file_collection

class FilePathRequest(BaseModel):
    filename: str
    project_id: str



# @router.get("/files/{file_id}/chunks", response_model=StandardResponse[Dict[str, Any]])
# async def get_file_chunks(
#     file_id: str,
#     limit: int = 50, 
#     current_user: UserPayload = Depends(get_current_user)
# ):
#     """
#     Debug Endpoint: Returns the actual text chunks stored in the Vector DB for a specific file.
#     """
#     # --- FIX START ---
#     try:
#         # Convert string ID to MongoDB ObjectId
#         obj_id = ObjectId(file_id)
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid file ID format")

#     # Query using "_id" instead of "file_id"
#     file_doc = await db.files.find_one({"_id": obj_id})
#     # --- FIX END ---
    
#     if not file_doc:
#          raise HTTPException(status_code=404, detail="File not found")
    
#     project_id = file_doc.get("project_id")
#     filename = file_doc.get("filename")

#     # 2. Verify Access (Ensure user has access to this project)
#     await files_middleware.verify_project(project_id, current_user)

#     # 3. Fetch chunks from Qdrant
#     vector_retriever = get_vector_retriever()
#     chunks = await vector_retriever.get_chunks_by_filename(
#         project_id=project_id, 
#         filename=filename,
#         limit=limit
#     )
    
#     return StandardResponse(
#         status="success",
#         status_code=200,
#         message=f"Retrieved {len(chunks)} chunks for inspection.",
#         data={
#             "file_id": file_id,
#             "filename": filename,
#             "project_id": project_id,
#             "total_chunks_retrieved": len(chunks),
#             "chunks": chunks
#         }
#     )