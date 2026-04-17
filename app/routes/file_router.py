# app/routes/file_router.py

from fastapi import APIRouter, Depends, UploadFile, File as FastAPIFile, HTTPException, Form, Request
from typing import List, Dict, Any
from app.services.document_processing.ocr_engine import get_ocr_engine
from datetime import datetime
import logging
import os
from bson import ObjectId

from app.middleware.role_checker import require_roles
from app.middleware import files_middleware
from app.models.UserFileSelection import UserFileSelectionModel
from app.dependencies import get_current_user, UserPayload
from app.services.redis.redis_service import redis_service
from app.schemas import StandardResponse, FileUploadResponseData, FileUploadErrorData
from app.database import db
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class CancelRequest(BaseModel):
    task_id: str


@router.post("/cancel_upload")
async def cancel_upload_signal(req: CancelRequest):
    if not req.task_id:
        raise HTTPException(status_code=400, detail="Task ID is required")
    await redis_service.cancel_job(req.task_id)
    return {"status": "cancelled", "task_id": req.task_id}


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    job_info = await redis_service.get_job_status(task_id)
    if not job_info:
        return StandardResponse(status="error", status_code=404, message="Task ID not found or expired.", data=None)

    return StandardResponse(
        status="success",
        status_code=200,
        message="Task status retrieved",
        data={"task_id": task_id, "status": job_info.get("status"), "result": job_info.get("data")}
    )


@router.post("/projects/{project_id}/files", response_model=StandardResponse[Dict[str, Any]])
async def upload_files_bulk(
    project_id: str,
    request: Request,
    files: List[UploadFile] = FastAPIFile(...),
    categories: List[str] = Form(...),
    compliance_types: List[str] = Form(...),
    current_user: UserPayload = Depends(require_roles("admin", "super_admin"))
):
    """Bulk upload for HR/CEO roles only. Sector is always project.organization_sector."""
    if not files:
        raise HTTPException(status_code=422, detail="No files were uploaded.")
    if len(files) != len(categories) or len(files) != len(compliance_types):
        raise HTTPException(status_code=422, detail="Count mismatch between files, categories, or compliance types.")

    project = await files_middleware.verify_project(project_id, current_user)
    current_sector = (project.get("organization_sector") or "").strip().upper()
    if not current_sector:
        raise HTTPException(status_code=500, detail="Organization sector missing in project configuration")

    selected_engine = "paddleocr"
    try:
        project_config = await db.project_configs.find_one({"project_id": project_id})
        if project_config:
            config_engine = project_config.get("ocr_engine")
            if config_engine in ["paddleocr", "easyocr"]:
                selected_engine = config_engine
    except Exception:
        selected_engine = "paddleocr"

    if selected_engine == "paddleocr":
        ocr_instance = get_ocr_engine()
        if not ocr_instance.is_service_available():
            raise HTTPException(status_code=503, detail="PaddleOCR is not available. Change OCR setting or retry later.")

    successful_uploads: List[Dict[str, Any]] = []
    failed_uploads: List[FileUploadErrorData] = []

    for i, uploaded_file in enumerate(files):
        if await request.is_disconnected():
            return StandardResponse(
                status="error",
                status_code=499,
                message="Client cancelled upload.",
                data={"successful_uploads": [], "failed_uploads": []}
            )

        file_path = None
        try:
            current_category = categories[i].strip()
            current_compliance_type = compliance_types[i].strip()

            _, file_path, file_hash = await files_middleware.save_file(uploaded_file, project_id)

            task_id = await files_middleware.run_qdrant_indexing(
                file_path=file_path,
                project_id=project_id,
                sector=current_sector,
                current_user=current_user,
                category=current_category,
                compliance_type=current_compliance_type,
                original_filename=uploaded_file.filename,
                ocr_engine=selected_engine,
                file_id=None,
                file_hash=file_hash
            )

            successful_uploads.append({
                "filename": uploaded_file.filename,
                "task_id": task_id,
                "status": "processing",
                "message": "File queued for background processing."
            })

        except Exception as e:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass

            error_msg = e.detail if isinstance(e, HTTPException) else str(e)
            failed_uploads.append(FileUploadErrorData(filename=uploaded_file.filename, error=error_msg))

    response_data = {"successful_uploads": successful_uploads, "failed_uploads": failed_uploads}

    if successful_uploads and not failed_uploads:
        return StandardResponse(status="success", status_code=202, message="All files uploaded and queued for processing.", data=response_data)
    if successful_uploads and failed_uploads:
        return StandardResponse(status="success", status_code=207, message="Partial success. Some files queued, others failed.", data=response_data)
    return StandardResponse(status="error", status_code=422, message="All files failed to upload.", data=response_data)


@router.get("/projects/{project_id}/files", response_model=StandardResponse[List[FileUploadResponseData]])
async def get_all_files(
    project_id: str,
    current_user: UserPayload = Depends(require_roles("admin", "super_admin", "user"))
):
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

    return StandardResponse(status="success", status_code=200, message="Files retrieved successfully.", data=response_data_list)


@router.get("/projects/{project_id}/files/{file_id}", response_model=StandardResponse[FileUploadResponseData])
async def get_file_metadata(
    project_id: str,
    file_id: str,
    current_user: UserPayload = Depends(get_current_user)
):
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

    return StandardResponse(status="success", status_code=200, message="File metadata retrieved successfully.", data=response_data)


@router.delete("/projects/{project_id}/files/{file_id}", response_model=StandardResponse[dict])
async def delete_file(
    project_id: str,
    file_id: str,
    current_user: UserPayload = Depends(require_roles("admin", "super_admin", "user"))
):
    file_doc = await files_middleware.get_file(project_id, file_id, current_user, check_physical_file=False)

    if current_user.role not in ["admin", "super_admin"]:
        if not file_doc.user_id or file_doc.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Permission denied. You can only delete files you uploaded.")

    await files_middleware.handle_file_deletion(file_doc=file_doc, project_id=project_id, user=current_user)

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
    obj = deselection.dict()
    obj["user_id"] = current_user.user_id
    obj["created_at"] = datetime.utcnow()
    obj["deselected"] = True

    await db.user_file_selection.update_one(
        {"user_id": obj["user_id"], "file_id": obj["file_id"], "source": obj["source"]},
        {"$set": obj},
        upsert=True
    )
    return {"status": "success", "deselected": {"file_id": obj["file_id"], "filename": obj["filename"]}}


@router.post("/reselect")
async def reselect_file(
    selection: UserFileSelectionModel,
    current_user: UserPayload = Depends(get_current_user)
):
    result = await db.user_file_selection.delete_one({
        "user_id": current_user.user_id,
        "file_id": selection.file_id,
        "project_id": selection.project_id
    })

    if result.deleted_count == 0:
        return {"status": "info", "message": "File was not hidden."}

    return {"status": "success", "message": "File restored successfully."}


@router.get("/deselected/{project_id}")
async def get_deselected_files(
    project_id: str,
    current_user: UserPayload = Depends(get_current_user)
):
    try:
        cursor = db.user_file_selection.find({
            "user_id": current_user.user_id,
            "project_id": project_id,
            "deselected": True
        })

        deselected_docs = await cursor.to_list(length=1000)
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
        logger.error(f"Error fetching deselected files: {str(e)}")
        return StandardResponse(
            status="error",
            status_code=500,
            message="Failed to fetch deselected files.",
            data=[]
        )
