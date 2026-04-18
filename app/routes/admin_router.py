# app/routes/admin_router.py

"""Admin Router (HR single-sector mode)."""

from fastapi import APIRouter, HTTPException, Depends
from app.dependencies import UserPayload, get_current_user
from app.schemas import StandardResponse
from app.database import Global_file_collection

router = APIRouter()


def verify_admin(current_user: UserPayload):
    if current_user.role.lower() not in ["admin", "super_admin"]:
        raise HTTPException(status_code=403, detail="Access denied. Administrator privileges required.")


@router.post("/upload-global", tags=["Admin"])
async def upload_global_sector_file(current_user: UserPayload = Depends(get_current_user)):
    verify_admin(current_user)
    raise HTTPException(
        status_code=410,
        detail="Global/regulatory sector upload is deprecated. Use project organization uploads."
    )


@router.get("/files", tags=["Admin"])
async def get_all_global_files(current_user: UserPayload = Depends(get_current_user)):
    # Project detail uses this endpoint for organization-level documents that
    # should be visible to any authenticated project participant, including
    # end users. Restricting this to admins caused the frontend to swallow a
    # 403 and render an empty list for users.

    docs = await Global_file_collection.find({}).to_list(None)
    files_list = []
    for doc in docs:
        doc["_id"] = str(doc["_id"])
        if "created_at" in doc and hasattr(doc["created_at"], "isoformat"):
            doc["created_at"] = doc["created_at"].isoformat()
        if "updated_at" in doc and hasattr(doc["updated_at"], "isoformat"):
            doc["updated_at"] = doc["updated_at"].isoformat()
        files_list.append(doc)

    return StandardResponse(
        status="success",
        status_code=200,
        message=f"Retrieved {len(files_list)} global files",
        data=files_list
    )


@router.get("/global-sectors", tags=["Admin"])
async def list_global_sectors(current_user: UserPayload = Depends(get_current_user)):
    verify_admin(current_user)
    raise HTTPException(
        status_code=410,
        detail="Global sectors endpoint is deprecated in HR single-sector mode."
    )

