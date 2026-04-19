from fastapi import APIRouter, HTTPException, Depends
from bson import ObjectId
from typing import List
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.models.project import ProjectModel, ProjectUpdateSchema
from app.models.project_config import ProjectConfig, ProjectConfigPatch
from app.database import project_collection, User_collection, project_config_collection
from app.middleware.project_middleware import validate_description, validate_industry, validate_project_name
from app.dependencies import get_current_user, UserPayload
from app.schemas import StandardResponse
from app.middleware.role_checker import require_roles
from app.auth.jwt_bearer import JWTBearer
from app.utils.logger import setup_logger
from app.config import normalize_complex_model

logger = setup_logger()
router = APIRouter()
IST = ZoneInfo("Asia/Kolkata")


def _normalize_org_sector(value: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="Organization name is required.")
    # Keep readable org name; unify spacing and uppercase for indexing filters.
    return " ".join(cleaned.split()).upper()


@router.post("/new_project", response_model=StandardResponse[dict])
async def create_project(
    project: ProjectModel,
    current_user: UserPayload = Depends(require_roles("admin", "super_admin")),
):
    user_id = current_user.user_id
    project.user_id = user_id

    validate_project_name(project.project_name)
    project.industry = validate_industry(project.industry)
    validate_description(project.description)

    org_sector = _normalize_org_sector(project.organization_sector)
    project.has_organizational_compliance = True
    project.organization_sector = org_sector
    project.sectors = [org_sector]

    project_data = project.model_dump()

    try:
        result = await project_collection.insert_one(project_data)
        return StandardResponse(
            status="success",
            status_code=200,
            message="Project created successfully",
            data={
                "inserted_id": str(result.inserted_id),
                "user_id": user_id,
                "organization_sector": org_sector,
                "sectors": [org_sector]
            }
        )
    except Exception as e:
        if "E11000" in str(e):
            raise HTTPException(status_code=400, detail="A project with this name already exists for your account.")
        raise HTTPException(status_code=500, detail=f"Failed to create project: {e}")


@router.post("/{project_id}/settings", dependencies=[Depends(require_roles("admin", "super_admin"))])
async def save_project_settings(project_id: str, config_data: ProjectConfig):
    if not ObjectId.is_valid(project_id):
        raise HTTPException(status_code=400, detail="Invalid project ID")

    data = config_data.dict()
    data["project_id"] = project_id
    data["updated_at"] = datetime.now(IST)
    if "created_at" in data:
        del data["created_at"]

    try:
        result = await project_config_collection.update_one(
            {"project_id": project_id},
            {"$set": data, "$setOnInsert": {"created_at": datetime.now(IST)}},
            upsert=True
        )
        return StandardResponse(
            status="success",
            status_code=200,
            message="AI Configuration saved successfully",
            data={"project_id": project_id, "is_new_config": (result.upserted_id is not None)}
        )
    except Exception as e:
        logger.error(f"[Project Router] Error saving settings: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@router.patch("/{project_id}/settings", response_model=StandardResponse[dict])
async def update_project_settings(
    project_id: str,
    config_update: ProjectConfigPatch,
    current_user: UserPayload = Depends(require_roles("admin", "super_admin")),
):
    if not ObjectId.is_valid(project_id):
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project_exists = await project_collection.find_one(
        {"_id": ObjectId(project_id), "user_id": current_user.user_id},
        {"_id": 1}
    )
    if not project_exists:
        raise HTTPException(status_code=403, detail="Access denied. You can only configure projects you created.")

    update_data = config_update.model_dump(exclude_unset=True)
    if not update_data:
        return StandardResponse(status="success", status_code=200, message="No changes detected", data={"project_id": project_id})

    update_data["updated_at"] = datetime.now(IST)
    update_data["project_id"] = project_id

    result = await project_config_collection.update_one(
        {"project_id": project_id},
        {"$set": update_data, "$setOnInsert": {"created_at": datetime.now(IST)}},
        upsert=True
    )

    return StandardResponse(
        status="success",
        status_code=200,
        message="AI Configuration updated successfully",
        data={
            "project_id": project_id,
            "updated_fields": list(update_data.keys()),
            "is_new_config": (result.upserted_id is not None)
        }
    )


@router.patch("/project/{project_id}", response_model=StandardResponse[dict])
async def update_project_details(
    project_id: str,
    update_data: ProjectUpdateSchema,
    current_user: UserPayload = Depends(require_roles("admin", "super_admin")),
):
    if not ObjectId.is_valid(project_id):
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project_obj_id = ObjectId(project_id)
    existing_project = await project_collection.find_one({"_id": project_obj_id})
    if not existing_project:
        raise HTTPException(status_code=404, detail="Project not found")
    if existing_project.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied. You can only update projects you created.")

    fields_to_update = update_data.model_dump(exclude_unset=True)
    if not fields_to_update:
        return StandardResponse(status="success", status_code=200, message="No changes detected", data={"project_id": project_id})

    if "project_name" in fields_to_update:
        new_name = fields_to_update["project_name"]
        if new_name != existing_project.get("project_name"):
            validate_project_name(new_name)
        else:
            del fields_to_update["project_name"]

    if "description" in fields_to_update:
        validate_description(fields_to_update["description"])

    if "industry" in fields_to_update:
        fields_to_update["industry"] = validate_industry(fields_to_update["industry"])

    if not fields_to_update:
        return StandardResponse(status="success", status_code=200, message="No effective changes detected", data={"project_id": project_id})

    fields_to_update["updated_at"] = datetime.now(timezone.utc)

    try:
        await project_collection.update_one({"_id": project_obj_id}, {"$set": fields_to_update})
        return StandardResponse(
            status="success",
            status_code=200,
            message="Project details updated successfully",
            data={"project_id": project_id, "updated_fields": list(fields_to_update.keys())}
        )
    except Exception as e:
        if "E11000" in str(e):
            raise HTTPException(status_code=400, detail="A project with this name already exists.")
        raise HTTPException(status_code=500, detail=f"Failed to update project: {str(e)}")


@router.get("/{project_id}/settings", dependencies=[Depends(JWTBearer())])
async def get_project_settings(project_id: str):
    if not ObjectId.is_valid(project_id):
        raise HTTPException(status_code=400, detail="Invalid project ID")

    config = await project_config_collection.find_one({"project_id": project_id})
    if config:
        config.pop("_id", None)
        config["complex_model"] = normalize_complex_model(config.get("complex_model"))
        return StandardResponse(status="success", status_code=200, message="Config found", data=config)

    defaults = {
        "project_id": project_id,
        "router_model": "qwen2.5:0.5b",
        "simple_model": "qwen2.5:1.5b-instruct",
        "complex_model": "gemma:7b",
        "search_strategy": "hybrid",
        "retrieval_depth": 5,
        "enable_reranking": True,
        "ocr_engine": "paddleocr",
        "is_default": True
    }
    return StandardResponse(status="success", status_code=200, message="Default config returned", data=defaults)


@router.get("/my_projects", response_model=StandardResponse[list])
async def get_my_projects(current_user: UserPayload = Depends(get_current_user)):
    user_id = current_user.user_id
    user_role = current_user.role

    if user_role == "admin":
        query_filter = {"user_id": user_id}
    else:
        user_doc = await User_collection.find_one({"_id": ObjectId(user_id)})
        admin_id = user_doc.get("admin_id") if user_doc else None
        if not admin_id:
            return StandardResponse(status="success", status_code=200, message="No organization found for this user", data=[])
        query_filter = {"user_id": admin_id}

    projects = []
    async for project in project_collection.find(query_filter):
        project["_id"] = str(project["_id"])
        projects.append(project)

    return StandardResponse(status="success", status_code=200, message="Projects retrieved successfully", data=projects)


@router.get("/project/{project_id}", response_model=StandardResponse[dict])
async def get_project(project_id: str, current_user: UserPayload = Depends(get_current_user)):
    try:
        project_obj_id = ObjectId(project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await project_collection.find_one({"_id": project_obj_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_owner_id = project.get("user_id")
    if current_user.role == "admin":
        if project_owner_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied.")
    else:
        user_doc = await User_collection.find_one({"_id": ObjectId(current_user.user_id)})
        admin_id = user_doc.get("admin_id") if user_doc else None
        if not admin_id or project_owner_id != admin_id:
            raise HTTPException(status_code=403, detail="Access denied. This project does not belong to your organization.")

    project["_id"] = str(project["_id"])
    project["organizational_compliance"] = "Yes"

    await project_collection.update_one({"_id": project_obj_id}, {"$set": {"updated_at": datetime.now(timezone.utc)}})

    return StandardResponse(status="success", status_code=200, message="Project retrieved successfully", data=project)


@router.get("/project/{project_id}/default-sectors", response_model=StandardResponse[dict])
async def get_project_default_sectors(project_id: str, current_user: UserPayload = Depends(get_current_user)):
    try:
        project_obj_id = ObjectId(project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await project_collection.find_one(
        {"_id": project_obj_id},
        {"project_name": 1, "sectors": 1, "industry": 1, "user_id": 1, "organization_sector": 1}
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    owner_id = project.get("user_id")
    if current_user.role == "admin":
        if owner_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
    else:
        user_doc = await User_collection.find_one({"_id": ObjectId(current_user.user_id)})
        admin_id = user_doc.get("admin_id") if user_doc else None
        if owner_id != admin_id:
            raise HTTPException(status_code=403, detail="Not authorized")

    org_sector = project.get("organization_sector") or (project.get("sectors") or [None])[0]

    return StandardResponse(
        status="success",
        status_code=200,
        message="Project default sectors retrieved successfully",
        data={
            "project_id": project_id,
            "project_name": project.get("project_name"),
            "default_sectors": [org_sector] if org_sector else [],
            "industry": project.get("industry")
        }
    )


@router.patch("/project/{project_id}/sectors", response_model=StandardResponse[dict])
async def update_project_sectors(
    project_id: str,
    sectors: List[str],
    current_user: UserPayload = Depends(require_roles("admin", "super_admin")),
):
    try:
        project_obj_id = ObjectId(project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await project_collection.find_one({"_id": project_obj_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    if not sectors or len(sectors) != 1:
        raise HTTPException(status_code=422, detail="Exactly one organization sector is required.")

    org_sector = _normalize_org_sector(sectors[0])

    await project_collection.update_one(
        {"_id": project_obj_id},
        {"$set": {
            "sectors": [org_sector],
            "organization_sector": org_sector,
            "has_organizational_compliance": True,
            "updated_at": datetime.now(timezone.utc)
        }}
    )

    return StandardResponse(
        status="success",
        status_code=200,
        message="Project organization sector updated successfully",
        data={"project_id": project_id, "updated_sectors": [org_sector], "organization_sector": org_sector}
    )


@router.delete("/project/{project_id}", response_model=StandardResponse[dict])
async def delete_project(
    project_id: str,
    current_user: UserPayload = Depends(require_roles("admin", "super_admin")),
):
    try:
        project_obj_id = ObjectId(project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await project_collection.find_one({"_id": project_obj_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied. You can only delete projects you created.")

    result = await project_collection.delete_one({"_id": project_obj_id})
    if result.deleted_count == 1:
        return StandardResponse(status="success", status_code=200, message="Project deleted successfully", data={"project_id": project_id})

    raise HTTPException(status_code=500, detail="Failed to delete project")
