from fastapi import APIRouter, HTTPException, Depends, Body
from bson import ObjectId
from typing import List, Optional
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# --- Local Imports ---
from app.models.project import ProjectModel,ProjectUpdateSchema
from app.models.project_config import ProjectConfig,ProjectConfigPatch # <--- IMPORTED NEW MODEL
from app.database import project_collection, User_collection, project_config_collection,get_database # <--- IMPORT get_database
from app.middleware.project_middleware import validate_description, validate_industry, validate_project_name, validate_sectors
from app.dependencies import get_current_user, UserPayload
from app.schemas import StandardResponse, ErrorDetail
from app.middleware.role_checker import require_roles
from app.auth.jwt_bearer import JWTBearer
from app.utils.logger import setup_logger # Ensure you have this


logger = setup_logger()

router = APIRouter()

# Helper for IST Time (Matches your model preference)
IST = ZoneInfo("Asia/Kolkata")

from fastapi import APIRouter, HTTPException, Depends
from bson import ObjectId
from typing import List, Optional
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# --- Local Imports ---
from app.models.project import ProjectModel
from app.database import project_collection
from app.middleware.project_middleware import validate_description, validate_industry, validate_project_name, validate_sectors
from app.dependencies import get_current_user, UserPayload
from app.schemas import StandardResponse
from app.middleware.role_checker import require_roles

router = APIRouter()

# Helper for IST Time
IST = ZoneInfo("Asia/Kolkata")

@router.post("/new_project", response_model=StandardResponse[dict])
async def create_project(
    project: ProjectModel,
    current_user: UserPayload = Depends(require_roles("admin")),
):
    user_id = current_user.user_id
    project.user_id = user_id

    try:
        # 1. Basic Validation
        validate_project_name(project.project_name)
        cleaned_industry = validate_industry(project.industry)
        validate_description(project.description)
        
        # ---------------- MERGING LOGIC ----------------
        # Start with the explicitly selected sectors
        raw_sectors = []

        # [NEW] Pre-calculation to check if everything is empty
        is_reg_empty = not project.regulatory_framework or project.regulatory_framework == "None"
        
        # Check if list is empty OR contains only "None" strings
        is_tp_empty = not project.third_party_framework or all(fw == "None" for fw in project.third_party_framework)
        
        is_org_empty = not project.has_organizational_compliance

        # MASTER VALIDATION: If everything is empty, stop here.
        if is_reg_empty and is_tp_empty and is_org_empty:
            raise HTTPException(
                status_code=400, 
                detail="Invalid Project: You must select at least one compliance framework."
            )

        # 1. Fix Regulatory Framework: Only add if it is NOT "None"
        if project.regulatory_framework and project.regulatory_framework != "None":
            raw_sectors.append(project.regulatory_framework)
        else:
            # Ensure it is stored as null in DB if "None" was sent
            project.regulatory_framework = None

        # 2. Fix Third Party Frameworks: Filter out "None"
        if project.third_party_framework:
            # Remove any occurrence of "None" from the list
            cleaned_third_party = [fw for fw in project.third_party_framework if fw != "None"]
            
            # Update the project object with the cleaned list
            project.third_party_framework = cleaned_third_party
            
            # Add valid frameworks to the search sectors
            raw_sectors.extend(cleaned_third_party)

        # --- ORGANIZATIONAL COMPLIANCE ---
        formatted_org_sector = None # Initialize variable

        if project.has_organizational_compliance:
            # 1. Validate that a name was actually provided
            if not project.organization_sector:
                raise HTTPException(
                    status_code=400,
                    detail="Organization sector is required when organizational compliance is enabled"
                )

            # 2. Create the unique formatted ID (THIS WAS MISSING)
            # Example: "MyCompany" -> "MYCOMPANY_696f5..."
            org_sector_clean = project.organization_sector.strip().upper()
            random_id = str(ObjectId()) 
            formatted_org_sector = f"{org_sector_clean}_{random_id}"

            # 3. Save this UNIQUE version to the database field 'organization_sector'
            project.organization_sector = formatted_org_sector

        else:
            project.organization_sector = None

        # Validate and Normalize the combined list
        cleaned_sectors = validate_sectors(raw_sectors)
        
        # Force all sectors to UPPERCASE
        cleaned_sectors = [s.upper() for s in cleaned_sectors]

        # ADD ORGANIZATION SECTOR (WITH RANDOM ID) TO THE LIST
        if formatted_org_sector:
            cleaned_sectors.append(formatted_org_sector)

    except HTTPException as e:
        raise e

    # --- DATA UPDATE ---
    project.industry = cleaned_industry
    
    # Update 'sectors' to include ONLY valid real sectors
    project.sectors = cleaned_sectors
    
    # NOTE: We are NOT clearing regulatory_framework or third_party_framework.
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
                "sectors": cleaned_sectors, # Will return ["SEBI", "TEST_6952..."]
                "regulatory_framework": project.regulatory_framework,
                "third_party_framework": project.third_party_framework,
                "organization_sector": project.organization_sector # Will return "TEST_6952..."
            }
        )
    except Exception as e:
        if "E11000" in str(e):
            raise HTTPException(status_code=400, detail="A project with this name already exists for your account.")
        raise HTTPException(status_code=500, detail=f"Failed to create project: {e}")

@router.post("/{project_id}/settings", dependencies=[Depends(require_roles("admin"))]) 
async def update_project_settings(
    project_id: str, 
    config_data: ProjectConfig
):
    try:
        if not ObjectId.is_valid(project_id):
             raise HTTPException(status_code=400, detail="Invalid project ID")

        # 1. Convert Pydantic model to dictionary
        data = config_data.dict()
        data["project_id"] = project_id
        
        # 2. Update timestamp
        data["updated_at"] = datetime.now(IST)
        if "created_at" in data:
             del data["created_at"] 
        
        # 3. Use the imported collection directly
        result = await project_config_collection.update_one(
            {"project_id": project_id},
            {"$set": data, "$setOnInsert": {"created_at": datetime.now(IST)}},
            upsert=True
        )
        
        return StandardResponse(
            status="success",
            status_code=200,
            message="AI Configuration saved successfully",
            data={
                "project_id": project_id,
                "is_new_config": (result.upserted_id is not None)
            }
        )

    except Exception as e:
        logger.error(f"[Project Router] Error saving settings: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
@router.patch("/{project_id}/settings", response_model=StandardResponse[dict])
async def update_project_settings(
    project_id: str,
    config_update: ProjectConfigPatch,  # <--- Uses your new Patch Model
    current_user: UserPayload = Depends(require_roles("admin")),
):
    """
    Partially updates AI configuration settings.
    Only the fields sent by the user will be updated.
    """
    
    # 1. Validate Project ID
    if not ObjectId.is_valid(project_id):
        raise HTTPException(status_code=400, detail="Invalid project ID format")
    
    # 2. SECURITY: Check if the user OWNS the project
    # We query the MAIN project collection to verify ownership.
    project_exists = await project_collection.find_one(
        {"_id": ObjectId(project_id), "user_id": current_user.user_id},
        {"_id": 1} 
    )

    if not project_exists:
        raise HTTPException(
            status_code=403, 
            detail="Access denied. You can only configure projects you created."
        )

    # 3. FILTER DATA: Remove unset fields
    # "exclude_unset=True" is critical. It strips out any fields the user didn't send.
    # If the user sends {"retrieval_depth": 8}, this dict will ONLY contain that one key.
    update_data = config_update.model_dump(exclude_unset=True)

    if not update_data:
        return StandardResponse(
            status="success",
            status_code=200,
            message="No changes detected",
            data={"project_id": project_id}
        )

    # 4. Prepare Metadata
    update_data["updated_at"] = datetime.now(IST)
    update_data["project_id"] = project_id 

    # 5. DB UPDATE (Upsert)
    try:
        result = await project_config_collection.update_one(
            {"project_id": project_id},
            {
                "$set": update_data,
                # If this config doc doesn't exist yet, created_at is added automatically
                "$setOnInsert": {"created_at": datetime.now(IST)} 
            },
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

    except Exception as e:
        # In production, log this error: logger.error(f"Config Update Failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to update configuration")

@router.patch("/project/{project_id}", response_model=StandardResponse[dict])
async def update_project_details(
    project_id: str,
    update_data: ProjectUpdateSchema, # <--- Uses the safe schema
    current_user: UserPayload = Depends(require_roles("admin")),
):
    """
    Updates basic project details (Name, Description, Industry).
    Strictly forbids updating Compliance Sectors/Frameworks.
    """
    user_id = current_user.user_id

    # 1. Validate ID
    if not ObjectId.is_valid(project_id):
        raise HTTPException(status_code=400, detail="Invalid project ID format")
    
    project_obj_id = ObjectId(project_id)

    # 2. SECURITY: Verify Ownership
    # We must fetch the project to ensure the current user is the owner
    existing_project = await project_collection.find_one({"_id": project_obj_id})

    if not existing_project:
        raise HTTPException(status_code=404, detail="Project not found")

    if existing_project.get("user_id") != user_id:
        raise HTTPException(
            status_code=403, 
            detail="Access denied. You can only update projects you created."
        )

    # 3. FILTER DATA: Remove unset fields
    fields_to_update = update_data.model_dump(exclude_unset=True)

    # If no valid fields were sent, stop here
    if not fields_to_update:
        return StandardResponse(
            status="success",
            status_code=200,
            message="No changes detected",
            data={"project_id": project_id}
        )

    # 4. VALIDATION LOGIC
    # We only run validation on the specific fields that are changing
    try:
        # A. Project Name
        if "project_name" in fields_to_update:
            new_name = fields_to_update["project_name"]
            # Optimization: Don't validate if the name hasn't actually changed
            if new_name != existing_project.get("project_name"):
                validate_project_name(new_name)
            else:
                # Remove it from update list to save DB work
                del fields_to_update["project_name"]

        # B. Description
        if "description" in fields_to_update:
            validate_description(fields_to_update["description"])

        # C. Industry
        if "industry" in fields_to_update:
            # Normalize industry (e.g., capitalize)
            cleaned_industry = validate_industry(fields_to_update["industry"])
            fields_to_update["industry"] = cleaned_industry

    except HTTPException as e:
        raise e  # Re-raise validation errors (like "Name too short")

    # Double check if we still have fields to update after optimizations
    if not fields_to_update:
         return StandardResponse(
            status="success",
            status_code=200,
            message="No effective changes detected",
            data={"project_id": project_id}
        )

    # 5. Add Timestamp
    fields_to_update["updated_at"] = datetime.now(timezone.utc)

    # 6. DB UPDATE
    try:
        await project_collection.update_one(
            {"_id": project_obj_id},
            {"$set": fields_to_update}
        )
        
        return StandardResponse(
            status="success",
            status_code=200,
            message="Project details updated successfully",
            data={
                "project_id": project_id,
                "updated_fields": list(fields_to_update.keys())
            }
        )

    except Exception as e:
        # Handle Duplicate Name Error
        if "E11000" in str(e):
            raise HTTPException(
                status_code=400, 
                detail="A project with this name already exists."
            )
        raise HTTPException(status_code=500, detail=f"Failed to update project: {str(e)}")
# =================================================================
# 3. GET AI CONFIGURATION (Updated to use collection object)
# =================================================================
@router.get("/{project_id}/settings", dependencies=[Depends(JWTBearer())])
async def get_project_settings(project_id: str):
    try:
        if not ObjectId.is_valid(project_id):
             raise HTTPException(status_code=400, detail="Invalid project ID")

        # Use the imported collection directly
        config = await project_config_collection.find_one({"project_id": project_id})
        
        if config:
            if "_id" in config: del config["_id"]
            return StandardResponse(
                status="success", status_code=200, message="Config found", data=config
            )
        else:
            # Return defaults
            defaults = {
                "project_id": project_id,
                "router_model": "qwen2.5:0.5b",
                "simple_model": "qwen2.5:7b-instruct-q4_K_M",
                "complex_model": "qwen2.5:14b",
                "search_strategy": "hybrid",
                "retrieval_depth": 5,
                "enable_reranking": True,
                "ocr_engine": "paddleocr",
                "is_default": True
            }
            return StandardResponse(
                status="success", status_code=200, message="Default config returned", data=defaults
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/my_projects", response_model=StandardResponse[list])
async def get_my_projects(current_user: UserPayload = Depends(get_current_user)):
    user_id = current_user.user_id
    user_role = current_user.role

    projects = []
    query_filter = {}

    if user_role == "admin":
        # 1. Admin Logic: See ONLY projects they created
        query_filter = {"user_id": user_id}
    
    else:
        # 2. User Logic: See projects created by their linked Admin
        
        # A. Fetch the current user's details to find their 'admin_id'
        user_doc = await User_collection.find_one({"_id": ObjectId(user_id)})
        
        # B. Get the admin_id (handle case where it might be missing)
        admin_id = user_doc.get("admin_id") if user_doc else None
        
        if not admin_id:
            # If the user has no admin (orphaned), they see nothing
            return StandardResponse(
                status="success",
                status_code=200,
                message="No organization found for this user",
                data=[]
            )

        # C. Filter projects where the CREATOR (user_id) is this User's Admin
        query_filter = {"user_id": admin_id}
    
    # --- Execute Query ---
    projects_cursor = project_collection.find(query_filter)
    
    async for project in projects_cursor:
        project["_id"] = str(project["_id"])
        projects.append(project)
    
    return StandardResponse(
        status="success",
        status_code=200,
        message="Projects retrieved successfully",
        data=projects
    )

@router.get("/project/{project_id}", response_model=StandardResponse[dict])
async def get_project(
    project_id: str,
    current_user: UserPayload = Depends(get_current_user),
):
    try:
        project_obj_id = ObjectId(project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await project_collection.find_one({"_id": project_obj_id})
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # --- VISIBILITY LOGIC ---
    project_owner_id = project.get("user_id")

    if current_user.role == "admin":
        # Admin can only see their OWN projects
        if project_owner_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied.")
            
    else:
        # User can only see projects owned by their ADMIN
        user_doc = await User_collection.find_one({"_id": ObjectId(current_user.user_id)})
        admin_id = user_doc.get("admin_id") if user_doc else None
        
        if not admin_id or project_owner_id != admin_id:
            raise HTTPException(status_code=403, detail="Access denied. This project does not belong to your organization.")

    project["_id"] = str(project["_id"])

    # --- ADD THIS BLOCK ---
    project["organizational_compliance"] = (
        "Yes" if project.get("has_organizational_compliance") else "No"
    )
    # ---------------------

    
    # Only update timestamp if allowed
    await project_collection.update_one(
        {"_id": project_obj_id},
        {"$set": {"updated_at": datetime.now(timezone.utc)}}
    )

    return StandardResponse(
        status="success",
        status_code=200,
        message="Project retrieved successfully",
        data=project
    )


@router.get("/project/{project_id}/default-sectors", response_model=StandardResponse[dict])
async def get_project_default_sectors(
    project_id: str,
    current_user: UserPayload = Depends(get_current_user),
):
    user_id = current_user.user_id
    try:
        project_obj_id = ObjectId(project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await project_collection.find_one(
        {"_id": project_obj_id},
        {"project_name": 1, "sectors": 1, "industry": 1, "user_id": 1}
    )
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # --- SECURITY UPDATE FOR SECTORS ---
    project_owner_id = project.get("user_id")
    
    if current_user.role == "admin":
        if project_owner_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
    else:
        # Check against admin_id
        user_doc = await User_collection.find_one({"_id": ObjectId(user_id)})
        admin_id = user_doc.get("admin_id") if user_doc else None
        if project_owner_id != admin_id:
            raise HTTPException(status_code=403, detail="Not authorized")

    return StandardResponse(
        status="success",
        status_code=200,
        message="Project default sectors retrieved successfully",
        data={
            "project_id": project_id,
            "project_name": project.get("project_name"),
            "default_sectors": project.get("sectors", []),
            "industry": project.get("industry")
        }
    )

@router.patch("/project/{project_id}/sectors", response_model=StandardResponse[dict])
async def update_project_sectors(
    project_id: str,
    sectors: List[str],
    current_user: UserPayload = Depends(require_roles("admin")),
):
    # ... (Same as your provided code) ...
    user_id = current_user.user_id
    try:
        project_obj_id = ObjectId(project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await project_collection.find_one({"_id": project_obj_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Only owner admin can update
    if project.get("user_id") != user_id:
         raise HTTPException(status_code=403, detail="Access denied")

    try:
        cleaned_sectors = validate_sectors(sectors)
        cleaned_sectors = [s.upper() for s in cleaned_sectors]
    except HTTPException as e:
        raise e

    result = await project_collection.update_one(
        {"_id": project_obj_id},
        {
            "$set": {
                "sectors": cleaned_sectors,
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )
    return StandardResponse(
        status="success",
        status_code=200,
        message="Project sectors updated successfully",
        data={
            "project_id": project_id,
            "updated_sectors": cleaned_sectors
        }
    )

# Delete only by admin and only their own projects

@router.delete("/project/{project_id}", response_model=StandardResponse[dict])
async def delete_project(
    project_id: str,
    current_user: UserPayload = Depends(require_roles("admin")),
):
    try:
        project_obj_id = ObjectId(project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    # --- SECURITY CHECK ---
    # First, find the project to check ownership before deleting
    project = await project_collection.find_one({"_id": project_obj_id})
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    # Ensure Admin A cannot delete Admin B's project
    if project.get("user_id") != current_user.user_id:
        raise HTTPException(
            status_code=403, 
            detail="Access denied. You can only delete projects you created."
        )

    result = await project_collection.delete_one({"_id": project_obj_id})

    if result.deleted_count == 1:
        return StandardResponse(
            status="success",
            status_code=200,
            message="Project deleted successfully",
            data={"project_id": project_id}
        )
    
    raise HTTPException(status_code=500, detail="Failed to delete project")