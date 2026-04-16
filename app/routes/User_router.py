# app/routes/User_router.py
from fastapi import APIRouter, HTTPException, Depends
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from bson import ObjectId
from datetime import datetime, timezone
import asyncio
import os # <--- ADDED: For physical file deletion
# Add this to your existing imports
from app.middleware.role_checker import require_roles
import logging

# --- Logger Setup ---
logger = logging.getLogger(__name__)

# --- Local Imports ---
from app.models.user import User
# --- Import all ASYNC collections ---
from app.database import (
    User_collection, 
    project_collection, 
    file_collection, 
    cache_collection,
    chat_history_collection,       # <--- ADDED
    user_file_selection_collection # <--- ADDED
)
from app.auth.jwt_handler import create_access_token, decode_access_token, decode_access_token_allow_expired
from app.dependencies import get_current_user, UserPayload
from app.schemas import StandardResponse


# --- Import Qdrant client and models ---
from app.core.llm_provider import get_qdrant_client
from app.config import settings
from qdrant_client import models

# --- Validation Middleware ---
from app.middleware.user_middleware import (
    validate_email,
    validate_name,
    validate_password,
    validate_role,
)

class OrgUserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

# --- Response Helpers ---
from app.middleware.response_helper_middleware import success_response, error_response_dict

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    token: str

@router.post("/register")
async def register_user(user: User):
    """
    Registers a new user after validating all input fields using middleware validators.
    Creates a hashed password and stores user in database.
    """
    try:
        # --- Validate all user inputs using middleware ---
        validate_name(user.name)
        validate_email(user.email)
        validate_password(user.password)
        validate_role(user.role)

        # Normalize email to lowercase
        normalized_email = user.email.lower()
        
        # Check if email already exists
        existing_user = await User_collection.find_one({"email": normalized_email})
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail=error_response_dict(
                    message="Email already registered",
                    code="EMAIL_EXISTS",
                    status_code=400
                )
            )
        
        # Hash password (limit to 72 chars for bcrypt)
        hashed_pw = await asyncio.to_thread(pwd_context.hash, user.password[:72])
        
        # Prepare user document
        user_dict = user.model_dump()
        user_dict["email"] = normalized_email
        user_dict["password"] = hashed_pw
        
        # Insert into database
        await User_collection.insert_one(user_dict)
        
        return success_response(
            data={"email": normalized_email, "name": user.name, "role": user.role},
            message="User registered successfully",
            status_code=200
        )
    
    except HTTPException as e:
        # Re-raise HTTP exceptions (validation errors, duplicate email)
        raise e
    except Exception as e:
        # Catch unexpected errors
        raise HTTPException(
            status_code=500,
            detail=error_response_dict(
                message=f"An unexpected error occurred during registration: {str(e)}",
                code="REGISTRATION_ERROR",
                status_code=500
            )
        )

@router.post("/login")
async def login_user(login: LoginRequest):
    """
    Authenticates user with email and password.
    Returns JWT access token on successful authentication.
    """
    try:
        normalized_email = login.email.lower()
        
        # Find user by email
        user = await User_collection.find_one({"email": normalized_email})
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail=error_response_dict(
                    message="Invalid email or password",
                    code="INVALID_CREDENTIALS",
                    status_code=401
                )
            )

        # Verify password
        is_valid_password = await asyncio.to_thread(
            pwd_context.verify, 
            login.password, 
            user["password"]
        )
        
        if not is_valid_password:
            raise HTTPException(
                status_code=401,
                detail=error_response_dict(
                    message="Invalid email or password",
                    code="INVALID_CREDENTIALS",
                    status_code=401
                )
            )
        
        # Create JWT token
        token = create_access_token({
            "user_id": str(user["_id"]), 
            "sub": user["email"],
            "role": user["role"]
        })
        
        return success_response(
            data={
                "access_token": token, 
                "token_type": "bearer",
                "user": {
                    "email": user["email"],
                    "name": user.get("name"),
                    "role": user["role"]
                }
            },
            message="Login successful",
            status_code=200
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response_dict(
                message=f"An unexpected error occurred during login: {str(e)}",
                code="LOGIN_ERROR",
                status_code=500
            )
        )

@router.post("/refresh")
async def refresh_token(payload: RefreshTokenRequest):
    """
    Refreshes the user's access token using the previous access token payload.
    Expiration is ignored during decode, but signature and required claims must
    still be valid.
    """
    try:
        decoded_token = decode_access_token_allow_expired(payload.token)

        if decoded_token is None:
            raise HTTPException(
                status_code=401,
                detail=error_response_dict(
                    message="Invalid refresh token",
                    code="INVALID_REFRESH_TOKEN",
                    status_code=401
                )
            )

        user_id = decoded_token.get("user_id")
        user_email = decoded_token.get("sub")
        user_role = decoded_token.get("role")

        if not user_id or not user_email or not user_role:
            raise HTTPException(
                status_code=401,
                detail=error_response_dict(
                    message="Refresh token payload is invalid",
                    code="INVALID_REFRESH_TOKEN_PAYLOAD",
                    status_code=401
                )
            )

        new_token = create_access_token({
            "user_id": user_id,
            "sub": user_email,
            "role": user_role
        })

        return success_response(
            data={
                "access_token": new_token,
                "token_type": "bearer"
            },
            message="Token refreshed successfully",
            status_code=200
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response_dict(
                message=f"An error occurred during token refresh: {str(e)}",
                code="REFRESH_TOKEN_ERROR",
                status_code=500
            )
        )

@router.delete("/me")
async def delete_user(current_user: UserPayload = Depends(get_current_user)):
    """
    Deletes the authenticated user and ALL their associated data:
    1. Chat History
    2. File Selections
    3. Projects
    4. Physical Files & Metadata
    5. Vector Embeddings (Qdrant)
    6. Cache Entries
    7. User Account
    """
    user_id = current_user.user_id
    logger.info(f"INFO: Initiating delete for user_id: {user_id}")

    try:
        # 1. Find all projects for this user
        projects_cursor = project_collection.find({"user_id": user_id})
        project_ids = [str(p["_id"]) async for p in projects_cursor]
        logger.info(f"INFO: Found {len(project_ids)} projects to delete.")

        # 2. Delete all Qdrant vector chunks owned by this user
        logger.info(f"INFO: Deleting vector data for user_id: {user_id}...")
        qdrant_client = get_qdrant_client()
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.owner_id",
                    match=models.MatchValue(value=user_id)
                )
            ]
        )
        
        await asyncio.to_thread(
            qdrant_client.delete,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points_selector=qdrant_filter
        )
        logger.info("INFO: Qdrant deletion complete.")

        # 3. Delete all RAG cache entries for this user
        cache_result = await cache_collection.delete_many({"user_id": user_id})
        logger.info(f"INFO: Deleted {cache_result.deleted_count} cache entries.")

        # 4. Delete Chat History (NEW)
        chat_result = await chat_history_collection.delete_many({"user_id": user_id})
        logger.info(f"INFO: Deleted {chat_result.deleted_count} chat history entries.")

        # 5. Delete File Selections (NEW)
        selection_result = await user_file_selection_collection.delete_many({"user_id": user_id})
        logger.info(f"INFO: Deleted {selection_result.deleted_count} file selection entries.")

        # 6. Delete all file metadata AND PHYSICAL FILES (UPDATED)
        file_count = 0
        if project_ids:
            # --- START PHYSICAL FILE DELETION ---
            # Fetch files first to get paths before deleting records
            files_cursor = file_collection.find({"project_id": {"$in": project_ids}})
            async for file_doc in files_cursor:
                file_path = file_doc.get("file_path")
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"INFO: Deleted physical file: {file_path}")
                    except Exception as e:
                        logger.warning(f"WARNING: Failed to delete physical file {file_path}: {e}")
            # --- END PHYSICAL FILE DELETION ---

            # Now delete database records
            file_result = await file_collection.delete_many({"project_id": {"$in": project_ids}})
            file_count = file_result.deleted_count
            logger.info(f"INFO: Deleted {file_count} file metadata entries.")

        # 7. Delete all projects from MongoDB
        project_result = await project_collection.delete_many({"user_id": user_id})
        logger.info(f"INFO: Deleted {project_result.deleted_count} project entries.")
        
        # 8. Delete the user
        user_delete_result = await User_collection.delete_one({"_id": ObjectId(user_id)})
        
        if user_delete_result.deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail=error_response_dict(
                    message="User not found",
                    code="USER_NOT_FOUND",
                    status_code=404
                )
            )

        logger.info(f"INFO: Successfully deleted user {user_id}.")

        return success_response(
            data={
                "deleted_user_id": user_id,
                "deleted_projects": len(project_ids),
                "deleted_files": file_count,
                "deleted_chat_messages": chat_result.deleted_count,
                "deleted_cache_entries": cache_result.deleted_count
            },
            message="User account and all associated data successfully deleted",
            status_code=200
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"ERROR: Failed during user deletion: {e}")
        raise HTTPException(
            status_code=500,
            detail=error_response_dict(
                message=f"An error occurred during account deletion: {str(e)}",
                code="DELETE_USER_ERROR",
                status_code=500
            )
        )
    

@router.delete("/org/delete/{user_id}")
async def admin_delete_user(
    user_id: str,
    current_user: UserPayload = Depends(require_roles("admin"))
):
    """
    Admin deletes a user that was created by them.
    Reuses the same delete logic as /users/me by passing
    a real UserPayload built from the DB record.
    """

    # 1. Fetch the user being deleted (we need their real email + role)
    user = await User_collection.find_one({"_id": ObjectId(user_id)})

    if not user:
        raise HTTPException(
            status_code=404,
            detail=error_response_dict(
                message="User not found",
                code="USER_NOT_FOUND",
                status_code=404
            )
        )

    # 2. Ensure this user belongs to the admin
    if user.get("admin_id") != current_user.user_id:
        raise HTTPException(
            status_code=403,
            detail=error_response_dict(
                message="You cannot delete users you did not create",
                code="NOT_ALLOWED",
                status_code=403
            )
        )

    # 3. Build a REAL UserPayload (email & role required by Pydantic)
    temp_payload = UserPayload(
        user_id=str(user["_id"]),
        email=user["email"],
        role=user["role"]
    )

    # 4. Reuse existing delete logic
    return await delete_user(current_user=temp_payload)



@router.get("/me")
async def get_current_user_info(current_user: UserPayload = Depends(get_current_user)):
    """
    Returns the current authenticated user's information.
    """
    try:
        user = await User_collection.find_one({"_id": ObjectId(current_user.user_id)})
        
        if not user:
            raise HTTPException(
                status_code=404,
                detail=error_response_dict(
                    message="User not found",
                    code="USER_NOT_FOUND",
                    status_code=404
                )
            )
        
        # Remove sensitive data
        user.pop("password", None)
        user["_id"] = str(user["_id"])
        
        return success_response(
            data=user,
            message="User information retrieved successfully",
            status_code=200
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response_dict(
                message=f"An error occurred while fetching user info: {str(e)}",
                code="GET_USER_ERROR",
                status_code=500
            )
        )
    
@router.post("/org/register")
async def register_organization_user(
    user_data: OrgUserCreate,  # <--- Use the new cleaner model here
    current_user: UserPayload = Depends(require_roles("admin"))
):
    """
    Admin creates a user account.
    - 'role' is automatically set to 'user'.
    - 'admin_id' is automatically linked to the logged-in Admin.
    """
    try:
        # 1. Validate inputs
        validate_name(user_data.name)
        validate_email(user_data.email)
        validate_password(user_data.password)

        # 2. Normalize email
        normalized_email = user_data.email.lower()
        
        # 3. Check duplicate email
        existing_user = await User_collection.find_one({"email": normalized_email})
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail=error_response_dict(
                    message="Email already registered",
                    code="EMAIL_EXISTS",
                    status_code=400
                )
            )
        
        # 4. Hash password
        hashed_pw = await asyncio.to_thread(pwd_context.hash, user_data.password[:72])
        
        # 5. Prepare user document for DB (Using the full User structure)
        # We manually construct the dictionary to ensure control over sensitive fields
        user_db_record = {
            "name": user_data.name,
            "email": normalized_email,
            "password": hashed_pw,
            "role": "user",                   # <--- FORCED
            "admin_id": current_user.user_id, # <--- AUTOMATICALLY LINKED
            
        }
        
        # 6. Insert into database
        result = await User_collection.insert_one(user_db_record)
        
        return success_response(
            data={
                "user_id": str(result.inserted_id),
                "email": normalized_email, 
                "name": user_data.name, 
                "role": "user",
                "created_by_admin": current_user.user_id
            },
            message="Organization user created successfully",
            status_code=200
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response_dict(
                message=f"Error creating org user: {str(e)}",
                code="ORG_REGISTRATION_ERROR",
                status_code=500
            )
        )

@router.get("/org/users")
async def get_my_organization_users(
    current_user: UserPayload = Depends(require_roles("admin"))
):
    """
    List only the users that were created by the current Admin.
    """
    try:
        users_list = []
        
        # Find users where 'admin_id' matches the current Admin's ID
        cursor = User_collection.find({"admin_id": current_user.user_id})
        
        async for u in cursor:
            users_list.append({
                "user_id": str(u["_id"]),
                "name": u.get("name"),
                "email": u.get("email"),
                "role": u.get("role"),
                # "created_at": ... (if you have this field)
            })

        return success_response(
            data=users_list,
            message=f"Retrieved {len(users_list)} organization users",
            status_code=200
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response_dict(
                message=f"Error fetching users: {str(e)}",
                code="GET_ORG_USERS_ERROR",
                status_code=500
            )
        )