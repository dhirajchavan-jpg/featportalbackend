# app/routes/async_query_router.py
from fastapi import APIRouter, Depends, HTTPException
from app.dependencies import UserPayload, get_current_user
from app.schemas import QueryRequest, StandardResponse
from app.services.redis.redis_service import redis_service
from app.middleware.prompt_validation import get_prompt_validator
from app.database import project_config_collection
from app.config import settings
from app.utils.logger import setup_logger
from app.middleware import files_middleware

logger = setup_logger()
router = APIRouter()


@router.post("/query/async", response_model=StandardResponse, tags=["Async RAG"])
async def submit_async_query(
    query_request: QueryRequest,
    current_user: UserPayload = Depends(get_current_user),
):
    """Submits a RAG query to Redis and returns task id immediately."""
    logger.info(f" [AsyncRouter] Incoming Query Request | User: {current_user.user_id}")
    logger.info(f" [AsyncRouter] User selected style: {query_request.style}")

    try:
        validator = get_prompt_validator()
        validator.validate(query_request.query, raise_on_detection=True)

        project_doc = await files_middleware.verify_project(query_request.project_id, current_user)
        organization_sector = (project_doc.get("organization_sector") or "").strip().upper()
        if not organization_sector:
            raise HTTPException(status_code=422, detail="Project organization sector is missing")

        project_config = await project_config_collection.find_one({"project_id": query_request.project_id})

        ai_settings = None
        if project_config:
            ai_settings = {
                "router_model": project_config.get("router_model", settings.ROUTER_MODEL),
                "simple_model": project_config.get("simple_model", settings.LLM_MODEL_SIMPLE),
                "complex_model": project_config.get("complex_model", settings.LLM_MODEL_COMPLEX),
                "search_strategy": project_config.get("search_strategy", "hybrid"),
                "retrieval_depth": project_config.get("retrieval_depth", 5),
                "enable_reranking": project_config.get("enable_reranking", True),
                "hallucination_threshold": project_config.get("hallucination_threshold", 100.0)
            }

        query_data = query_request.dict()
        query_data["sector"] = organization_sector
        query_data["ai_config"] = ai_settings

        user_data = {
            "user_id": current_user.user_id,
            "email": current_user.email,
            "role": current_user.role
        }

        task_id = await redis_service.enqueue_job(
            job_type="query",
            job_data=query_data,
            user_data=user_data
        )

        return StandardResponse(
            status="success",
            status_code=202,
            message="Query submitted for processing",
            data={"task_id": task_id, "status": "queued"}
        )

    except HTTPException as e:
        logger.warning(f" [AsyncRouter] Validation/HTTP Error: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f" [AsyncRouter] Unexpected Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/status/{task_id}", response_model=StandardResponse, tags=["Async RAG"])
async def check_query_status(
    task_id: str,
    current_user: UserPayload = Depends(get_current_user)
):
    """Polls status for queued async query."""
    logger.debug(f" [AsyncRouter] Checking status for Task ID: {task_id}")

    try:
        job_result = await redis_service.get_job_status(task_id)

        if not job_result:
            raise HTTPException(status_code=404, detail="Task not found or expired")

        status = job_result.get("status")
        if status in ["queued", "processing"]:
            return StandardResponse(
                status="success",
                message="Query is being processed...",
                data={"task_id": task_id, "status": status}
            )

        if status == "failed":
            error_msg = job_result.get("data", {}).get("error", "Unknown error")
            return StandardResponse(
                status="error",
                status_code=500,
                message="Processing failed",
                data={"task_id": task_id, "status": "failed", "error": error_msg}
            )

        if status == "completed":
            return StandardResponse(
                status="success",
                status_code=200,
                message="Query completed",
                data={
                    "task_id": task_id,
                    "status": "completed",
                    **job_result.get("data", {})
                }
            )

        return StandardResponse(data={"status": "unknown"})

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f" [AsyncRouter] Error checking status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
