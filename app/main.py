# app/main.py
import os
import asyncio
import time
import httpx
import uuid
from app.core.llm_provider import get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.utils.log_reader import get_logs_by_date_range
from app.utils.logging_config import setup_logging
# from fastapi.responses import PlainTextResponse
from datetime import date, datetime, timedelta
import pandas as pd
import aiofiles
from app.schemas import FullSystemMonitor
from io import BytesIO
from fastapi.responses import StreamingResponse, FileResponse
from app.database import project_config_collection, chat_history_collection

from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks,Query
from typing import List
from opentelemetry import trace
from app.middleware.rate_limit_middleware import RateLimitMiddleware
from fastapi.middleware.cors import CORSMiddleware
from app.middleware.exception_handler import http_exception_handler, general_exception_handler
from app.middleware.request_id_middleware import request_id_middleware
from app.core.phoenix_utils import evaluate_rag_interaction

from app.middleware import files_middleware 
from app.config import settings, normalize_complex_model
from app.database import create_indexes, chat_history_collection,verify_connection,close_mongo_connection,project_config_collection
from app.dependencies import UserPayload, get_current_user
from app.middleware.custom_expection import CustomExceptionMiddleware
from app.middleware.monitoring import MonitoringMiddleware, metrics_app, get_analytics_summary, system_metrics, update_system_metrics
from app.core.llm_provider import load_models
from app.utils.logger import setup_logger
from app.middleware.prompt_validation import get_prompt_validator
from fastapi.staticfiles import StaticFiles
from app.routes.file_viewer import router as file_viewer_router
from app.middleware.role_checker import require_roles

from app.routes import evaluation
from fastapi import BackgroundTasks
# from app.core.phoenix_utils import setup_phoenix_instrumentation, evaluate_hallucination # <--- Import custom utils

from app.schemas import (
    DeleteRequest, QueryRequest, UpdateRequest,
    StandardResponse, QueryResponseData, RagUploadResponseData,
    SourceDocument, ChatHistoryEntry, RetrievalStats
)
# from app.services import rag_service
from app.services.rag.file_indexing import process_and_index_file, delete_document_by_source, process_and_index_global_file, delete_global_document_by_source, update_document_sector as update_document_sector_service
from app.services.rag.history_and_cache import _save_file_upload_to_history
from app.services.rag.pipeline_orchestrator import query_rag_pipeline
from app.monitoring.reports.reports_service import generate_report
from app.monitoring.repositories.drift_query_repository import get_drift_queries

import magic
import tempfile

# ============================================================================
# NEW IMPORTS FOR ASYNC RAG
# ============================================================================
from app.routes.async_query_router import router as AsyncQueryRouter
from app.services.redis.redis_service import redis_service 

# ============================================================================
# ARIZE PHOENIX IMPORTS
# ============================================================================
from phoenix.otel import register as phoenix_register
setup_logging()

logger = setup_logger()

# ============================================================================
# PHOENIX TRACER PROVIDER (initialized in lifespan)
# ============================================================================
tracer_provider = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tracer_provider
    logger.info("[INFO] Server starting up...")

    # ========================================================================
    # ARIZE PHOENIX INITIALIZATION
    # ========================================================================
    try:
        tracer_provider = phoenix_register(
            project_name=os.getenv("PHOENIX_PROJECT_NAME", "compliance-rag-api2.0"),
            endpoint="http://localhost:6006/v1/traces",
            batch=True,  # Production-ready batching
            auto_instrument=True,  # Auto-trace OpenAI, LangChain, LlamaIndex, etc.
        )
        logger.info("[INFO] Arize Phoenix tracing initialized successfully.")
    except Exception as e:
        logger.warning(f"[WARNING] Failed to initialize Phoenix tracing: {e}")
        tracer_provider = None
    # ========================================================================

    # ========================================================================
    # NEW: VERIFY REDIS CONNECTION (Async Queue Check)
    # ========================================================================
    try:
        await redis_service.client.ping()
        logger.info(f"[INFO] Connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    except Exception as e:
        logger.warning(f"[WARNING] Redis connection failed: {e}. Async features may not work.")
    # ========================================================================

    # Verify MongoDB connection with pooling config (non-fatal if unavailable)
    mongo_connected = await verify_connection()
    if mongo_connected:
        await create_indexes()
    else:
        logger.warning("[WARNING] MongoDB is unavailable at startup. Continuing without DB-dependent features.")

    load_models()
    logger.info("[INFO] All models loaded. Server is ready.")
    yield
    # Shutdown handler
    logger.info("[INFO] Server shutting down...")
    logger.info("[INFO] Cleaning up resources...")
    
    # ========================================================================
    # ARIZE PHOENIX CLEANUP
    # ========================================================================
    if tracer_provider:
        try:
            tracer_provider.shutdown()
            logger.info("[INFO] Phoenix tracer provider shut down.")
        except Exception as e:
            logger.warning(f"[WARNING] Error shutting down Phoenix: {e}")
    # ========================================================================
    
    logger.info("[INFO] Shutdown complete.")

    #  Close MongoDB connection pool gracefully
    await close_mongo_connection()

app = FastAPI(
    title="Secure Compliance RAG API", 
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://featcomply.featsystems.ai:8071","http://featcomply.featsystems.ai:8070","http://localhost:3000","http://localhost:3001","http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MonitoringMiddleware)

# Mount Prometheus metrics endpoint
app.mount("/metrics", metrics_app)

def count_total_api_routes(app: FastAPI):
    routes = []
    for route in app.routes:
        # Only include real HTTP routes, skip websockets/staticfiles/etc.
        if hasattr(route, "methods") and hasattr(route, "path"):
            if route.path.startswith("/docs") or route.path.startswith("/openapi"):
                continue
            if route.path.startswith("/redoc"):
                continue

            routes.append({
                "path": route.path,
                "methods": list(route.methods)
            })

    return len(routes)


# Analytics endpoint for React dashboard
@app.get("/admin/analytics")
async def get_analytics():
    """Get detailed analytics for admin dashboard"""
    summary = get_analytics_summary()
    
    # Calculate overall statistics
    total_requests = sum(item['total_requests'] for item in summary)
    total_errors = sum(item['error_count'] for item in summary)
    overall_error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0

    # NEW: Count actual API routes (total endpoints)
    total_api_endpoints = count_total_api_routes(app)

    # Active endpoints = only those hit at least once
    active_endpoints = len([e for e in summary if e['total_requests'] > 0])

    return {
        'endpoints': summary,
        'system': system_metrics,
        'overview': {
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate': round(overall_error_rate, 2),

            # NEW VALUES
            'total_endpoints': total_api_endpoints,
            'active_endpoints': active_endpoints,
        }
    }

# Start system metrics collector on startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(update_system_metrics())




os.makedirs("uploads", exist_ok=True)


# app.mount("/uploads", StaticFiles(directory="docs"), name="uploads")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.add_middleware(RateLimitMiddleware)

app.middleware("http")(request_id_middleware)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

from app.routes.User_router import router as User_router
from app.routes.project_router import router as Project_router
from app.routes.file_router import router as File_router
from app.routes.admin_router import router as Admin_router
from app.routes.super_admin import router as SuperAdmin_router

# ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
# MAX_FILE_SIZE = 10 * 1024 * 1024
# ALLOWED_MIME_TYPES = {
#     ".pdf": "application/pdf",
#     ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
#     ".txt": "text/plain"
# }

app.add_middleware(CustomExceptionMiddleware)

app.include_router(evaluation.router)
app.include_router(User_router, prefix="/users", tags=["User Management"])
app.include_router(Project_router, prefix="/projects", tags=["Projects"])
app.include_router(File_router, prefix="/files", tags=["File Management & Upload"])
app.include_router(Admin_router, prefix="/admin") 
app.include_router(SuperAdmin_router, prefix="/super-admin", tags=["Super Admin Management"])
app.include_router(file_viewer_router)

# ========================================================================
# NEW: REGISTER ASYNC ROUTER (Exposes /rag/query/async)
# ========================================================================
app.include_router(AsyncQueryRouter, prefix="/rag", tags=["Async Query"])


@app.on_event("startup")
async def startup_event():
    start_report_scheduler()
    
@app.post("/internal/test-report/{report_type}")
async def test_report(report_type: str):
    return await generate_report(report_type.upper())

@app.get("/reports/{report_reference_id}/drift-queries")
async def fetch_report_drift_queries(
    report_reference_id: str,
):
    return await get_drift_queries(
        report_reference_id=report_reference_id
    )

@app.get("/reports/{report_reference_id}/download-excel")
async def download_drift_report_excel(report_reference_id: str):
    """
    Streams an Excel file containing ALL drift queries for a specific report.
    This URL can be embedded directly into a PDF.
    """
    # 1. Fetch the data using your existing repository function
    data = await get_drift_queries(report_reference_id=report_reference_id)
    items = data.get("items", [])

    # 2. Convert to Pandas DataFrame
    if not items:
        # Handle empty case
        df = pd.DataFrame(columns=["timestamp", "drift_type", "query", "score"])
    else:
        # Flatten the data for Excel
        export_data = []
        for item in items:
            export_data.append({
                "Timestamp": item.get("evaluated_at"),
                "Drift Type": item.get("drift_type"),
                "Query": item.get("query"),
                "Score": item.get("score")
            })
        df = pd.DataFrame(export_data)

    # 3. Write to Excel Memory Buffer
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Drift Data")
    
    output.seek(0)

    # 4. Stream the Response
    filename = f"Drift_Report_{report_reference_id}.xlsx"
    headers = {
        'Content-Disposition': f'attachment; filename="{filename}"'
    }
    
    return StreamingResponse(
        output, 
        headers=headers, 
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

async def stream_to_disk(upload_file: UploadFile, dest_path: str):
    """Stream file directly to disk without loading into memory"""
    async with aiofiles.open(dest_path, 'wb') as f:
        while chunk := await upload_file.read(8192):  # 8KB chunks
            await f.write(chunk)


@app.get("/health", response_model=StandardResponse[dict], tags=["System"])
async def health_check():
    """Health check endpoint to verify system status."""
    import torch
    
    system_status = {
        "status": "healthy",
        "api_version": "3.0.0",
        "cuda_available": torch.cuda.is_available(),
        "phoenix_tracing": tracer_provider is not None,
    }
    
    if torch.cuda.is_available():
        system_status["gpu_device"] = torch.cuda.get_device_name(0)
        system_status["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 2)
    
    return StandardResponse(
        status="success",
        status_code=200,
        message="System is healthy",
        data=system_status
    )

@app.post("/upload", response_model=StandardResponse[RagUploadResponseData], tags=["RAG Document Management"])
async def upload_document(
    project_id: str = Form(...),
    file: UploadFile = File(...),
    current_user: UserPayload = Depends(require_roles("admin", "super_admin")),
):
    # Resolve sector from project details only (single-sector model).
    if not project_id or not project_id.strip():
        raise HTTPException(status_code=422, detail="Project ID field cannot be empty.")

    normalized_project_id = project_id.lower().strip()

    try:
        project_doc = await files_middleware.verify_project(normalized_project_id, current_user)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Project verification failed: {e}")

    org_sector = (project_doc.get("organization_sector") or "").strip().upper()
    if not org_sector:
        raise HTTPException(status_code=422, detail="Project organization sector is missing. Please update project details.")

    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in settings.ALLOWED_MIME_TYPES.keys():
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(settings.ALLOWED_MIME_TYPES.keys())}"
        )

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds {settings.MAX_FILE_SIZE / (1024*1024)} MB"
        )

    try:
        mime_type = await asyncio.to_thread(magic.from_buffer, content, mime=True)
        expected_mime = settings.ALLOWED_MIME_TYPES.get(file_extension)
        if not expected_mime or mime_type != expected_mime:
            raise HTTPException(
                status_code=400,
                detail=f"File type mismatch. Extension is '{file_extension}' but content is '{mime_type}'. Dangerous file rejected."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during file type validation: {e}")

    file_id = str(uuid.uuid4())
    safe_filename = file.filename.replace(" ", "_").replace("/", "")
    unique_filename = f"{file_id}-{safe_filename}"
    permanent_file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name

    try:
        def write_temp_file():
            with open(temp_file_path, "wb") as buffer:
                buffer.write(content)
        await asyncio.to_thread(write_temp_file)

        chat_id = f"user_{current_user.user_id}_project_{normalized_project_id}"

        logger.info(f"--- [SYNC] Starting indexing for {file.filename} from temp file... ---")
        await asyncio.to_thread(
            process_and_index_file,
            file_path=temp_file_path,
            project_id=normalized_project_id,
            sector=org_sector,
            current_user=current_user,
            original_filename=file.filename
        )
        logger.info(f"--- [SYNC] Finished indexing for {file.filename}. ---")

        await _save_file_upload_to_history(
            chat_id=chat_id,
            user_id=current_user.user_id,
            project_id=normalized_project_id,
            sector=org_sector,
            file_id=file_id,
            file_name=file.filename
        )

        def write_permanent_file():
            with open(permanent_file_path, "wb") as buffer:
                buffer.write(content)
        await asyncio.to_thread(write_permanent_file)

        response_data = RagUploadResponseData(
            file_id=file_id,
            file_name=file.filename,
            sector=org_sector,
            message="File uploaded and indexed successfully."
        )

        return StandardResponse(
            status_code=200,
            message="File successfully processed and indexed.",
            data=response_data
        )

    except Exception as e:
        logger.error(f"[ERROR] Upload failed: {e}. Starting rollback...")
        try:
            await delete_document_by_source(
                filename=file.filename,
                project_id=normalized_project_id,
                current_user=current_user
            )
        except Exception as delete_e:
            logger.error(f"[ROLLBACK ERROR] Failed to delete from Qdrant: {delete_e}")

        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    finally:
        try:
            temp_file.close()
        except Exception:
            pass
        await file.close()


@app.post("/admin/upload-global", tags=["Admin"])
async def upload_global_sector_file(*args, **kwargs):
    raise HTTPException(
        status_code=410,
        detail="Global/regulatory sector uploads are deprecated. Use project-scoped organization uploads only."
    )

@app.get("/query/history/{project_id}", response_model=StandardResponse[List[ChatHistoryEntry]], tags=["RAG Query"])
async def get_chat_history(
    project_id: str,
    current_user: UserPayload = Depends(get_current_user)
):
    normalized_project_id = project_id.lower().strip()
    chat_id = f"user_{current_user.user_id}_project_{normalized_project_id}"
    
    # --- 1. Fetch Project Configuration for Limit ---
    # Default to 15 if config is missing or field is not set
    history_limit = 15 
    
    config = await project_config_collection.find_one({"project_id": project_id})
    if config and "chat_history_limit" in config:
        history_limit = config["chat_history_limit"]

    # --- 2. Define Projection ---
    projection = {
        "created_at": 1,
        "message_type": 1,
        "user_query": 1,
        "llm_answer": 1,
        "source_documents": 1,
        "file_id": 1,
        "file_name": 1,
        "sector": 1,
        "_id": 0
    }
    
    # --- 3. Query with Dynamic Limit ---
    history_cursor = chat_history_collection.find(
        {"chat_id": chat_id},
        projection=projection
    ).sort("created_at", -1).limit(history_limit) # <--- Use the variable here
    
    history_list = await history_cursor.to_list(length=history_limit)
    history_list.reverse() 
    
    for message in history_list:
        if "sector" not in message:
            message["sector"] = "unknown"
    
    return StandardResponse(
        status="success",
        status_code=200,
        message=f"Chat history retrieved successfully (Limit: {history_limit}).",
        data=history_list
    )

@app.post("/query", response_model=StandardResponse[QueryResponseData], tags=["RAG Query"])
async def query_documents(
    query_request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: UserPayload = Depends(get_current_user),
):
    """Single-sector query endpoint with project-scoped organization sector."""
    validator = get_prompt_validator()
    validator.validate(query_request.query, raise_on_detection=True)

    logger.info(f"\n{'='*50}")
    logger.info(" [API] Incoming Query Request")
    logger.info(f" User: {current_user.user_id}")
    logger.info(f" Project ID: {query_request.project_id}")
    logger.info(f" Query: {query_request.query}")
    logger.info(f"{'='*50}")

    # Resolve organization sector from project details (source of truth).
    project_doc = await files_middleware.verify_project(query_request.project_id, current_user)
    organization_sector = (project_doc.get("organization_sector") or "").strip().upper()
    if not organization_sector:
        raise HTTPException(status_code=422, detail="Project organization sector is missing")

    try:
        logger.info(f" [Config] Attempting to fetch AI Config for Project: {query_request.project_id}")
        project_config = await project_config_collection.find_one({"project_id": query_request.project_id})

        if project_config:
            logger.info("[Config] Found custom configuration in MongoDB.")
            ai_settings = {
                "router_model": project_config.get("router_model", settings.ROUTER_MODEL),
                "simple_model": project_config.get("simple_model", settings.LLM_MODEL_SIMPLE),
                "complex_model": normalize_complex_model(project_config.get("complex_model")),
                "search_strategy": project_config.get("search_strategy", "hybrid"),
                "retrieval_depth": project_config.get("retrieval_depth", 5),
                "enable_reranking": project_config.get("enable_reranking", True),
                "chat_history_limit": project_config.get("chat_history_limit", 10)
            }
        else:
            logger.info(" [Config] No custom configuration found. Falling back to System Defaults.")
            ai_settings = {
                "router_model": settings.ROUTER_MODEL,
                "simple_model": settings.LLM_MODEL_SIMPLE,
                "complex_model": settings.LLM_MODEL_COMPLEX,
                "search_strategy": "hybrid",
                "retrieval_depth": 5,
                "enable_reranking": True,
                "chat_history_limit": 10
            }

        if query_request.top_k:
            ai_settings["retrieval_depth"] = query_request.top_k
        if query_request.use_reranking is not None:
            ai_settings["enable_reranking"] = query_request.use_reranking

    except Exception as e:
        logger.error(f" [Config Error] Failed to load settings: {e}. Falling back to defaults.")
        ai_settings = None

    logger.info(" [Pipeline] Starting RAG Pipeline Orchestrator...")
    result = await query_rag_pipeline(
        query=query_request.query,
        current_user=current_user,
        project_id=query_request.project_id,
        sectors=[organization_sector],
        excluded_files=query_request.excluded_files,
        ai_config=ai_settings,
        background_tasks=background_tasks
    )

    source_docs_list = []
    for doc in result.get("source_documents", []):
        if isinstance(doc, dict):
            source_docs_list.append(SourceDocument(
                page_content=doc.get("page_content", ""),
                metadata=doc.get("metadata", {}),
                relevance_score=doc.get("relevance_score"),
                dense_score=doc.get("dense_score"),
                sparse_score=doc.get("sparse_score"),
                rerank_score=doc.get("rerank_score")
            ))

    current_span = trace.get_current_span()
    span_context = current_span.get_span_context()
    trace_id = None
    if span_context.is_valid:
        span_id = format(span_context.span_id, "016x")
        trace_id = format(span_context.trace_id, "032x")
        background_tasks.add_task(
            evaluate_rag_interaction,
            query=query_request.query,
            response=result.get("result", ""),
            retrieved_docs=source_docs_list,
            span_id=span_id
        )

    pipe_stats = result.get("retrieval_stats", {})
    retrieval_stats = RetrievalStats(
        total_chunks_searched=pipe_stats.get("total_chunks_searched", len(source_docs_list)),
        chunks_retrieved=len(source_docs_list),
        sources_queried=result.get("sources_queried", [query_request.project_id]),
        excluded_files_count=result.get("excluded_files_count", 0),
        retrieval_method=pipe_stats.get("retrieval_method", "hybrid"),
        reranking_applied=pipe_stats.get("reranking_applied", True),
        processing_time_ms=result.get("processing_time", 0) * 1000
    )

    response_data = QueryResponseData(
        result=result.get("result", ""),
        source_documents=source_docs_list,
        retrieval_stats=retrieval_stats,
        sources_used=result.get("sources_used")
    )

    return StandardResponse(
        status="success",
        status_code=200,
        message="Query processed successfully",
        data=response_data,
        meta={
            "model_used": result.get("model_used"),
            "query_complexity": result.get("meta", {}).get("query_complexity"),
            "from_cache": result.get("from_cache", False),
            "trace_id": trace_id,
            "evaluation_status": "scheduled",
            "evaluation_dashboard": "http://localhost:8000/eval-dashboard"
        }
    )

@app.post("/delete-document", response_model=StandardResponse[dict], tags=["RAG Document Management"])
async def delete_document(
    delete_request: DeleteRequest,
    current_user: UserPayload = Depends(get_current_user),
):
    result = await delete_document_by_source(
        delete_request.filename, 
        delete_request.project_id,
        current_user
    )
    return StandardResponse(
        status="success",
        status_code=200,
        message="Deletion initiated successfully.",
        data={"qdrant_response": str(result)},
    )

@app.post("/update-document-sector", response_model=StandardResponse[dict], tags=["RAG Document Management"])
async def update_document_sector(
    update_request: UpdateRequest,
    current_user: UserPayload = Depends(get_current_user),
):
    result = await update_document_sector_service(
        update_request.filename, 
        update_request.project_id,
        update_request.new_sector, 
        current_user
    )
    return StandardResponse(
        status="success",
        status_code=200,
        message="Update successful.",
        data={"qdrant_response": str(result)},
    )

@app.get("/", response_model=StandardResponse[dict], tags=["Root"])
def read_root(): 
    return StandardResponse(
        status="success",
        status_code=200,
        message="Secure Compliance RAG API is running.",
        data={"docs_url": "/docs"},
    )

@app.get("/debug/export-chunks")
def export_chunks(source: str):
    qdrant = get_qdrant_client()
    if not qdrant:
        raise HTTPException(500, "Qdrant client not initialized")

    collection_name = "my_private_documents"

    chunks = []
    offset = 0
    limit = 200

    while True:

        # OLD Qdrant: scroll returns 1 tuple
        # NEW Qdrant: scroll returns 2 tuple
        result = qdrant.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset
        )

        # unpack safely
        if isinstance(result, tuple) and len(result) == 2:
            points, next_offset = result
        else:
            points = result
            next_offset = None

        # --- STOP CONDITION #1: no points returned ---
        if not points:
            break

        # --- process points ---
        for p in points:
            meta = p.payload.get("metadata", {})
            if meta.get("source") == source:
                chunks.append({
                    "chunk_id": meta.get("chunk_id"),
                    "text": p.payload.get("page_content"),
                    "page": meta.get("page_number")
                })

        # --- STOP CONDITION #2: next_offset didn’t change (OLD Qdrant) ---
        if next_offset is None or next_offset == offset:
            break

        # move forward
        offset = next_offset

    return {"total": len(chunks), "chunks": chunks}



@app.get("/logs/download-excel", tags=["Admin"])
async def download_logs_excel(
    start_date: date = Query(..., description="Format: YYYY-MM-DD (e.g., 2025-12-30)"), 
    end_date: date = Query(..., description="Format: YYYY-MM-DD (e.g., 2025-12-30)"),
    log_type: str = Query(..., description="Select 'system' or 'rag'", enum=["system", "rag"])
):
    """
    Download logs filtered by date and type.
    - log_type='system': Downloads logs/system/system.log
    - log_type='rag': Downloads logs/ai/ai_rag.log
    """
    BASE_LOG_DIR = "logs"
    
    # 1. Determine target file based on user selection
    if log_type == "system":
        target_file = os.path.join(BASE_LOG_DIR, "system", "system.log")
        sheet_name = "System Logs"
    elif log_type == "rag":
        target_file = os.path.join(BASE_LOG_DIR, "ai", "ai_rag.log")
        sheet_name = "AI RAG Logs"
    else:
        # This shouldn't happen due to 'enum' validation, but good for safety
        raise HTTPException(status_code=400, detail="Invalid log type.")

    # 2. Check if file exists
    parsed_data = []
    if os.path.exists(target_file):
        try:
            with open(target_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip(): continue

                    # 3. Parse Date & Filter
                    try:
                        # Extract date from start of line (YYYY-MM-DD)
                        # Example log: 2025-12-30 10:00:00 - ...
                        log_date_str = line[:10]
                        log_date = datetime.strptime(log_date_str, "%Y-%m-%d").date()
                        
                        # Filter Logic
                        if not (start_date <= log_date <= end_date):
                            continue
                    except ValueError:
                        # Skip lines that don't start with a date (stack traces, etc.)
                        # Or you can choose to include them as part of the previous log
                        continue

                    # 4. Structure the Data
                    parts = line.split(' - ')
                    if len(parts) >= 3:
                        parsed_data.append({
                            "Timestamp": parts[0].strip(),
                            "Logger": parts[1].strip(),
                            "Level": parts[2].strip(),
                            "Message": " - ".join(parts[3:]).strip()
                        })
                    else:
                        parsed_data.append({
                            "Timestamp": "", 
                            "Logger": "RAW", 
                            "Level": "", 
                            "Message": line.strip()
                        })

        except Exception as e:
            logger.error(f"Error reading log file {target_file}: {e}")
            raise HTTPException(status_code=500, detail="Failed to read log file")

    # 5. Create DataFrame
    df = pd.DataFrame(parsed_data)
    if df.empty:
        # Return empty DF structure if no logs found
        df = pd.DataFrame(columns=["Timestamp", "Logger", "Level", "Message"])

    # 6. Generate Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    
    output.seek(0)

    # 7. Return File
    filename = f"{log_type}_logs_{start_date}_{end_date}.xlsx"
    headers = {
        'Content-Disposition': f'attachment; filename="{filename}"'
    }
    
    return StreamingResponse(
        output, 
        headers=headers, 
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# from app.core.llm_provider import get_llm_stats
# import psutil

# # ... (other imports)

# # Global http_client (keep this as you have it)
# http_client = httpx.AsyncClient(timeout=20.0)
# @app.get("/admin/monitoring", response_model=StandardResponse[FullSystemMonitor], tags=["Admin"])
# async def get_full_system_monitoring():
#     """
#     Aggregates:
#     1. Local System (CPU/RAM)
#     2. Model Server (GPU Stats + Embedder/OCR/Layout Status)
#     3. Local LLM Stats (Ollama Status via get_llm_stats)
#     """
    
#     # 1. Get Local Metrics
#     current_status = {
#         "cpu_percent": psutil.cpu_percent(interval=None),
#         "memory_percent": psutil.virtual_memory().percent,
#         "disk_usage": psutil.disk_usage('/').percent,
#         "network_connections": len(psutil.net_connections()),
#         "model_server": None
#     }

#     # ========================================================================
#     # NEW: Updated to pick first server from settings list for monitoring
#     # ========================================================================
#     if hasattr(settings, "model_server_urls_list") and settings.model_server_urls_list:
#         model_server_url = settings.model_server_urls_list[0]
#     else:
#         # Fallback to env var or default
#         model_server_url = os.getenv("MODEL_SERVER_URL", "http://localhost:8074")
    
#     try:
#         # 2. Call Model Server (Port 8074 or first configured)
#         resp = await http_client.get(f"{model_server_url}/monitor")
        
#         if resp.status_code == 200:
#             remote_data = resp.json()
            
#             # --- MERGE LOGIC START ---
#             if "models" in remote_data and isinstance(remote_data["models"], dict):
#                 # Fetch fresh LLM stats using the getter function
#                 local_llm_data = get_llm_stats()
                
#                 # Merge {"llm": ...} into the remote models list
#                 remote_data["models"].update(local_llm_data)
#             # --- MERGE LOGIC END ---
            
#             current_status["model_server"] = remote_data
#         else:
#             logger.warning(f"Model server returned status {resp.status_code}")
            
#     except Exception as e:
#         # Log specific error for debugging
#         logger.error(f"Model Server Connection Failed: {type(e).__name__} - {e}")
#         # Return partial data (system stats only) if model server fails
        
#     return StandardResponse(
#         status="success",
#         status_code=200,
#         message="System monitoring data retrieved.",
#         data=current_status
#     )








