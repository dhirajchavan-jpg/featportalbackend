# app/services/redis/worker.py
import asyncio
import json
import os
import signal
import sys
import time
from typing import Callable, Any
from bson import ObjectId # <--- NEEDED FOR CLEANUP
import logging
from logging.handlers import RotatingFileHandler



# --- PATH SETUP ---
# Ensure we can import 'app' modules
sys.path.append(os.getcwd())

# --- IMPORTS ---
from app.config import settings
from app.utils.logger import setup_logger
from app.services.redis.redis_service import redis_service
from app.services.rag.pipeline_orchestrator import query_rag_pipeline
# --- NEW IMPORTS: File Indexing Functions ---
from app.services.rag.file_indexing import process_and_index_file, process_and_index_global_file
from app.dependencies import UserPayload
# --- ADDED COLLECTIONS FOR ROLLBACK ---
from app.database import (
    verify_connection,
    close_mongo_connection,
    create_indexes,
    file_collection,
    Global_file_collection,
    project_collection
)
from app.core.llm_provider import load_models
from app.services.rag.evaluation_runner import _run_comprehensive_evaluation_background
from app.models.Files import FileModel  # <--- ADDED: To create DB record

# Initialize Logger
logger = setup_logger()
try:
    # 1. Ensure the 'logs/ai' directory exists
    log_dir = os.path.join(os.getcwd(), "logs", "ai")
    os.makedirs(log_dir, exist_ok=True)
    
    # 2. Point to the shared log file
    shared_log_path = os.path.join(log_dir, "ai_rag.log")

    # 3. Add Handler if not present
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        file_handler = RotatingFileHandler(
            shared_log_path, 
            maxBytes=10*1024*1024, # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        # Add [WORKER] prefix so you can distinguish them in Excel
        formatter = logging.Formatter('%(asctime)s - [WORKER] - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        logger.info(f" [Worker] Logging initialized. Merging logs into: {shared_log_path}")

except Exception as e:
    logger.error(f"Failed to setup worker file logging: {e}")

# Global Control Flags
RUNNING = True
PROCESSING_ACTIVE = False  # Prevents hard kills during a job

# --- HELPER: Mock Background Tasks ---
class DummyBackgroundTasks:
    """
    Mocks FastAPI's BackgroundTasks.
    """
    def add_task(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        # We assume forced execution via pipeline flag now, so this remains for compatibility
        pass

async def process_job(job_json: str):
    """
    Decodes the job, determines its type (query vs upload), runs the appropriate logic, and saves the result.
    """
    global PROCESSING_ACTIVE
    PROCESSING_ACTIVE = True  # Mark start of processing
    
    try:
        # 1. Parse Job
        job = json.loads(job_json)
        task_id = job.get("task_id", "unknown")
        
        # --- DETERMINE JOB TYPE & PAYLOAD ---
        # Default to "query" if missing (backward compatibility)
        job_type = job.get("job_type", "query") 
        
        # 'job_data' holds the generic payload (was 'query_data')
        job_data = job.get("job_data", job.get("query_data", {}))
        user_data = job.get("user_data", {})
        
        logger.info(f"\n{'='*60}")
        logger.info(f" [Worker] START PROCESSING Job: {task_id} | Type: {job_type}")
        logger.info(f" [Worker] User: {user_data.get('email')}")
        logger.info(f"{'='*60}")

        # 2. Reconstruct User Payload
        # Used for permission checks and metadata tagging
        current_user = UserPayload(
            user_id=user_data.get("user_id"),
            email=user_data.get("email"),
            role=user_data.get("role", "user")
        )

        # 3. Update Status to Processing
        logger.debug(f" [Worker] Updating Redis status to 'processing'...")
        await redis_service.update_job_result(task_id, {}, status="processing")
        
        start_time = time.time()

        # =========================================================================
        # CASE A: QUERY (RAG Chat)
        # =========================================================================
        if job_type == "query":
            logger.info(f" [Worker] Invoking query_rag_pipeline...")
            
            # Use Dummy Tasks
            dummy_bg = DummyBackgroundTasks()

            #  CALL PIPELINE WITH FORCED EVALUATION 
            result = await query_rag_pipeline(
                query=job_data["query"],
                current_user=current_user,
                project_id=job_data["project_id"],
                sectors=[job_data.get("sector")] if job_data.get("sector") else None,
                excluded_files=job_data.get("excluded_files"),
                style=job_data.get("style") or "Detailed", 
                ai_config=job_data.get("ai_config"), 
                background_tasks=dummy_bg,
                skip_evaluation=True # Worker handles eval separately below
            )

            eval_data = result.pop("_eval_data", None)
            
            duration = round(time.time() - start_time, 2)
            logger.info(f" [Worker] Pipeline finished in {duration}s")

            # Format Result for Frontend
            response_data = {
                "result": result.get("result", ""),
                "source_documents": result.get("source_documents", []),
                "retrieval_stats": result.get("retrieval_stats", {}),
                "sources_used": result.get("sources_used", {}),
                "is_comparative": result.get("is_comparative", False),
                "processing_time": duration
            }

            # Save Success to Redis
            await redis_service.update_job_result(task_id, response_data, status="completed")
            logger.info(f" [Worker] Job {task_id} COMPLETED successfully.")
            
            # Run Evaluation
            if eval_data:
                logger.info(f" [Worker] Starting Post-Response Evaluation...")
                try:
                    await _run_comprehensive_evaluation_background(eval_data)
                    logger.info(f" [Worker] Evaluation Finished.")
                except Exception as e:
                    logger.error(f" [Worker] Evaluation Failed: {e}")

        # =========================================================================
        # CASE B: FILE UPLOAD (Project File)
        # =========================================================================
        elif job_type == "file_upload":
            logger.info(f" [Worker] Processing File Upload: {job_data.get('original_filename')}")

            try:
                # Trust only backend-derived project sector by re-reading project doc.
                project_obj = await project_collection.find_one({"_id": ObjectId(job_data["project_id"])})
                if not project_obj:
                    raise RuntimeError("Project not found for queued upload job")

                resolved_sector = (project_obj.get("organization_sector") or "").strip().upper()
                if not resolved_sector:
                    raise RuntimeError("Project organization sector is missing")

                if job_data.get("sector") and job_data.get("sector").strip().upper() != resolved_sector:
                    logger.warning(" [Worker] Job sector mismatch detected. Overriding with project organization sector.")

                result = await asyncio.to_thread(
                    process_and_index_file,
                    file_path=job_data["file_path"],
                    project_id=job_data["project_id"],
                    sector=resolved_sector,
                    current_user=current_user,
                    doc_type=job_data.get("doc_type", "general"),
                    original_filename=job_data.get("original_filename"),
                    file_id=job_data.get("file_id"),
                    ocr_engine=job_data.get("ocr_engine", "paddleocr")
                )

                duration = round(time.time() - start_time, 2)
                result["processing_time"] = duration

                if await redis_service.is_job_cancelled(task_id):
                    logger.warning(f" [Worker] Job {task_id} CANCELLED by user. Aborting DB Save.")
                    await redis_service.update_job_result(task_id, {"error": "Cancelled by user"}, status="cancelled")
                    return

                file_obj_id = ObjectId(job_data["file_id"])
                file_doc = FileModel(
                    project_id=job_data["project_id"],
                    filename=job_data["original_filename"],
                    file_url=job_data["file_path"],
                    file_hash=job_data.get("file_hash"),
                    sector=resolved_sector,
                    category=job_data.get("category", "General"),
                    compliance_type=job_data.get("compliance_type", "General"),
                    user_id=current_user.user_id
                )

                doc_dict = file_doc.model_dump(exclude={"file_id"})
                doc_dict["_id"] = file_obj_id
                await file_collection.insert_one(doc_dict)

                await redis_service.update_job_result(task_id, result, status="completed")
                logger.info(f" [Worker] File Indexing Job {task_id} COMPLETED.")

            except Exception as e:
                logger.error(f" [Worker] File indexing failed: {e}")
                raise e

        elif job_type == "global_file_upload":
            logger.info(f" [Worker] Processing Global File Upload: {job_data.get('original_filename')}")

            try:
                result = await asyncio.to_thread(
                    process_and_index_global_file,
                    file_path=job_data["file_path"],
                    sector=job_data["sector"],
                    file_id=job_data.get("file_id"),
                    original_filename=job_data.get("original_filename"),
                    extra_metadata=job_data.get("extra_metadata", {}),
                    ocr_engine=job_data.get("ocr_engine", "paddleocr")
                )

                duration = round(time.time() - start_time, 2)
                result["processing_time"] = duration

                if await redis_service.is_job_cancelled(task_id):
                    logger.warning(f" [Worker] Global job {task_id} CANCELLED by user.")
                    await redis_service.update_job_result(task_id, {"error": "Cancelled by user"}, status="cancelled")
                    return

                await redis_service.update_job_result(task_id, result, status="completed")
                logger.info(f" [Worker] Global File Job {task_id} COMPLETED.")

            except Exception as e:
                logger.error(f" [Worker] Global indexing failed: {e}")

                # Rollback global metadata if indexing failed
                try:
                    if job_data.get("file_id"):
                        await Global_file_collection.delete_one({"file_id": job_data.get("file_id")})
                except Exception:
                    pass

                # Cleanup physical file on failure
                try:
                    file_path = job_data.get("file_path")
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                except Exception:
                    pass

                raise e

        # =========================================================================
        # UNKNOWN TYPE
        # =========================================================================
        else:
            logger.error(f" [Worker] Unknown job type: {job_type}")
            await redis_service.update_job_result(task_id, {"error": f"Unknown job type: {job_type}"}, status="failed")

    except Exception as e:
        logger.error(f" [Worker] CRITICAL ERROR processing job {task_id if 'task_id' in locals() else 'unknown'}: {e}", exc_info=True)
        if 'task_id' in locals():
            await redis_service.update_job_result(task_id, {"error": str(e)}, status="failed")
            
    finally:
        PROCESSING_ACTIVE = False # Mark end of processing

async def worker_loop():
    logger.info(f" [Worker] >>> STARTING WORKER PROCESS (Reliable Queue Mode) <<<")
    
    # 1. Initialize DB
    logger.info(" [Worker] Initializing MongoDB connection...")
    await verify_connection()
    await create_indexes()
    
    # 2. Initialize Models (LLM, Qdrant)
    logger.info(" [Worker] Loading Models and Clients...")
    load_models()
    
    # Define Queues for Reliable Pattern
    QUEUE_MAIN = settings.REDIS_QUEUE_NAME
    QUEUE_PROCESSING = f"{settings.REDIS_QUEUE_NAME}:processing"
    
    logger.info(f" [Worker] Listening on {QUEUE_MAIN} (Backup: {QUEUE_PROCESSING})")

    # 1. RECOVERY STEP
    while True:
        try:
            stranded_job = await redis_service.client.rpop(QUEUE_PROCESSING)
            if not stranded_job:
                break
            logger.warning(" [Worker] Found stranded job from previous crash. Reprocessing...")
            await process_job(stranded_job)
        except Exception as e:
            logger.error(f" [Worker] Error during recovery: {e}")
            break

    # 2. MAIN LOOP
    while RUNNING or PROCESSING_ACTIVE:
        
        # Graceful Shutdown Logic
        if not RUNNING and PROCESSING_ACTIVE:
            logger.info(" [Worker] Shutdown requested. Waiting for current job to finish...")
            await asyncio.sleep(1)
            continue
        elif not RUNNING:
            break

        try:
            # SAFE FETCH: Move from Main -> Processing atomically
            job_json = await redis_service.client.brpoplpush(QUEUE_MAIN, QUEUE_PROCESSING, timeout=2)
            
            if job_json:
                await process_job(job_json)
                await redis_service.client.lrem(QUEUE_PROCESSING, 1, job_json)
            
        except Exception as e:
             if RUNNING:
                logger.error(f" [Worker] Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(1)

    # Cleanup
    logger.info(" [Worker] Shutting down resources...")
    await close_mongo_connection()
    logger.info(" [Worker] Shutdown complete. Bye!")

def signal_handler(sig, frame):
    global RUNNING
    logger.warning(" [Worker] Signal received. Stopping worker loop...")
    RUNNING = False

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    asyncio.run(worker_loop())





