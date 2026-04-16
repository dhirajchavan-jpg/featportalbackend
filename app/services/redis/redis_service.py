# app/services/redis/redis_service.py
import json
import uuid
import asyncio
import redis.asyncio as redis
from typing import Optional, Dict, Any
from app.config import settings
from app.utils.logger import setup_logger
# Initialize logger
logger = setup_logger()
class RedisService:
    def __init__(self):
        # Construct Redis URL
        self.redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
        logger.info(f"[RedisService] Initializing Redis connection to: {self.redis_url}")
        try:
            # --- MODIFIED: Added stability options for Windows ---
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                # Fix for WinError 64: Keep TCP connection alive
                socket_keepalive=True,
                # Fix for Stale Connections: Auto-ping every 30s
                health_check_interval=30,
                socket_connect_timeout=10,
                retry_on_timeout=True
            )
            logger.info("[RedisService] Redis client created successfully.")
        except Exception as e:
            logger.critical(f"[RedisService] Failed to create Redis client: {e}", exc_info=True)
            raise e
    # --- MODIFIED: Generic Enqueue Method with Retry Logic ---
    async def enqueue_job(self, job_type: str, job_data: Dict[str, Any], user_data: Dict[str, Any]) -> str:
        """
        Pushes a generic job to the Redis List.
        """
        task_id = str(uuid.uuid4())
        logger.info(f"[RedisService] Enqueueing Job | ID: {task_id} | Type: {job_type} | User: {user_data.get('user_id')}")
        job_payload = {
            "task_id": task_id,
            "status": "queued",
            "job_type": job_type,  # Stores the type of job
            "job_data": job_data,  # Generic payload
            "user_data": user_data,
            "created_at": asyncio.get_event_loop().time()
        }
        # --- RETRY LOOP FOR WINDOWS STABILITY ---
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 1. Push actual job to the Queue
                logger.debug(f"[RedisService] Pushing payload to list: {settings.REDIS_QUEUE_NAME}")
                await self.client.lpush(settings.REDIS_QUEUE_NAME, json.dumps(job_payload))
                # 2. Set initial status key (for fast lookup by frontend)
                status_key = f"job:{task_id}"
                logger.debug(f"[RedisService] Setting initial status key: {status_key}")
                await self.client.setex(
                    status_key,
                    settings.REDIS_RESULT_TTL,
                    json.dumps({
                        "status": "queued",
                        "task_id": task_id,
                        "job_type": job_type
                    })
                )
                logger.info(f"[RedisService] Job {task_id} successfully enqueued.")
                return task_id
            except (redis.ConnectionError, OSError) as e:
                # Catch WinError 64 and retry
                logger.warning(f"[RedisService] Enqueue attempt {attempt+1}/{max_retries} failed: {e}. Retrying...")
                if attempt == max_retries - 1:
                    logger.error(f"[RedisService] All retry attempts failed for job {task_id}", exc_info=True)
                    raise e
                await asyncio.sleep(0.5)
            except Exception as e:
                # Non-connection errors (e.g., serialization) fail immediately
                logger.error(f"[RedisService] Error enqueueing job {task_id}: {e}", exc_info=True)
                raise e
    async def get_job_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the status or result of a job.
        """
        key = f"job:{task_id}"
        logger.debug(f"[RedisService] Fetching status for key: {key}")
        try:
            data = await self.client.get(key)
            if data:
                logger.debug(f"[RedisService] Data found for {task_id}")
                return json.loads(data)
            else:
                logger.warning(f"[RedisService] No data found for {task_id} (Expired or Invalid)")
                return None
        except Exception as e:
            logger.error(f"[RedisService] Error fetching status for {task_id}: {e}", exc_info=True)
            return None
    async def update_job_result(self, task_id: str, result: Dict[str, Any], status: str = "completed"):
        """
        Called by the Worker to save the final answer.
        """
        key = f"job:{task_id}"
        logger.info(f"[RedisService] Updating Job {task_id} | Status: {status}")
        try:
            payload = {
                "status": status,
                "task_id": task_id,
                "data": result,
                "completed_at": asyncio.get_event_loop().time()
            }
            # Save result with TTL
            await self.client.setex(
                key,
                settings.REDIS_RESULT_TTL,
                json.dumps(payload)
            )
            logger.info(f"[RedisService] Job {task_id} result saved successfully.")
        except Exception as e:
            logger.error(f"[RedisService] Failed to update result for {task_id}: {e}", exc_info=True)
    # --- NEW: CANCELLATION METHODS ---
    async def cancel_job(self, task_id: str):
        """
        Sets a specific 'cancelled' flag for the job.
        Used by API Router when user cancels upload.
        """
        key = f"job:{task_id}:cancel"
        try:
            # Set a flag that expires in 1 hour (plenty of time for worker to see it)
            await self.client.setex(key, 3600, "true")
            logger.info(f"[RedisService] Job {task_id} marked for cancellation in Redis.")
        except Exception as e:
            logger.error(f"[RedisService] Failed to mark job {task_id} as cancelled: {e}")
    async def is_job_cancelled(self, task_id: str) -> bool:
        """
        Checks if the cancellation flag exists.
        Used by Worker before finalizing job.
        """
        key = f"job:{task_id}:cancel"
        try:
            value = await self.client.get(key)
            return value == "true"
        except Exception as e:
            logger.error(f"[RedisService] Failed to check cancellation status for {task_id}: {e}")
            return False
# Global Instance
redis_service = RedisService()