# services/rate_limit_service.py
import asyncio
from datetime import datetime, timedelta
from typing import Tuple

from pymongo import ReturnDocument
from pymongo.errors import PyMongoError

from app.config import settings
from app.database import db

# Mongo collections
UPLOADS_COLL = db.get_collection("rate_limits_uploads")
CHAT_COLL = db.get_collection("rate_limits_chat")

UPLOAD_LIMIT = settings.RATE_LIMIT_UPLOADS_PER_HOUR
CHAT_LIMIT = settings.RATE_LIMIT_CHAT_PER_MINUTE   # e.g., 2 per minute
CHAT_CAPACITY = settings.RATE_LIMIT_CHAT_PER_MINUTE



# ============================================================
# UPLOADS: FIXED WINDOW — PER HOUR
# ============================================================
async def try_consume_uploads(user_id: str, count: int) -> Tuple[bool, int, int]:
    now = datetime.utcnow()

    window_start = now.replace(minute=0, second=0, microsecond=0)
    window_end = window_start + timedelta(hours=1)
    seconds_to_reset = int((window_end - now).total_seconds())

    window_key = f"{user_id}:{window_start.strftime('%Y%m%d%H')}"

    try:
        filter_doc = {
            "key": window_key,
            "$or": [
                {"count": {"$lt": UPLOAD_LIMIT}},
                {"count": {"$exists": False}},
            ],
        }

        update_doc = {
            "$inc": {"count": count},
            "$setOnInsert": {
                "key": window_key,
                "user_id": user_id,
                "window_start": window_start,
                "expires_at": window_end + timedelta(seconds=10),
            },
        }

        doc = await UPLOADS_COLL.find_one_and_update(
            filter_doc,
            update_doc,
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

        if doc is None:
            existing = await UPLOADS_COLL.find_one({"key": window_key})
            current_count = existing["count"] if existing else 0
            remaining = max(0, UPLOAD_LIMIT - current_count)
            return False, remaining, seconds_to_reset

        new_count = doc.get("count", 0)
        remaining = max(0, UPLOAD_LIMIT - new_count)

        return True, remaining, seconds_to_reset

    except PyMongoError:
        return False, 0, seconds_to_reset


# ============================================================
# CHAT: FIXED WINDOW — PER MINUTE
# ============================================================
# ===========================
# CHAT — FIXED WINDOW PER MINUTE (FINAL WORKING VERSION)
# ===========================
# ===========================
# CHAT — Sliding Window per user
# ===========================
async def try_consume_chat_token(user_id: str, count: int = 1):
    """
    Sliding-window chat rate limiting.
    Limits: CHAT_CAPACITY requests inside a 60-second window.
    Window starts at the user's first request.
    """
    now = datetime.utcnow()
    window_seconds = 60

    try:
        doc = await CHAT_COLL.find_one({"user_id": user_id})

        # -----------------------------------------------------
        # FIRST REQUEST EVER → CREATE NEW WINDOW
        # -----------------------------------------------------
        if not doc:
            new_doc = {
                "user_id": user_id,
                "window_start": now,
                "count": count,
                "expires_at": now + timedelta(seconds=window_seconds + 5),  # buffer
            }
            await CHAT_COLL.insert_one(new_doc)
            remaining = CHAT_CAPACITY - count
            return True, remaining, 0

        window_start = doc["window_start"]
        old_count = doc.get("count", 0)

        # -----------------------------------------------------
        # WINDOW EXPIRED → RESET & ACCEPT
        # -----------------------------------------------------
        if (now - window_start).total_seconds() >= window_seconds:
            # Reset window
            new_values = {
                "window_start": now,
                "count": count,
                "expires_at": now + timedelta(seconds=window_seconds + 5),
            }
            await CHAT_COLL.update_one(
                {"user_id": user_id},
                {"$set": new_values}
            )
            remaining = CHAT_CAPACITY - count
            return True, remaining, 0

        # -----------------------------------------------------
        # WINDOW ACTIVE → CHECK LIMIT
        # -----------------------------------------------------
        new_count = old_count + count

        if new_count > CHAT_CAPACITY:
            # Too many requests
            retry_after = window_seconds - int((now - window_start).total_seconds())
            remaining = max(0, CHAT_CAPACITY - old_count)
            return False, remaining, retry_after

        # -----------------------------------------------------
        # ACCEPT WITHIN WINDOW → INCREMENT COUNTER
        # -----------------------------------------------------
        await CHAT_COLL.update_one(
            {"user_id": user_id},
            {"$set": {
                "count": new_count,
                "expires_at": window_start + timedelta(seconds=window_seconds + 5),
            }}
        )

        remaining = CHAT_CAPACITY - new_count
        return True, remaining, 0

    except Exception:
        # Safe fallback — deny request if DB error
        return False, 0, 1

