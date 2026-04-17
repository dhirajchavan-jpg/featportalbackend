# app/database.py

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from dotenv import load_dotenv
import os
from app.config import settings

import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize defaults so import-time failures do not crash the module
client = None
db = None
pooling_options = {
    'maxPoolSize': 50,
    'minPoolSize': 10,
    'maxIdleTimeMS': 30000,
    'waitQueueTimeoutMS': 5000,
    'connectTimeoutMS': 10000,
    'socketTimeoutMS': 20000,
    'serverSelectionTimeoutMS': 5000,
    'retryWrites': True,
    'retryReads': True,
}

# --- MongoDB Connection Setup with Connection Pooling ---
try:
    # Connection details
    MONGO_URI = os.getenv("MONGO_URI", settings.MONGO_URI)
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", settings.MONGO_DB_NAME)

    # Create Async client with pooling options
    client = AsyncIOMotorClient(MONGO_URI, **pooling_options)

    logging.info(" MongoDB client initialized with connection pooling!")

    # Select database
    db = client[MONGO_DB_NAME]

    chat_history_collection = db["chat_history"]

    # Collections are now Async
    cache_collection = db.get_collection(
        getattr(settings, "MONGO_COLLECTION_NAME", "cache")
    )
    User_collection = db["users"]
    project_collection = db["projects"]
    file_collection = db["files"]
    Global_file_collection = db["global_files"]
    project_config_collection = db["project_configs"]
    user_file_selection_collection = db["user_file_selection"]
    super_admin_collection = db["super_admins"]
    db.user_file_selection = user_file_selection_collection

except Exception as e:
    logger.info(f" Could not initialize MongoDB client: {e}")


def get_database():
    """
    Returns the AsyncIOMotorDatabase instance.
    Used by routers/services to access collections dynamically.
    """
    return db


# --- Index Creation (async function) ---
async def create_indexes():
    """
    Create indexes to improve performance and ensure data uniqueness.
    """
    if db is None:
        logging.warning(" MongoDB not initialized. Skipping index creation.")
        return

    try:
        # Rate limiting indexes
        await db.rate_limits_uploads.create_index("expires_at", expireAfterSeconds=0)
        await db.rate_limits_uploads.create_index("key", unique=True)
        await db.rate_limits_chat.create_index("user_id", unique=True)

        # Performance Cache Index
        await cache_collection.create_index(
            [("cache_key", ASCENDING)],
            unique=True,
            name="cache_key_unique_index"
        )

        # Project Index
        await project_collection.create_index(
            [("user_id", ASCENDING), ("project_name", ASCENDING)],
            unique=True,
            name="unique_user_project_name"
        )

        # Chat History Index
        await chat_history_collection.create_index(
            [
                ("chat_id", ASCENDING),
                ("created_at", DESCENDING)
            ],
            name="chat_history_index"
        )

        # Project Config Index
        await project_config_collection.create_index(
            [("project_id", ASCENDING)],
            unique=True,
            name="project_config_unique_index"
        )

        # File metadata indexes
        await file_collection.create_index(
            [("project_id", ASCENDING), ("filename", ASCENDING)],
            name="project_filename_index"
        )
        await file_collection.create_index(
            [("project_id", ASCENDING), ("file_hash", ASCENDING)],
            name="project_file_hash_index"
        )

        # Global file indexes
        await Global_file_collection.create_index(
            [("file_hash", ASCENDING)],
            unique=True,
            name="global_file_hash_unique_index"
        )
        await Global_file_collection.create_index(
            [("filename", ASCENDING), ("created_at", DESCENDING)],
            name="global_filename_created_at_index"
        )

        # Hidden selection index
        await user_file_selection_collection.create_index(
            [("user_id", ASCENDING), ("project_id", ASCENDING), ("file_id", ASCENDING)],
            name="user_project_file_selection_index"
        )

        logger.info(" MongoDB indexes created successfully!")

    except Exception as e:
        logging.error(f" Index creation failed: {e}")


# --- Connection Health Check (Optional) ---
async def verify_connection() -> bool:
    """
    Verify MongoDB connection is working.
    Returns True if connected, else False. Does not raise.
    """
    if client is None:
        logging.warning(" MongoDB client is not initialized.")
        return False

    try:
        # Trigger a simple database operation
        await client.admin.command('ping')
        logging.info(" Connected to MongoDB successfully!")

        # Print connection pool stats
        logging.info(
            f"Pool Config: minPoolSize={pooling_options['minPoolSize']}, "
            f"maxPoolSize={pooling_options['maxPoolSize']}"
        )
        return True
    except Exception as e:
        logging.error(f" MongoDB connection failed: {e}")
        return False


# --- Graceful Shutdown ---
async def close_mongo_connection():
    """
    Close MongoDB connection pool gracefully.
    Call this during application shutdown.
    """
    if client is None:
        return
    client.close()
    logging.info(" MongoDB connection closed")

