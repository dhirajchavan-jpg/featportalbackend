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

# --- MongoDB Connection Setup with Connection Pooling ---
try:
    # Connection details
    MONGO_URI = os.getenv("MONGO_URI", settings.MONGO_URI)
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", settings.MONGO_DB_NAME)

    # --- Connection Pooling Configuration ---
    pooling_options = {
        'maxPoolSize': 50,           # Maximum connections per server (default: 100)
        'minPoolSize': 10,           # Minimum connections to maintain (default: 0)
        'maxIdleTimeMS': 30000,      # Max time connection can be idle (30 seconds)
        'waitQueueTimeoutMS': 5000,  # Max time to wait for available connection
        'connectTimeoutMS': 10000,   # Connection establishment timeout (10 seconds)
        'socketTimeoutMS': 20000,    # Socket operation timeout (20 seconds)
        'serverSelectionTimeoutMS': 5000,  # Server selection timeout
        'retryWrites': True,         # Retry write operations on network errors
        'retryReads': True           # Retry read operations on network errors
    }

    # Create Async client with pooling options
    client = AsyncIOMotorClient(MONGO_URI, **pooling_options)

    # Test connection (async-compatible check)
    # Note: list_database_names() should be awaited in an async context
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
    logger.info(f" Could not connect to MongoDB: {e}")  # <--- Added f-string




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

        logger.info(" MongoDB indexes created successfully!")

    except Exception as e:
        logging.error(f" Index creation failed: {e}")  # <--- Added f-string


# --- Connection Health Check (Optional) ---
async def verify_connection():
    """
    Verify MongoDB connection is working.
    Call this at startup to ensure connectivity.
    """
    try:
        # Trigger a simple database operation
        await client.admin.command('ping')
        logging.info(" Connected to MongoDB successfully!")
        
        # Print connection pool stats
        logging.info(f"Pool Config: minPoolSize={pooling_options['minPoolSize']}, "
                     f"maxPoolSize={pooling_options['maxPoolSize']}")
    except Exception as e:
        logging.error(f" MongoDB connection failed: {e}")
        raise


# --- Graceful Shutdown ---
async def close_mongo_connection():
    """
    Close MongoDB connection pool gracefully.
    Call this during application shutdown.
    """
    client.close()
    logging.info(" MongoDB connection closed")
