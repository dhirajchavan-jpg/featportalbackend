import os
import logging.config

# 1. Define Directory Paths
BASE_LOG_DIR = "logs"
SYSTEM_LOG_DIR = os.path.join(BASE_LOG_DIR, "system")
AI_LOG_DIR = os.path.join(BASE_LOG_DIR, "ai")

# 2. Create directories if they don't exist
os.makedirs(SYSTEM_LOG_DIR, exist_ok=True)
os.makedirs(AI_LOG_DIR, exist_ok=True)

# 3. Define File Paths
SYSTEM_LOG_FILE = os.path.join(SYSTEM_LOG_DIR, "system.log")
RAG_LOG_FILE = os.path.join(AI_LOG_DIR, "ai_rag.log")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "verbose": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        # System Handler -> logs/system/system.log
        "system_file": {
            "class": "logging.FileHandler",
            "filename": SYSTEM_LOG_FILE,
            "formatter": "standard",
            "level": "INFO",
            "encoding": "utf-8",
        },
        # AI Handler -> logs/ai/ai_rag.log
        "rag_file": {
            "class": "logging.FileHandler",
            "filename": RAG_LOG_FILE,
            "formatter": "verbose",
            "level": "INFO",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        # --- ROOT LOGGER ---
        # Catches Uvicorn, Database, Main App, etc.
        "": {
            "handlers": ["console", "system_file"],
            "level": "INFO",
        },
        
        # --- AI & RAG LOGGERS ---
        # 1. Capture EVERYTHING inside the 'app/services/' folder
        # This covers: rag, retrieval, context_manager, embedding, etc.
        "app.services": { 
            "handlers": ["console", "rag_file"], 
            "level": "INFO", 
            "propagate": False 
        },

        # 2. Capture specific 'app/core/llm_provider' module
        "app.core.llm_provider": { 
            "handlers": ["console", "rag_file"], 
            "level": "INFO", 
            "propagate": False 
        },

        # 3. Keep Prompt Validation in AI logs (optional)
        "app.middleware.prompt_validation": { 
            "handlers": ["console", "rag_file"], 
            "level": "INFO", 
            "propagate": False 
        },
    }
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)