import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from opentelemetry import trace
from app.config import settings
from app.database import chat_history_collection, cache_collection

# Set up tracing
tracer = trace.get_tracer(__name__)

# Set up logging
logger = logging.getLogger(__name__)


async def _get_chat_history(chat_id: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Retrieve recent chat history."""
    history_cursor = chat_history_collection.find(
        {"chat_id": chat_id, "message_type": "text"},
        projection={"user_query": 1, "llm_answer": 1, "sector": 1, "_id": 0}
    ).sort("created_at", -1).limit(limit)
    
    history_list = await history_cursor.to_list(length=limit)
    history_list.reverse()
    return history_list


async def _save_to_history(
    chat_id: str,
    user_id: str,
    query: str,
    project_id: str,
    sectors: Optional[List[str]],
    answer: str,
    source_documents: list = None,   # Existing
    retrieval_stats: dict = None,    # <--- NEW
    meta_data: dict = None,          # <--- NEW
    style: str = None                # <--- ADDED: Style parameter
):
    """Save query and response to chat history."""

    
    # 1. Clean source_documents to ensure they are MongoDB-safe dicts
    clean_docs = []
    if source_documents:
        for doc in source_documents:
            # If it's already a dict, use it; otherwise convert object attributes
            if isinstance(doc, dict):
                clean_docs.append(doc)
            else:
                # Fallback for objects (like LangChain Documents)
                clean_docs.append({
                    "page_content": getattr(doc, "page_content", ""),
                    "metadata": getattr(doc, "metadata", {}),
                    "relevance_score": getattr(doc, "relevance_score", None),
                    "rerank_score": getattr(doc, "rerank_score", None)
                })
    history_entry = {
        "chat_id": chat_id,
        "user_id": user_id,
        "project_id": project_id,
        "sectors": sectors,
        "sector": sectors[0] if sectors else None,
        "message_type": "text",
        "user_query": query,
        "llm_answer": answer,
        "created_at": datetime.utcnow(),
        "source_documents": clean_docs,      
        "retrieval_stats": retrieval_stats,  
        "meta": meta_data,
        "style": style,                      # <--- ADDED: Store style in history
        "file_id": None,
        "file_name": None
    }
    try:
        await chat_history_collection.insert_one(history_entry)
        logger.info(f"[HISTORY] Saved detailed interaction for chat_id: {chat_id}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to save chat history: {e}")


async def _save_file_upload_to_history(
    chat_id: str,
    user_id: str,
    project_id: str,
    sector: str,
    file_id: str,
    file_name: str
):
    """Save file upload event to chat history."""
    history_entry = {
        "chat_id": chat_id,
        "user_id": user_id,
        "project_id": project_id,
        "sector": sector,
        "message_type": "file",
        "file_id": file_id,
        "file_name": file_name,
        "created_at": datetime.utcnow(),
        "user_query": None,
        "llm_answer": None
    }
    try:
        await chat_history_collection.insert_one(history_entry)
    except Exception as e:
        logger.error(f"[ERROR] Failed to save file upload history: {e}")


def _build_cache_key(
    user_id: str, 
    project_id: str, 
    query: str, 
    sectors: Optional[List[str]],
    excluded_files: Optional[List[str]],
    style: str = "Detailed"          # <--- ADDED: Style parameter with default
) -> str:
    """Build cache key including all sources, exclusions, and style."""
    normalized_query = query.lower().strip()
    sectors_str = ",".join(sorted(sectors)) if sectors else "none"
    exclusions_str = ",".join(sorted(excluded_files)) if excluded_files else "none"
    
    # <--- UPDATED: Added style to the unique key string
    return f"user_{user_id}|project_{project_id}|sectors_{sectors_str}|excl_{exclusions_str}|style_{style}|query_{normalized_query}"


async def _cache_result(
    cache_key: str,
    user_id: str,
    query: str,
    project_id: str,
    sectors: Optional[List[str]],
    answer: str,
    style: str = None                # <--- ADDED: Style parameter
):
    """Cache query result."""
    new_cache_entry = {
        "cache_key": cache_key,
        "user_id": user_id,
        "user_query": query,
        "project_id": project_id,
        "sectors": sectors,
        "llm_answer": answer,
        "style": style,              # <--- ADDED: Store style in cache
        "created_at": datetime.utcnow()
    }
    try:
        await cache_collection.insert_one(new_cache_entry)
    except Exception as e:
        logger.error(f"[ERROR] Failed to cache result: {e}")