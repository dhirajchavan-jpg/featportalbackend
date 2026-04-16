from datetime import timedelta
import time
import logging
from datetime import datetime, timedelta
from typing import List
from qdrant_client import QdrantClient, models
from app.config import settings

logger = logging.getLogger(__name__)

# --- GLOBAL SECTOR CACHE ---
_SECTOR_CACHE = {
    "data": [],
    "last_updated": None
}
SECTOR_CACHE_TTL = timedelta(minutes=10)

def get_available_sectors() -> List[str]:
    """Get list of all available global sectors from Qdrant with caching."""
    global _SECTOR_CACHE
    
    now = datetime.utcnow()
    if _SECTOR_CACHE["data"] and _SECTOR_CACHE["last_updated"]:
        if now - _SECTOR_CACHE["last_updated"] < SECTOR_CACHE_TTL:
            logger.info(f"[AvailableSectors] Serving {len(_SECTOR_CACHE['data'])} sectors from cache")
            return _SECTOR_CACHE["data"]

    
    
    try:
        logger.info("[AvailableSectors] Cache expired. Scanning Qdrant for sectors...")
        client = QdrantClient(url=settings.QDRANT_URL)
        
        sectors = set()
        offset = None
        
        while True:
            results, offset = client.scroll(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.is_global",
                            match=models.MatchValue(value=True)
                        )
                    ]
                ),
                limit=5000,
                offset=offset,
                with_payload=["metadata.source", "metadata.sector"],
                with_vectors=False
            )
            
            for point in results:
                meta = point.payload.get('metadata', {})
                sector = meta.get('source') or meta.get('sector')
                if sector:
                    sectors.add(sector)
            
            if offset is None:
                break
        
        sector_list = sorted(list(sectors))
        
        _SECTOR_CACHE["data"] = sector_list
        _SECTOR_CACHE["last_updated"] = now
        
        logger.info(f"[AvailableSectors] Refresh complete. Found {len(sector_list)} sectors.")
        return sector_list
    
    except Exception as e:
        logger.error(f"[AvailableSectors] Error: {e}")
        return _SECTOR_CACHE.get("data", [])