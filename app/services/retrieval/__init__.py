# app/services/retrieval/__init__.py
"""
Layer 4: Advanced Retrieval System

Multi-source retrieval with file-level blacklisting:
- Unified source handling (Project IDs + Sectors treated identically)
- File-level exclusion support
- Hybrid search (BM25 + Dense Vector)
- RRF Fusion
- Cross-encoder Reranking
"""
import logging

# Configure logger for this package initialization
logger = logging.getLogger(__name__)

logger.info("[RetrievalLayer] Initializing Advanced Retrieval System (Layer 4)...")

# 1. Query Processor
from app.services.retrieval.query_processor import get_query_processor, QueryProcessor
logger.info("[RetrievalLayer] Successfully imported QueryProcessor components.")

# 2. Model Router
from app.services.retrieval.model_router import get_model_router, ModelRouter
logger.info("[RetrievalLayer] Successfully imported ModelRouter components.")

# 3. BM25 Retriever
from app.services.retrieval.bm25_retriever import get_bm25_retriever, BM25Retriever
logger.info("[RetrievalLayer] Successfully imported BM25Retriever components.")

# 4. Vector Retriever
from app.services.retrieval.vector_retriever import get_vector_retriever, VectorRetriever
logger.info("[RetrievalLayer] Successfully imported VectorRetriever components.")

# 5. RRF Fusion
from app.services.retrieval.rrf_fusion import get_rrf_fusion, RRFFusion
logger.info("[RetrievalLayer] Successfully imported RRFFusion components.")

# 6. Reranker
from app.services.retrieval.reranker import get_reranker, BGEReranker
logger.info("[RetrievalLayer] Successfully imported BGEReranker components.")

logger.info("[RetrievalLayer] All retrieval services initialized and ready.")

__all__ = [
    'get_query_processor',
    'QueryProcessor',
    'get_model_router',
    'ModelRouter',
    'get_bm25_retriever',
    'BM25Retriever',
    'get_vector_retriever',
    'VectorRetriever',
    'get_rrf_fusion',
    'RRFFusion',
    'get_reranker',
    'BGEReranker',
]