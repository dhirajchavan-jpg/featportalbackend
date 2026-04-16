# app/services/retrieval/vector_retriever.py
"""
Layer 4: Dense Vector Retriever
"""

from typing import List, Dict, Any
from qdrant_client import models
from app.config import settings
from app.schemas import SearchFilter
import logging

# --- FIX: STANDARD OPENTELEMETRY ---
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

from app.core.llm_provider import get_qdrant_client, get_dense_embedder

class VectorRetriever:
    def __init__(self):
        logger.info("[VectorRetriever] Initializing VectorRetriever...")
        self.qdrant_client = get_qdrant_client()
        self.dense_embedder = get_dense_embedder()
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        logger.info(f"[VectorRetriever] Initialized. Target Collection: {self.collection_name}")

    async def retrieve(
        self,
        query: str,
        search_filter: SearchFilter,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Performs dense retrieval. 
        """
        logger.info(f"[VectorRetriever] Starting retrieve for query: '{query[:50]}...'")
        
        # --- START TRACE ---
        with tracer.start_as_current_span("Dense Retrieval") as span:
            span.set_attribute("openinference.span.kind", "RETRIEVER")
            span.set_attribute("input.value", query)

            top_k = settings.DENSE_TOP_K
            
            try:
                # 1. Generate Embedding
                logger.info("[VectorRetriever] Step 1: Generating query embedding...")
                # query_vector = self.dense_embedder.embed_query(query)
                query_vector = await self.dense_embedder.aembed_query(query)
                logger.info("[VectorRetriever] Embedding generated successfully.")
                
                # 2. Build Composite Filter
                logger.info("[VectorRetriever] Step 2: Building search filters...")
                qdrant_filter = self._build_qdrant_filter(search_filter)
                
                # 3. Execute Search
                logger.info(f"[VectorRetriever] Step 3: Executing Qdrant search in '{self.collection_name}'...")
                results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=(settings.QDRANT_DENSE_VECTOR_NAME, query_vector),
                    query_filter=qdrant_filter,
                    limit=top_k,
                    with_payload=True,
                    timeout=30 
                )
                
                count = len(results)
                logger.info(f"[VectorRetriever] Search complete. Found {count} raw chunks.")
                
                logger.info("[VectorRetriever] Formatting results...")
                final_results = self._format_results(results)
                
                logger.info(f"[VectorRetriever] Returning {len(final_results)} formatted documents.")
                span.set_attribute("output.value", f"{len(final_results)} documents found")
                return final_results
                
            except Exception as e:
                logger.error(f"[VectorRetriever] Critical Error during retrieval: {e}")
                span.record_exception(e)
                return []
            
    async def get_chunks_by_filename(self, project_id: str, filename: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetches stored chunks for a specific file to inspect indexing quality.
        Uses Qdrant 'Scroll' to get points without vector similarity.
        """
        logger.info(f"[VectorRetriever] Inspecting chunks for file: '{filename}' in project: '{project_id}'")
        
        # Build filter for exact file match
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=project_id)
                ),
                models.FieldCondition(
                    key="metadata.file_name",
                    match=models.MatchValue(value=filename)
                )
            ]
        )
        
        try:
            # Use 'scroll' to retrieve raw points
            response = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False # We don't need the dense vector array for inspection
            )
            
            points, _ = response
            
            # Format nicely
            return [{
                "chunk_id": point.id,
                "content": point.payload.get('page_content', ''),
                "metadata": point.payload.get('metadata', {})
            } for point in points]
            
        except Exception as e:
            logger.error(f"[VectorRetriever] Failed to fetch file chunks: {e}")
            return []

    def _build_qdrant_filter(self, search_filter: SearchFilter) -> models.Filter:
        logger.info(f"[VectorRetriever] Constructing filter. Sources: {search_filter.sources}, Excluded: {len(search_filter.excluded_files) if search_filter.excluded_files else 0} files")
        
        must_conditions = []
        must_not_conditions = []
        
        if search_filter.sources:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchAny(any=search_filter.sources)
                )
            )
        
        if search_filter.excluded_files:
            must_not_conditions.append(
                models.FieldCondition(
                    key="metadata.file_name",
                    match=models.MatchAny(any=search_filter.excluded_files)
                )
            )
            
        # 3. NEW: Query Understanding Filters
        # Only apply if we have high confidence and fields are present
        if search_filter.query_understanding and search_filter.query_understanding.confidence > 0.5:
            qu = search_filter.query_understanding
            
            # Filter by Authority (maps to 'sector' or 'source' usually, but let's assume specific metadata)
            # If your file tagging logic puts RBI/SEBI in metadata.sector:
            if qu.authority:
                logger.info(f"[VectorRetriever] Applying Authority Filter: {qu.authority}")
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.sector", 
                        match=models.MatchValue(value=qu.authority)
                    )
                )
            
            # Filter by Document Type (Circular, Act, etc.)
            if qu.document_type:
                logger.info(f"[VectorRetriever] Applying Doc Type Filter: {qu.document_type}")
                must_conditions.append(
                    models.FieldCondition(
                        key="metadata.document_type", # Ensure your indexer saves this key!
                        match=models.MatchValue(value=qu.document_type)
                    )
                )
        filter_dict = {}
        if must_conditions: filter_dict['must'] = must_conditions
        if must_not_conditions: filter_dict['must_not'] = must_not_conditions
        
        if filter_dict:
            logger.info(f"[VectorRetriever] Filter built with {len(must_conditions)} must and {len(must_not_conditions)} must_not conditions.")
            return models.Filter(**filter_dict)
        else:
            logger.info("[VectorRetriever] No filters applied.")
            return None

    def _format_results(self, results) -> List[Dict[str, Any]]:
        return [{
            'rank': i + 1,
            'score': res.score,
            'id': res.id,
            'content': res.payload.get('page_content', ''),
            'metadata': res.payload.get('metadata', {}),
            'retrieval_type': 'vector'
        } for i, res in enumerate(results)]

_vector_retriever = None
def get_vector_retriever() -> VectorRetriever:
    global _vector_retriever
    if _vector_retriever is None:
        logger.info("[VectorRetriever] Creating singleton instance.")
        _vector_retriever = VectorRetriever()
    else:
        logger.info("[VectorRetriever] Returning existing singleton instance.")
    return _vector_retriever