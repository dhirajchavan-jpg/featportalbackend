# app/services/retrieval/bm25_retriever.py
"""
Layer 4: BM25 Sparse Retriever
Supports Parallel Search across Multiple Indices (Projects + Sectors)
"""

from typing import List, Dict, Any
import asyncio
from qdrant_client import models
from app.config import settings
from app.schemas import SearchFilter
from app.core.llm_provider import get_qdrant_client, get_sparse_embedder
import logging

# --- FIX: USE STANDARD OPENTELEMETRY WITH STRINGS ---
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class BM25Retriever:
    def __init__(self):
        logger.info("[BM25Retriever] Initializing...")
        self.qdrant_client = get_qdrant_client()
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        logger.info(f"[BM25Retriever] Initialized with collection: {self.collection_name}")

    async def retrieve(
        self,
        query: str,
        search_filter: SearchFilter,
        top_k: int = None,
        sources_to_search: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches multiple BM25 indices simultaneously.
        """
        # --- START TRACE ---
        with tracer.start_as_current_span("BM25 Retrieval") as span:
            # Manually set the Phoenix attributes (No imports needed)
            span.set_attribute("openinference.span.kind", "RETRIEVER")
            span.set_attribute("input.value", query)

            top_k = settings.BM25_TOP_K
            
            logger.info(f"[BM25] Starting retrieval for query: '{query[:50]}...' (Top K: {top_k})")
            
            # 1. Determine targets
            targets = sources_to_search if sources_to_search else search_filter.sources
            
            if not targets:
                logger.warning("[BM25] No targets (sources) specified for search. Returning empty results.")
                return []

            logger.info(f"[BM25] Parallel Search Targets identified: {targets}")

            # 2. Define Single Search Logic
            async def search_single_index(target_id: str):
                # Trace individual shard searches
                with tracer.start_as_current_span(f"bm25_shard_{target_id}"):
                    try:
                        logger.info(f"[BM25] Processing shard: {target_id}")
                        sparse_embedder = get_sparse_embedder(target_id)
                        
                        if sparse_embedder is None or sparse_embedder.bm25_index is None:
                            logger.info(f"[BM25] Sparse embedder/index not found for target: {target_id}. Skipping.")
                            return []

                        sparse_vec = sparse_embedder.get_sparse_embedding(query)
                        if not sparse_vec:
                            logger.info(f"[BM25] No sparse vector terms generated for query in target: {target_id}. Skipping.")
                            return []
                        
                        logger.info(f"[BM25] Generated sparse vector for {target_id} with {len(sparse_vec)} active terms.")

                        specific_filter = models.Filter(
                            must=[models.FieldCondition(key="metadata.source", match=models.MatchValue(value=target_id))]
                        )
                        
                        if search_filter.excluded_files:
                            if not specific_filter.must_not: specific_filter.must_not = []
                            specific_filter.must_not.append(
                                models.FieldCondition(key="metadata.file_name", match=models.MatchAny(any=search_filter.excluded_files))
                            )
                            logger.info(f"[BM25] Applied exclusion filter for {len(search_filter.excluded_files)} files in target {target_id}.")

                        logger.info(f"[BM25] Executing Qdrant search for target: {target_id}")
                        results = self.qdrant_client.search(
                            collection_name=self.collection_name,
                            query_vector=models.NamedSparseVector(
                                name=settings.QDRANT_SPARSE_VECTOR_NAME,
                                vector=models.SparseVector(
                                    indices=list(sparse_vec.keys()),
                                    values=list(sparse_vec.values())
                                )
                            ),
                            query_filter=specific_filter,
                            limit=top_k,
                            with_payload=True,
                            timeout=10
                        )
                        logger.info(f"[BM25] Found {len(results)} results in target: {target_id}")
                        return results
                    except Exception as e:
                        logger.warning(f"[BM25] Failed to search '{target_id}': {e}", exc_info=True)
                        return []

            # 3. Execute in Parallel
            logger.info(f"[BM25] Spawning {len(targets)} parallel search tasks...")
            tasks = [search_single_index(target) for target in targets]
            results_list = await asyncio.gather(*tasks)
            logger.info("[BM25] All parallel tasks completed.")

            # 4. Merge & Deduplicate
            all_results = []
            seen_ids = set()
            
            for i, batch in enumerate(results_list):
                if batch:
                    logger.info(f"[BM25] Merging batch {i+1}/{len(results_list)} containing {len(batch)} items.")
                for result in batch:
                    if result.id not in seen_ids:
                        seen_ids.add(result.id)
                        all_results.append(result)

            logger.info(f"[BM25] Total unique results before sorting: {len(all_results)}")

            all_results.sort(key=lambda x: x.score, reverse=True)
            final_results = self._format_results(all_results[:top_k])
            
            logger.info(f"[BM25] Final results count after top_k cut: {len(final_results)}")
            
            # Log output size
            span.set_attribute("output.value", f"{len(final_results)} documents found")
            
            return final_results

    def _format_results(self, results) -> List[Dict[str, Any]]:
        logger.info(f"[BM25] Formatting {len(results)} results for output.")
        return [{
            'rank': i + 1,
            'score': res.score,
            'id': res.id,
            'content': res.payload.get('page_content', ''),
            'metadata': res.payload.get('metadata', {}),
            'retrieval_type': 'bm25'
        } for i, res in enumerate(results)]

_bm25_retriever = None
def get_bm25_retriever() -> BM25Retriever:
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever()
    return _bm25_retriever