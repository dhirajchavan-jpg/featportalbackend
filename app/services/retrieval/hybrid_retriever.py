# app/services/retrieval/hybrid_retriever.py

from typing import List, Dict, Any
import asyncio
from app.config import settings
from app.schemas import SearchFilter
import logging

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

from app.services.retrieval.bm25_retriever import get_bm25_retriever
from app.services.retrieval.vector_retriever import get_vector_retriever
from app.services.retrieval.reranker import get_reranker

class HybridRetriever:
    def __init__(self):
        logger.info("[HybridRetriever] Initializing components...")
        self.bm25 = get_bm25_retriever()
        self.vector = get_vector_retriever()
        self.reranker = get_reranker()

    async def retrieve(
        self,
        query: str,
        project_id: str,
        selected_sectors: List[str], 
        excluded_files: List[str] = None,
        top_k_final: int = 5,
        # --- NEW ARGUMENTS ---
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        use_reranking: bool = True
    ) -> Dict[str, Any]:
        
        with tracer.start_as_current_span("Hybrid RAG Pipeline") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("input.value", query)

            logger.info(f" [HybridRetriever] Query: '{query[:50]}...'")
            logger.info(f" [Params] Dense: {dense_weight}, Sparse: {sparse_weight}, Rerank: {use_reranking}")

            combined_sources = []
            if project_id: combined_sources.append(project_id)
            if selected_sectors: combined_sources.extend(selected_sectors)
            
            if not combined_sources:
                logger.warning(" [HybridRetriever] No sources provided.")
                return {'results': [], 'stats': {}}

            search_filter = SearchFilter(sources=combined_sources, excluded_files=excluded_files)

            # --- DYNAMIC STRATEGY EXECUTION ---
            bm25_res = []
            vector_res = []
            
            run_bm25 = sparse_weight > 0.0
            run_vector = dense_weight > 0.0
            
            tasks = []
            
            if run_vector:
                logger.info("    Launching Vector Search...")
                tasks.append(self.vector.retrieve(query, search_filter))
            
            if run_bm25:
                logger.info("    Launching BM25 Search...")
                tasks.append(self.bm25.retrieve(
                    query, 
                    search_filter, 
                    sources_to_search=combined_sources 
                ))
            
            # Execute
            results = await asyncio.gather(*tasks)
            
            # Unpack
            idx = 0
            if run_vector:
                vector_res = results[idx]
                idx += 1
            if run_bm25:
                bm25_res = results[idx]
            
            logger.info(f"    Results: Vector={len(vector_res)}, BM25={len(bm25_res)}")

            # --- FUSION LOGIC ---
            if run_vector and run_bm25:
                logger.info("   Performing RRF Fusion...")
                fused_results = self._rrf_fusion(bm25_res, vector_res, k=60)
            else:
                # Single path: just take the one that ran
                logger.info("   Skipping Fusion (Single Strategy Active).")
                fused_results = vector_res if run_vector else bm25_res

            # --- RERANKING LOGIC ---
            if use_reranking and fused_results:
                logger.info(f"   Reranking {len(fused_results)} docs...")
                # Check if model loaded
                if self.reranker.model is None:
                    try: 
                        self.reranker.load_model()
                    except Exception as e:
                        logger.warning(f"    Reranker load failed: {e}")
                        return {'results': fused_results[:top_k_final]}

                reranked_results = await self.reranker.rerank(query, fused_results, top_k=top_k_final)
                logger.info(f"   Reranking complete. Top {len(reranked_results)} kept.")
            else:
                logger.info("   Reranking skipped.")
                reranked_results = fused_results[:top_k_final]
            
            result_payload = {
                'results': reranked_results,
                'stats': {
                    'bm25_count': len(bm25_res),
                    'vector_count': len(vector_res),
                    'scope': combined_sources
                }
            }
            
            span.set_attribute("output.value", f"Returned {len(reranked_results)} docs")
            return result_payload

    def _rrf_fusion(self, list_a, list_b, k=60):
        with tracer.start_as_current_span("RRF Fusion") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            
            scores = {}
            doc_map = {}
            
            # Process List A (usually BM25)
            for rank, doc in enumerate(list_a):
                doc_id = doc['id']
                doc_map[doc_id] = doc
                scores[doc_id] = scores.get(doc_id, 0) + (1 / (k + rank + 1))
            
            # Process List B (usually Vector)
            for rank, doc in enumerate(list_b):
                doc_id = doc['id']
                if doc_id not in doc_map: doc_map[doc_id] = doc
                scores[doc_id] = scores.get(doc_id, 0) + (1 / (k + rank + 1))
            
            # Sort
            sorted_docs = []
            for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                doc = doc_map[doc_id]
                doc['rrf_score'] = score
                sorted_docs.append(doc)
            return sorted_docs[:settings.RRF_TOP_K * 2]

_hybrid_retriever = None
def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever