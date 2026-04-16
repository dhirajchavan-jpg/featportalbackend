# app/services/retrieval/rrf_fusion.py
"""
Layer 4: Reciprocal Rank Fusion (RRF)
Combines BM25 and vector search results using RRF algorithm.
"""

from typing import List, Dict, Any
from collections import defaultdict
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class RRFFusion:
    """
    Reciprocal Rank Fusion for combining multiple retrieval results.
    RRF Score = Σ(1 / (k + rank_i)) for each result list
    """
    
    def __init__(self, k: int = None):
        self.k =settings.RRF_K  # Default: 60
        logger.info(f"[RRFFusion] Initialized with k={self.k}")
    
    def fuse(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Fuse BM25 and vector search results using RRF.
        
        Args:
            bm25_results: List of BM25 results
            vector_results: List of vector search results
            top_k: Number of top results to return
            
        Returns:
            Fused and ranked results
        """
        top_k = top_k or settings.RRF_TOP_K
        
        logger.info(f"[RRFFusion] Starting Fusion. Inputs: {len(bm25_results)} BM25 results + {len(vector_results)} Vector results. Target Top-K: {top_k}")
        
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        doc_data = {}  # Store document data
        
        # Process BM25 results
        logger.info("[RRFFusion] Processing BM25 results...")
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result['id']
            rrf_scores[doc_id] += 1.0 / (self.k + rank)
            
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    'id': doc_id,
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'bm25_rank': rank,
                    'bm25_score': result['score'],
                    'vector_rank': None,
                    'vector_score': None
                }
        
        # Process vector results
        logger.info("[RRFFusion] Processing Vector results...")
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result['id']
            rrf_scores[doc_id] += 1.0 / (self.k + rank)
            
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    'id': doc_id,
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'bm25_rank': None,
                    'bm25_score': None,
                    'vector_rank': rank,
                    'vector_score': result['score']
                }
            else:
                doc_data[doc_id]['vector_rank'] = rank
                doc_data[doc_id]['vector_score'] = result['score']
        
        # Sort by RRF score
        logger.info(f"[RRFFusion] Sorting {len(rrf_scores)} unique documents by RRF score...")
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Build final results
        logger.info(f"[RRFFusion] Building final result list for top {len(sorted_docs)} documents...")
        fused_results = []
        for rank, (doc_id, rrf_score) in enumerate(sorted_docs, start=1):
            doc = doc_data[doc_id]
            doc['rrf_rank'] = rank
            doc['rrf_score'] = rrf_score
            doc['fusion_method'] = 'rrf'
            
            # Determine which retrieval methods found this doc
            found_in = []
            if doc['bm25_rank']:
                found_in.append('bm25')
            if doc['vector_rank']:
                found_in.append('vector')
            doc['found_in'] = found_in
            
            fused_results.append(doc)
        
        logger.info(f"[RRFFusion] Fused to {len(fused_results)} results")
        return fused_results
    
    def get_fusion_stats(self, fused_results: List[Dict]) -> Dict[str, Any]:
        """Get statistics about fusion results."""
        logger.info("[RRFFusion] Calculating fusion stats...")
        if not fused_results:
            logger.info("[RRFFusion] No results to calculate stats for.")
            return {}
        
        found_in_both = sum(1 for r in fused_results if len(r.get('found_in', [])) == 2)
        found_in_bm25_only = sum(1 for r in fused_results if r.get('found_in') == ['bm25'])
        found_in_vector_only = sum(1 for r in fused_results if r.get('found_in') == ['vector'])
        
        stats = {
            'total_results': len(fused_results),
            'found_in_both': found_in_both,
            'found_in_bm25_only': found_in_bm25_only,
            'found_in_vector_only': found_in_vector_only,
            'avg_rrf_score': sum(r['rrf_score'] for r in fused_results) / len(fused_results),
            'max_rrf_score': max(r['rrf_score'] for r in fused_results),
            'min_rrf_score': min(r['rrf_score'] for r in fused_results)
        }
        logger.info(f"[RRFFusion] Stats calculated: {stats}")
        return stats
    
    def explain_ranking(self, result: Dict[str, Any]) -> str:
        """Generate explanation for a result's ranking."""
        explanation_parts = []
        
        explanation_parts.append(f"RRF Rank: {result['rrf_rank']}")
        explanation_parts.append(f"RRF Score: {result['rrf_score']:.4f}")
        
        if result.get('bm25_rank'):
            explanation_parts.append(f"BM25 Rank: {result['bm25_rank']} (score: {result['bm25_score']:.4f})")
        
        if result.get('vector_rank'):
            explanation_parts.append(f"Vector Rank: {result['vector_rank']} (score: {result['vector_score']:.4f})")
        
        found_in = result.get('found_in', [])
        if len(found_in) == 2:
            explanation_parts.append("Found by both retrieval methods (strong match)")
        elif 'bm25' in found_in:
            explanation_parts.append("Found by keyword search only")
        elif 'vector' in found_in:
            explanation_parts.append("Found by semantic search only")
        
        return " | ".join(explanation_parts)


# Singleton instance
_rrf_fusion = None

def get_rrf_fusion() -> RRFFusion:
    """Get or create singleton RRFFusion instance."""
    global _rrf_fusion
    if _rrf_fusion is None:
        logger.info("[RRFFusion] Creating singleton instance.")
        _rrf_fusion = RRFFusion()
    else:
        logger.info("[RRFFusion] Returning existing singleton instance.")
    return _rrf_fusion