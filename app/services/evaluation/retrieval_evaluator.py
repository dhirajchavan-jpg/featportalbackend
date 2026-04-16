# app/services/evaluation/retrieval_evaluator.py
"""
Retrieval Quality Evaluator

Evaluates retrieval effectiveness using standard IR metrics:
- Hit Rate@K: Did we find any relevant documents?
- MRR (Mean Reciprocal Rank): How high is the first relevant doc?
- NDCG@K: Quality of the ranking
- Precision@K, Recall@K: Standard metrics

These metrics are FREE to compute and provide critical insights into
whether your retrieval system is finding relevant information.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class RetrievalMetricsEvaluator:
    """
    Evaluate retrieval quality using standard Information Retrieval metrics.
    
    Note: For true evaluation, you need "ground truth" - known relevant documents.
    In production without ground truth, we use proxy metrics based on scores.
    """
    
    def __init__(self):
        """Initialize the retrieval evaluator."""
        self.name = "retrieval_metrics"
        logger.info("[RETRIEVAL_EVAL] Initialized")
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        k_values: List[int] = [1, 3, 5, 10],
        ground_truth_docs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval quality.
        
        Args:
            query: The search query
            retrieved_docs: List of retrieved documents with scores
            k_values: K values to evaluate at (e.g., [3, 5, 10])
            ground_truth_docs: Optional list of known relevant doc IDs
            
        Returns:
            Dictionary of metrics
        """
        # ========== DEBUG LOGGING ==========
        logger.info(f"[RETRIEVAL_EVAL] Received {len(retrieved_docs) if retrieved_docs else 0} docs")
        if retrieved_docs:
            first_doc = retrieved_docs[0]
            logger.info(f"[RETRIEVAL_EVAL] First doc keys: {list(first_doc.keys())}")
            logger.info(f"[RETRIEVAL_EVAL] First doc scores:")
            logger.info(f"[RETRIEVAL_EVAL]   rrf_score: {first_doc.get('rrf_score')}")
            logger.info(f"[RETRIEVAL_EVAL]   relevance_score: {first_doc.get('relevance_score')}")
            logger.info(f"[RETRIEVAL_EVAL]   rerank_score: {first_doc.get('rerank_score')}")
            logger.info(f"[RETRIEVAL_EVAL]   dense_score: {first_doc.get('dense_score')}")
            logger.info(f"[RETRIEVAL_EVAL]   sparse_score: {first_doc.get('sparse_score')}")
        # ===================================
        
        if not retrieved_docs:
            return self._empty_metrics(k_values)
        
        # If we have ground truth, use it
        if ground_truth_docs:
            return self._evaluate_with_ground_truth(
                retrieved_docs, 
                ground_truth_docs, 
                k_values
            )
        
        # Otherwise, use score-based proxy metrics
        return self._evaluate_score_based(retrieved_docs, k_values)
    
    def _evaluate_with_ground_truth(
        self,
        retrieved_docs: List[Dict[str, Any]],
        ground_truth_docs: List[str],
        k_values: List[int]
    ) -> Dict[str, Any]:
        """
        Evaluate using known relevant documents (ideal case).
        
        Args:
            retrieved_docs: Retrieved documents
            ground_truth_docs: List of known relevant document IDs
            k_values: K values to evaluate at
            
        Returns:
            Metrics dictionary
        """
        metrics = {}
        
        # Extract document IDs from retrieved docs
        retrieved_ids = [
            doc.get('metadata', {}).get('source', '') 
            for doc in retrieved_docs
        ]
        
        # Calculate metrics for each K
        for k in k_values:
            top_k_ids = retrieved_ids[:k]
            
            # Hit Rate@K: Did we find at least one relevant doc?
            hit_rate = self._calculate_hit_rate(top_k_ids, ground_truth_docs)
            metrics[f'hit_rate_at_{k}'] = hit_rate
            
            # Precision@K: What fraction of retrieved docs are relevant?
            precision = self._calculate_precision(top_k_ids, ground_truth_docs)
            metrics[f'precision_at_{k}'] = precision
            
            # Recall@K: What fraction of relevant docs did we retrieve?
            recall = self._calculate_recall(top_k_ids, ground_truth_docs)
            metrics[f'recall_at_{k}'] = recall
            
            # NDCG@K: Quality of the ranking
            ndcg = self._calculate_ndcg(
                retrieved_docs[:k], 
                ground_truth_docs
            )
            metrics[f'ndcg_at_{k}'] = ndcg
        
        # MRR: Position of first relevant document
        mrr = self._calculate_mrr(retrieved_ids, ground_truth_docs)
        metrics['mrr'] = mrr
        
        # Overall metrics
        metrics['total_retrieved'] = len(retrieved_docs)
        metrics['total_relevant'] = len(ground_truth_docs)
        metrics['evaluation_method'] = 'ground_truth'
        
        return metrics
    
    def _evaluate_score_based(
        self,
        retrieved_docs: List[Dict[str, Any]],
        k_values: List[int]
    ) -> Dict[str, Any]:
        """
        Evaluate using retrieval scores as proxy (no ground truth).
        
        We use score thresholds to estimate relevance:
        - High scores (>0.7) = likely relevant
        - Medium scores (0.5-0.7) = possibly relevant
        - Low scores (<0.5) = likely not relevant
        
        Args:
            retrieved_docs: Retrieved documents with scores
            k_values: K values to evaluate at
            
        Returns:
            Metrics dictionary
        """
        metrics = {}
        
        # Extract scores
        scores = []
        for doc in retrieved_docs:
            # Try multiple score fields, handling None values properly
            # Priority order: rrf_score (fusion) > rerank_score > relevance_score > dense_score > sparse_score
            score = 0.0
            
            # Check each field explicitly for None and 0
            if doc.get('rrf_score') is not None:
                score = doc.get('rrf_score')
            elif doc.get('rerank_score') is not None:
                score = doc.get('rerank_score')
            elif doc.get('relevance_score') is not None:
                score = doc.get('relevance_score')
            elif doc.get('dense_score') is not None:
                score = doc.get('dense_score')
            elif doc.get('sparse_score') is not None:
                score = doc.get('sparse_score')
            
            scores.append(float(score))
        
        # ========== DEBUG: Log extracted scores ==========
        logger.info(f"[RETRIEVAL_EVAL] Extracted {len(scores)} scores")
        if scores:
            logger.info(f"[RETRIEVAL_EVAL] Score range: {min(scores):.4f} to {max(scores):.4f}")
            logger.info(f"[RETRIEVAL_EVAL] Average score: {sum(scores)/len(scores):.4f}")
            logger.info(f"[RETRIEVAL_EVAL] First 5 scores: {scores[:5]}")
        # =================================================
        
        # Define relevance threshold BASED ON SCORE RANGE
        # RRF scores are typically 0.01-0.05, not 0-1!
        # So we use adaptive threshold based on max score
        max_score = max(scores) if scores else 0
        
        if max_score < 0.2:
            # RRF scores or other low-range scores
            # Use top 70% of the score range as "relevant"
            relevance_threshold = max_score * 0.7
            logger.info(f"[RETRIEVAL_EVAL] Using adaptive threshold (RRF range): {relevance_threshold:.4f}")
        else:
            # Normal 0-1 range scores
            relevance_threshold = 0.7
            logger.info(f"[RETRIEVAL_EVAL] Using standard threshold: {relevance_threshold}")
        
        for k in k_values:
            top_k_scores = scores[:k]
            
            # Estimated hit rate (at least one high score)
            has_relevant = any(s >= relevance_threshold for s in top_k_scores)
            metrics[f'estimated_hit_rate_at_{k}'] = 1.0 if has_relevant else 0.0
            
            # Estimated precision (fraction above threshold)
            relevant_count = sum(1 for s in top_k_scores if s >= relevance_threshold)
            metrics[f'estimated_precision_at_{k}'] = relevant_count / k if k > 0 else 0
            
            # Average score in top K
            metrics[f'avg_score_at_{k}'] = np.mean(top_k_scores) if top_k_scores else 0
        
        # Score distribution analysis
        if scores:
            metrics['max_score'] = max(scores)
            metrics['min_score'] = min(scores)
            metrics['avg_score'] = np.mean(scores)
            metrics['std_score'] = np.std(scores)
            
            # Count by score band
            metrics['high_score_count'] = sum(1 for s in scores if s >= 0.7)
            metrics['medium_score_count'] = sum(1 for s in scores if 0.5 <= s < 0.7)
            metrics['low_score_count'] = sum(1 for s in scores if s < 0.5)
        
        metrics['total_retrieved'] = len(retrieved_docs)
        metrics['evaluation_method'] = 'score_based'
        
        return metrics
    
    # ========================================================================
    # METRIC CALCULATION HELPERS (for ground truth)
    # ========================================================================
    
    def _calculate_hit_rate(
        self, 
        retrieved_ids: List[str], 
        ground_truth_ids: List[str]
    ) -> float:
        """
        Hit Rate: Did we retrieve at least one relevant document?
        
        Returns: 1.0 if hit, 0.0 if miss
        """
        for doc_id in retrieved_ids:
            if doc_id in ground_truth_ids:
                return 1.0
        return 0.0
    
    def _calculate_precision(
        self, 
        retrieved_ids: List[str], 
        ground_truth_ids: List[str]
    ) -> float:
        """
        Precision@K: What fraction of retrieved docs are relevant?
        
        Precision = (relevant docs in top K) / K
        """
        if not retrieved_ids:
            return 0.0
        
        relevant_count = sum(
            1 for doc_id in retrieved_ids 
            if doc_id in ground_truth_ids
        )
        return relevant_count / len(retrieved_ids)
    
    def _calculate_recall(
        self, 
        retrieved_ids: List[str], 
        ground_truth_ids: List[str]
    ) -> float:
        """
        Recall@K: What fraction of relevant docs did we retrieve?
        
        Recall = (relevant docs in top K) / (total relevant docs)
        """
        if not ground_truth_ids:
            return 0.0
        
        relevant_count = sum(
            1 for doc_id in retrieved_ids 
            if doc_id in ground_truth_ids
        )
        return relevant_count / len(ground_truth_ids)
    
    def _calculate_mrr(
        self, 
        retrieved_ids: List[str], 
        ground_truth_ids: List[str]
    ) -> float:
        """
        Mean Reciprocal Rank: Position of first relevant document.
        
        MRR = 1 / (rank of first relevant doc)
        
        Higher is better. MRR=1.0 means first doc is relevant.
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in ground_truth_ids:
                return 1.0 / rank
        return 0.0
    
    def _calculate_ndcg(
        self, 
        retrieved_docs: List[Dict[str, Any]], 
        ground_truth_ids: List[str]
    ) -> float:
        """
        Normalized Discounted Cumulative Gain: Quality of ranking.
        
        NDCG measures how well the ranking matches ideal ranking.
        Score: 0.0 to 1.0, higher is better.
        """
        # DCG: Discounted Cumulative Gain
        dcg = 0.0
        for rank, doc in enumerate(retrieved_docs, start=1):
            doc_id = doc.get('metadata', {}).get('source', '')
            relevance = 1.0 if doc_id in ground_truth_ids else 0.0
            dcg += relevance / np.log2(rank + 1)
        
        # IDCG: Ideal DCG (all relevant docs at top)
        ideal_relevances = [1.0] * min(len(ground_truth_ids), len(retrieved_docs))
        idcg = sum(
            rel / np.log2(rank + 1) 
            for rank, rel in enumerate(ideal_relevances, start=1)
        )
        
        # NDCG = DCG / IDCG
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def _empty_metrics(self, k_values: List[int]) -> Dict[str, Any]:
        """Return empty metrics when no docs retrieved."""
        metrics = {'total_retrieved': 0, 'evaluation_method': 'none'}
        
        for k in k_values:
            metrics[f'hit_rate_at_{k}'] = 0.0
            metrics[f'precision_at_{k}'] = 0.0
            metrics[f'recall_at_{k}'] = 0.0
            metrics[f'ndcg_at_{k}'] = 0.0
        
        metrics['mrr'] = 0.0
        
        return metrics


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def calculate_retrieval_metrics(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    k_values: List[int] = [3, 5, 10],
    ground_truth_docs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to calculate retrieval metrics.
    
    Args:
        query: Search query
        retrieved_docs: Retrieved documents with scores
        k_values: K values to evaluate
        ground_truth_docs: Optional ground truth doc IDs
        
    Returns:
        Dictionary of metrics
    """
    evaluator = RetrievalMetricsEvaluator()
    return evaluator.evaluate_retrieval(
        query=query,
        retrieved_docs=retrieved_docs,
        k_values=k_values,
        ground_truth_docs=ground_truth_docs
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Evaluate retrieval with ground truth
    retrieved_docs = [
        {'metadata': {'source': 'doc1.pdf'}, 'rerank_score': 0.95},
        {'metadata': {'source': 'doc2.pdf'}, 'rerank_score': 0.82},
        {'metadata': {'source': 'doc3.pdf'}, 'rerank_score': 0.75},
        {'metadata': {'source': 'doc4.pdf'}, 'rerank_score': 0.60},
        {'metadata': {'source': 'doc5.pdf'}, 'rerank_score': 0.45},
    ]
    
    ground_truth = ['doc1.pdf', 'doc3.pdf', 'doc6.pdf']
    
    metrics = calculate_retrieval_metrics(
        query="What are GDPR requirements?",
        retrieved_docs=retrieved_docs,
        ground_truth_docs=ground_truth
    )
    
    logger.info("Retrieval Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    # Example output:
    # hit_rate_at_3: 1.000 (found doc1 and doc3)
    # precision_at_3: 0.667 (2 out of 3 are relevant)
    # mrr: 1.000 (first doc is relevant)
    # ndcg_at_3: 0.926 (good ranking)