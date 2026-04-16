# app/services/evaluation/reranking_evaluator.py
"""
Reranking Effectiveness Evaluator

Evaluates how well reranking improves the initial retrieval results:
1. Score improvement (before vs after)
2. Ranking correlation (how much did order change)
3. Position changes (which documents moved up/down)
4. Reciprocal rank improvement

This helps determine if reranking is actually improving results.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from scipy.stats import spearmanr, kendalltau

logger = logging.getLogger(__name__)


class RerankingEffectivenessEvaluator:
    """
    Evaluate reranking effectiveness by comparing before/after.
    """
    
    def __init__(self):
        """Initialize reranking evaluator."""
        self.name = "reranking_effectiveness"
        logger.info("[RERANKING_EVAL] Initialized")
    
    def evaluate_reranking(
        self,
        docs_before_rerank: List[Dict[str, Any]],
        docs_after_rerank: List[Dict[str, Any]],
        reranking_time_ms: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate reranking effectiveness.
        
        Args:
            docs_before_rerank: Documents before reranking (with scores)
            docs_after_rerank: Documents after reranking (with rerank scores)
            reranking_time_ms: Time taken to rerank
            
        Returns:
            Dictionary of metrics
        """
        if not docs_before_rerank or not docs_after_rerank:
            return self._empty_metrics(reranking_time_ms)
        
        metrics = {
            'reranking_time_ms': reranking_time_ms,
            'num_docs_before': len(docs_before_rerank),
            'num_docs_after': len(docs_after_rerank)
        }
        
        # Extract scores
        scores_before = self._extract_scores(docs_before_rerank, score_type='before')
        scores_after = self._extract_scores(docs_after_rerank, score_type='after')
        
        # Score improvement
        score_metrics = self._analyze_score_changes(scores_before, scores_after)
        metrics.update(score_metrics)
        
        # Ranking changes
        ranking_metrics = self._analyze_ranking_changes(
            docs_before_rerank, 
            docs_after_rerank
        )
        metrics.update(ranking_metrics)
        
        # Position changes
        position_metrics = self._analyze_position_changes(
            docs_before_rerank,
            docs_after_rerank
        )
        metrics.update(position_metrics)
        
        # Overall effectiveness
        effectiveness = self._calculate_overall_effectiveness(metrics)
        metrics['reranking_effectiveness_score'] = effectiveness
        
        return metrics
    
    def _extract_scores(
        self,
        docs: List[Dict[str, Any]],
        score_type: str
    ) -> List[float]:
        """
        Extract scores from documents.
        
        Args:
            docs: List of documents
            score_type: 'before' or 'after'
            
        Returns:
            List of scores
        """
        scores = []
        for doc in docs:
            if score_type == 'after':
                # After reranking, use rerank_score
                score = doc.get('rerank_score', 0.0)
            else:
                # Before reranking, use relevance_score or dense_score
                score = (
                    doc.get('relevance_score') or 
                    doc.get('dense_score') or 
                    doc.get('sparse_score') or 
                    0.0
                )
            scores.append(float(score))
        return scores
    
    def _analyze_score_changes(
        self,
        scores_before: List[float],
        scores_after: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze how scores changed.
        """
        if not scores_before or not scores_after:
            return {}
        
        # Ensure same length
        min_len = min(len(scores_before), len(scores_after))
        scores_before = scores_before[:min_len]
        scores_after = scores_after[:min_len]
        
        # Calculate improvements
        improvements = [
            after - before 
            for before, after in zip(scores_before, scores_after)
        ]
        
        return {
            'score_before_mean': np.mean(scores_before),
            'score_before_max': max(scores_before),
            'score_after_mean': np.mean(scores_after),
            'score_after_max': max(scores_after),
            'score_improvement_mean': np.mean(improvements),
            'score_improvement_max': max(improvements),
            'score_improvement_min': min(improvements),
            'docs_improved': sum(1 for imp in improvements if imp > 0),
            'docs_degraded': sum(1 for imp in improvements if imp < 0),
            'docs_unchanged': sum(1 for imp in improvements if imp == 0)
        }
    
    def _analyze_ranking_changes(
        self,
        docs_before: List[Dict[str, Any]],
        docs_after: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze how the ranking order changed.
        """
        # Create doc ID to rank mappings
        ids_before = [self._get_doc_id(doc) for doc in docs_before]
        ids_after = [self._get_doc_id(doc) for doc in docs_after]
        
        # Build rank mappings
        rank_before = {doc_id: rank for rank, doc_id in enumerate(ids_before)}
        rank_after = {doc_id: rank for rank, doc_id in enumerate(ids_after)}
        
        # Find common documents
        common_ids = set(ids_before) & set(ids_after)
        
        if not common_ids:
            return {'ranking_correlation': 0.0}
        
        # Get ranks for common docs
        ranks_before = [rank_before[doc_id] for doc_id in common_ids]
        ranks_after = [rank_after[doc_id] for doc_id in common_ids]
        
        # Calculate correlation
        if len(ranks_before) > 1:
            spearman_corr, _ = spearmanr(ranks_before, ranks_after)
            kendall_corr, _ = kendalltau(ranks_before, ranks_after)
        else:
            spearman_corr = 1.0
            kendall_corr = 1.0
        
        return {
            'ranking_spearman_correlation': float(spearman_corr),
            'ranking_kendall_correlation': float(kendall_corr),
            'ranking_stability': float(spearman_corr)  # High = stable, Low = changed a lot
        }
    
    def _analyze_position_changes(
        self,
        docs_before: List[Dict[str, Any]],
        docs_after: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze specific position changes (which docs moved where).
        """
        ids_before = [self._get_doc_id(doc) for doc in docs_before]
        ids_after = [self._get_doc_id(doc) for doc in docs_after]
        
        # Track position changes
        position_changes = []
        for i, doc_id in enumerate(ids_after[:5]):  # Top 5 only
            if doc_id in ids_before:
                old_pos = ids_before.index(doc_id)
                new_pos = i
                change = old_pos - new_pos  # Positive = moved up
                position_changes.append(change)
        
        if not position_changes:
            return {}
        
        metrics = {
            'top_5_position_changes_mean': np.mean(position_changes),
            'top_5_position_changes_max': max(position_changes),
            'top_5_moved_up': sum(1 for c in position_changes if c > 0),
            'top_5_moved_down': sum(1 for c in position_changes if c < 0)
        }
        
        # Check if top 3 changed
        top_3_before = set(ids_before[:3])
        top_3_after = set(ids_after[:3])
        metrics['top_3_changed'] = (top_3_before != top_3_after)
        metrics['top_3_overlap'] = len(top_3_before & top_3_after)
        
        return metrics
    
    def _calculate_overall_effectiveness(
        self,
        metrics: Dict[str, Any]
    ) -> float:
        """
        Calculate overall reranking effectiveness score (0-1).
        
        Higher score = reranking is working well.
        """
        score = 0.0
        weights = []
        
        # Factor 1: Score improvement (40%)
        if 'score_improvement_mean' in metrics:
            improvement = metrics['score_improvement_mean']
            # Normalize to 0-1 (assuming improvements typically -0.2 to +0.2)
            improvement_score = min(max((improvement + 0.2) / 0.4, 0), 1)
            score += improvement_score * 0.4
            weights.append(0.4)
        
        # Factor 2: Top docs improved (30%)
        if 'docs_improved' in metrics and 'num_docs_after' in metrics:
            improved_ratio = metrics['docs_improved'] / max(metrics['num_docs_after'], 1)
            score += improved_ratio * 0.3
            weights.append(0.3)
        
        # Factor 3: Top 3 quality (30%)
        if 'score_after_max' in metrics:
            # High max score after reranking = good
            max_score_normalized = min(metrics['score_after_max'], 1.0)
            score += max_score_normalized * 0.3
            weights.append(0.3)
        
        # Normalize by total weights
        if sum(weights) > 0:
            return score / sum(weights)
        return 0.5  # Neutral if we can't calculate
    
    def _get_doc_id(self, doc: Dict[str, Any]) -> str:
        """Extract unique document identifier."""
        metadata = doc.get('metadata', {})
        return (
            metadata.get('source') or 
            metadata.get('file_name') or 
            doc.get('id') or 
            str(hash(doc.get('page_content', '')[:100]))
        )
    
    def _empty_metrics(self, reranking_time_ms: Optional[float]) -> Dict[str, Any]:
        """Return empty metrics."""
        return {
            'reranking_time_ms': reranking_time_ms,
            'reranking_effectiveness_score': 0.0,
            'num_docs_before': 0,
            'num_docs_after': 0
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def evaluate_reranking_effectiveness(
    docs_before_rerank: List[Dict[str, Any]],
    docs_after_rerank: List[Dict[str, Any]],
    reranking_time_ms: Optional[float] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate reranking.
    
    Args:
        docs_before_rerank: Documents before reranking
        docs_after_rerank: Documents after reranking
        reranking_time_ms: Time taken
        
    Returns:
        Metrics dictionary
    """
    evaluator = RerankingEffectivenessEvaluator()
    return evaluator.evaluate_reranking(
        docs_before_rerank=docs_before_rerank,
        docs_after_rerank=docs_after_rerank,
        reranking_time_ms=reranking_time_ms
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Documents before reranking
    docs_before = [
        {
            'metadata': {'source': 'doc1.pdf'},
            'relevance_score': 0.75,
            'page_content': 'GDPR Article 32...'
        },
        {
            'metadata': {'source': 'doc2.pdf'},
            'relevance_score': 0.65,
            'page_content': 'Data protection...'
        },
        {
            'metadata': {'source': 'doc3.pdf'},
            'relevance_score': 0.60,
            'page_content': 'Compliance checklist...'
        },
        {
            'metadata': {'source': 'doc4.pdf'},
            'relevance_score': 0.55,
            'page_content': 'Security measures...'
        },
    ]
    
    # After reranking: doc3 moved to top, doc2 moved down
    docs_after = [
        {
            'metadata': {'source': 'doc3.pdf'},
            'rerank_score': 0.92,
            'page_content': 'Compliance checklist...'
        },
        {
            'metadata': {'source': 'doc1.pdf'},
            'rerank_score': 0.88,
            'page_content': 'GDPR Article 32...'
        },
        {
            'metadata': {'source': 'doc4.pdf'},
            'rerank_score': 0.78,
            'page_content': 'Security measures...'
        },
        {
            'metadata': {'source': 'doc2.pdf'},
            'rerank_score': 0.70,
            'page_content': 'Data protection...'
        },
    ]
    
    metrics = evaluate_reranking_effectiveness(
        docs_before_rerank=docs_before,
        docs_after_rerank=docs_after,
        reranking_time_ms=85.3
    )
    
    logger.info("Reranking Effectiveness Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.3f}")
        elif isinstance(value, bool):
            logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {key}: {value}")
    