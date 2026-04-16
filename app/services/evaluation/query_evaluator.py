# app/services/evaluation/query_evaluator.py
"""
Query Processing Evaluator (Remote Mode)

Evaluates query processing quality:
1. Query expansion effectiveness
2. Query clarity improvement
3. Semantic preservation

*Updated to use Model Server to prevent API memory crashes*
"""

import logging
import time
import numpy as np
import httpx
import random
from typing import Dict, Any, Optional, List
from app.config import settings  # Required for server URLs

logger = logging.getLogger(__name__)

class QueryProcessingEvaluator:
    """
    Evaluate query processing and expansion using Remote Model Server.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize query evaluator in REMOTE mode.
        """
        self.name = "query_processing"
        self.model_name = model_name
        
        # --- NEW: Load Server List from Config ---
        self.server_nodes = settings.model_server_urls_list
        if not self.server_nodes:
            self.server_nodes = ["http://localhost:8074"]
            
        logger.info(f"[QUERY_EVAL] Initialized in REMOTE mode. Targets: {self.server_nodes}")

    def _get_api_url(self) -> str:
        """Pick a random server node and return the eval endpoint."""
        base_url = random.choice(self.server_nodes)
        return f"{base_url.rstrip('/')}/vectorize_eval"

    def _get_embeddings_remote(self, texts: List[str]) -> List[List[float]]:
        """Fetch embeddings from the Model Server."""
        target_url = self._get_api_url()
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(target_url, json={"texts": texts})
                if response.status_code == 200:
                    return response.json().get("embeddings", [])
                else:
                    logger.warning(f"[QUERY_EVAL] Server error {response.status_code}: {response.text}")
                    return []
        except Exception as e:
            logger.error(f"[QUERY_EVAL] Failed to get remote embeddings: {e}")
            return []

    def evaluate_query_processing(
        self,
        original_query: str,
        expanded_query: str,
        processing_time_ms: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate query processing.
        """
        metrics = {
            'processing_time_ms': processing_time_ms,
            'original_length': len(original_query),
            'expanded_length': len(expanded_query),
            'expansion_ratio': len(expanded_query) / max(len(original_query), 1)
        }
        
        # Semantic preservation (Remote)
        semantic_metrics = self._analyze_semantic_preservation(
            original_query,
            expanded_query
        )
        metrics.update(semantic_metrics)
        
        # Clarity improvement (Local Heuristic)
        clarity_metrics = self._analyze_clarity_improvement(
            original_query,
            expanded_query
        )
        metrics.update(clarity_metrics)
        
        # Expansion quality (Score Calculation)
        expansion_quality = self._calculate_expansion_quality(metrics)
        metrics['expansion_quality_score'] = expansion_quality
        
        return metrics
    
    def _analyze_semantic_preservation(
        self,
        original: str,
        expanded: str
    ) -> Dict[str, Any]:
        """
        Check if expansion preserves original meaning using Remote Embeddings.
        """
        try:
            # 1. Get embeddings from Server
            embeddings = self._get_embeddings_remote([original, expanded])
            
            if not embeddings or len(embeddings) < 2:
                return {}

            vec_orig = np.array(embeddings[0])
            vec_exp = np.array(embeddings[1])
            
            # 2. Calculate Cosine Similarity (Locally via Numpy)
            norm_orig = np.linalg.norm(vec_orig)
            norm_exp = np.linalg.norm(vec_exp)
            
            if norm_orig == 0 or norm_exp == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(vec_orig, vec_exp) / (norm_orig * norm_exp))
            
            return {
                'semantic_similarity': similarity,
                'semantic_preserved': similarity >= 0.7  # Threshold
            }
        except Exception as e:
            logger.warning(f"[QUERY_EVAL] Semantic analysis failed: {e}")
            return {}
    
    def _analyze_clarity_improvement(
        self,
        original: str,
        expanded: str
    ) -> Dict[str, Any]:
        """
        Analyze if expansion improved clarity.
        """
        # Check for improvements
        improvements = {
            'added_context': len(expanded) > len(original) * 1.2,
            'proper_capitalization': expanded[0].isupper() if expanded else False,
            'ends_with_punctuation': expanded[-1] in '.?!' if expanded else False,
            'no_typos_detected': len(original.split()) == len(expanded.split()) or 
                                 len(expanded.split()) > len(original.split())
        }
        
        clarity_score = sum(improvements.values()) / len(improvements)
        
        return {
            'clarity_improvements': sum(improvements.values()),
            'clarity_score': float(clarity_score),
            **{f'improvement_{k}': v for k, v in improvements.items()}
        }
    
    def _calculate_expansion_quality(
        self,
        metrics: Dict[str, Any]
    ) -> float:
        """
        Calculate overall expansion quality (0-1).
        """
        score = 0.0
        weights = []
        
        # Factor 1: Semantic preservation (50%)
        if 'semantic_similarity' in metrics:
            similarity = metrics['semantic_similarity']
            score += similarity * 0.5
            weights.append(0.5)
        
        # Factor 2: Clarity improvement (30%)
        if 'clarity_score' in metrics:
            clarity = metrics['clarity_score']
            score += clarity * 0.3
            weights.append(0.3)
        
        # Factor 3: Appropriate expansion (20%)
        if 'expansion_ratio' in metrics:
            ratio = metrics['expansion_ratio']
            if ratio < 1.1:
                expansion_score = ratio 
            elif ratio > 1.5:
                expansion_score = max(0, 2.0 - ratio) 
            else:
                expansion_score = 1.0
            score += expansion_score * 0.2
            weights.append(0.2)
        
        if sum(weights) > 0:
            return score / sum(weights)
        return 0.5


# Convenience function
def evaluate_query_processing(
    original_query: str,
    expanded_query: str,
    processing_time_ms: Optional[float] = None,
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """Convenience function to evaluate query processing."""
    evaluator = QueryProcessingEvaluator(model_name=model_name)
    return evaluator.evaluate_query_processing(
        original_query=original_query,
        expanded_query=expanded_query,
        processing_time_ms=processing_time_ms
    )