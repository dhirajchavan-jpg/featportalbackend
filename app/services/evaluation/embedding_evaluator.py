# app/services/evaluation/embedding_evaluator.py
"""
Embedding Quality Evaluator

Evaluates embedding model quality using:
1. MTEB (Massive Text Embedding Benchmark) scores
2. Embedding similarity analysis
3. Semantic coherence metrics

Note: MTEB evaluation is compute-intensive and typically done offline
for model comparison. For production, we use lighter proxy metrics.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

logger = logging.getLogger(__name__)


class EmbeddingQualityEvaluator:
    """
    Evaluate embedding quality in production.
    
    For full MTEB evaluation (offline), use the mteb library separately.
    This class focuses on production-ready metrics.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding evaluator.
        
        Args:
            model_name: Optional model name for similarity checks
        """
        self.name = "embedding_quality"
        self.model_name = model_name
        self.similarity_model = None
        
        # Load a lightweight model for similarity checks if needed
        if model_name:
            try:
                self.similarity_model = SentenceTransformer(model_name)
                logger.info(f"[EMBEDDING_EVAL] Loaded similarity model: {model_name}")
            except Exception as e:
                logger.warning(f"[EMBEDDING_EVAL] Could not load model: {e}")
        
        logger.info("[EMBEDDING_EVAL] Initialized")
    
    def evaluate_embeddings(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        doc_embeddings: Optional[List[np.ndarray]] = None,
        doc_texts: Optional[List[str]] = None,
        embedding_time_ms: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate embedding quality for a query.
        
        Args:
            query: The search query
            query_embedding: Query embedding vector (optional)
            doc_embeddings: Document embedding vectors (optional)
            doc_texts: Document texts (optional)
            embedding_time_ms: Time taken to generate embeddings
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'evaluation_method': 'production_proxy',
            'embedding_time_ms': embedding_time_ms
        }
        
        # Basic embedding statistics
        if query_embedding is not None:
            metrics.update(self._analyze_embedding_vector(query_embedding, prefix='query'))
        
        if doc_embeddings:
            # Analyze document embeddings
            for i, emb in enumerate(doc_embeddings[:5]):  # First 5 only
                stats = self._analyze_embedding_vector(emb, prefix=f'doc_{i}')
                # Only keep summary stats for docs
                metrics[f'doc_{i}_norm'] = stats.get(f'doc_{i}_norm')
        
        # Similarity analysis (if we have both query and doc embeddings)
        if query_embedding is not None and doc_embeddings:
            sim_metrics = self._analyze_similarities(query_embedding, doc_embeddings)
            metrics.update(sim_metrics)
        
        # Semantic coherence (if we have model and texts)
        if self.similarity_model and doc_texts:
            coherence = self._analyze_semantic_coherence(query, doc_texts)
            metrics.update(coherence)
        
        return metrics
    
    def _analyze_embedding_vector(
        self, 
        embedding: np.ndarray, 
        prefix: str = 'embedding'
    ) -> Dict[str, Any]:
        """
        Analyze a single embedding vector.
        
        Returns statistics about the embedding quality.
        """
        embedding = np.array(embedding)
        
        return {
            f'{prefix}_dimension': len(embedding),
            f'{prefix}_norm': float(np.linalg.norm(embedding)),
            f'{prefix}_mean': float(np.mean(embedding)),
            f'{prefix}_std': float(np.std(embedding)),
            f'{prefix}_min': float(np.min(embedding)),
            f'{prefix}_max': float(np.max(embedding)),
            f'{prefix}_sparsity': float(np.sum(np.abs(embedding) < 0.01) / len(embedding))
        }
    
    def _analyze_similarities(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Analyze similarity distribution between query and documents.
        """
        query_emb = np.array(query_embedding)
        
        similarities = []
        for doc_emb in doc_embeddings:
            doc_emb = np.array(doc_emb)
            # Cosine similarity
            sim = np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
            )
            similarities.append(float(sim))
        
        if not similarities:
            return {}
        
        return {
            'similarity_max': max(similarities),
            'similarity_min': min(similarities),
            'similarity_mean': np.mean(similarities),
            'similarity_std': np.std(similarities),
            'similarity_range': max(similarities) - min(similarities),
            'high_similarity_count': sum(1 for s in similarities if s > 0.7),
            'medium_similarity_count': sum(1 for s in similarities if 0.5 <= s <= 0.7),
            'low_similarity_count': sum(1 for s in similarities if s < 0.5)
        }
    
    def _analyze_semantic_coherence(
        self,
        query: str,
        doc_texts: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze semantic coherence using sentence transformers.
        
        This measures how semantically similar the retrieved documents are
        to each other and to the query.
        """
        if not self.similarity_model or not doc_texts:
            return {}
        
        try:
            # Encode query and documents
            query_emb = self.similarity_model.encode(query, convert_to_tensor=True)
            doc_embs = self.similarity_model.encode(doc_texts[:5], convert_to_tensor=True)
            
            # Query-document similarities
            query_doc_sims = util.cos_sim(query_emb, doc_embs)[0].cpu().numpy()
            
            # Inter-document similarities
            doc_doc_sims = util.cos_sim(doc_embs, doc_embs).cpu().numpy()
            # Get upper triangle (excluding diagonal)
            inter_doc_sims = doc_doc_sims[np.triu_indices_from(doc_doc_sims, k=1)]
            
            return {
                'semantic_query_doc_mean': float(np.mean(query_doc_sims)),
                'semantic_query_doc_max': float(np.max(query_doc_sims)),
                'semantic_inter_doc_mean': float(np.mean(inter_doc_sims)) if len(inter_doc_sims) > 0 else 0.0,
                'semantic_coherence_score': float(np.mean(query_doc_sims))  # High = good retrieval
            }
        except Exception as e:
            logger.warning(f"[EMBEDDING_EVAL] Semantic analysis failed: {e}")
            return {}
    
    def evaluate_mteb_offline(
        self,
        model_name: str,
        tasks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run MTEB benchmark offline (for model comparison).
        
        WARNING: This is VERY compute-intensive and should be run separately,
        not in production pipeline.
        
        Args:
            model_name: Model to evaluate
            tasks: Optional list of MTEB tasks to run
            
        Returns:
            MTEB scores
        """
        try:
            from mteb import MTEB
            
            # Default to retrieval tasks
            if tasks is None:
                tasks = [
                    "NFCorpus",           # Medical retrieval
                    "SciFact",            # Scientific fact verification
                    "ArguAna",            # Argument retrieval
                    "TRECCOVID",          # COVID-19 retrieval
                ]
            
            logger.info(f"[MTEB] Starting evaluation for {model_name} on tasks: {tasks}")
            logger.warning("[MTEB] This will take a long time (hours)!")
            
            # Load model
            model = SentenceTransformer(model_name)
            
            # Run evaluation
            evaluation = MTEB(tasks=tasks)
            results = evaluation.run(model, output_folder=f"mteb_results/{model_name}")
            
            logger.info(f"[MTEB] Evaluation complete. Results: {results}")
            
            return results
            
        except ImportError:
            logger.error("[MTEB] mteb package not installed. Install with: pip install mteb")
            return {"error": "mteb not installed"}
        except Exception as e:
            logger.error(f"[MTEB] Evaluation failed: {e}")
            return {"error": str(e)}


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def evaluate_embedding_quality(
    query: str,
    query_embedding: Optional[np.ndarray] = None,
    doc_embeddings: Optional[List[np.ndarray]] = None,
    doc_texts: Optional[List[str]] = None,
    embedding_time_ms: Optional[float] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate embedding quality.
    
    Args:
        query: Search query
        query_embedding: Query embedding vector
        doc_embeddings: Document embeddings
        doc_texts: Document texts
        embedding_time_ms: Time taken
        model_name: Model name for semantic analysis
        
    Returns:
        Metrics dictionary
    """
    evaluator = EmbeddingQualityEvaluator(model_name=model_name)
    return evaluator.evaluate_embeddings(
        query=query,
        query_embedding=query_embedding,
        doc_embeddings=doc_embeddings,
        doc_texts=doc_texts,
        embedding_time_ms=embedding_time_ms
    )


# ============================================================================
# MTEB REFERENCE SCORES (for comparison)
# ============================================================================

MTEB_REFERENCE_SCORES = {
    # Top models from MTEB leaderboard (as of Dec 2024)
    "bge-large-en-v1.5": {
        "avg_score": 0.654,
        "retrieval_avg": 0.533,
        "description": "Good general-purpose model"
    },
    "bge-m3": {
        "avg_score": 0.665,
        "retrieval_avg": 0.542,
        "description": "Multilingual model"
    },
    "gte-large": {
        "avg_score": 0.648,
        "retrieval_avg": 0.528,
        "description": "Alibaba model"
    },
    "e5-large-v2": {
        "avg_score": 0.641,
        "retrieval_avg": 0.520,
        "description": "Microsoft model"
    }
}


def get_mteb_reference_score(model_name: str) -> Optional[Dict[str, Any]]:
    """Get reference MTEB scores for a model."""
    return MTEB_REFERENCE_SCORES.get(model_name)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Evaluate embeddings in production
    query = "What are GDPR requirements?"
    
    # Simulated embeddings (in practice, these come from your embedding model)
    query_emb = np.random.randn(1024)  # 1024-dim embedding
    doc_embs = [np.random.randn(1024) for _ in range(5)]
    doc_texts = [
        "GDPR requires data protection by design",
        "Article 32 specifies security measures",
        "Unrelated document about weather",
        "GDPR compliance checklist for organizations",
        "Data retention policies under GDPR"
    ]
    
    metrics = evaluate_embedding_quality(
        query=query,
        query_embedding=query_emb,
        doc_embeddings=doc_embs,
        doc_texts=doc_texts,
        embedding_time_ms=45.2,
        model_name="all-MiniLM-L6-v2"  # Lightweight model for demo
    )
    
    logger.info("Embedding Quality Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.3f}")
        else:
            logger.info(f"  {key}: {value}")