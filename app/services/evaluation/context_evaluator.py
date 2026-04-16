# app/services/evaluation/context_evaluator.py
"""
Context Quality Evaluator (Remote Mode)

Evaluates the quality of assembled context before generation:
1. Context relevance to query
2. Redundancy detection (duplicate information)
3. Coverage (does context have enough info to answer?)
4. Coherence (do docs fit together logically?)
5. Token efficiency (information density)

*Updated to use Remote Model Server to prevent API memory crashes*
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import re
from collections import Counter

# --- NEW IMPORTS FOR REMOTE SERVER ---
import httpx
import random
from app.config import settings

logger = logging.getLogger(__name__)

class ContextQualityEvaluator:
    """
    Evaluate assembled context quality using Remote Model Server.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize context evaluator in REMOTE mode.
        """
        self.name = "context_quality"
        self.model_name = model_name
        
        # --- NEW: Load Server List from Config ---
        self.server_nodes = settings.model_server_urls_list
        if not self.server_nodes:
            self.server_nodes = ["http://localhost:8074"]
            
        logger.info(f"[CONTEXT_EVAL] Initialized in REMOTE mode. Targets: {self.server_nodes}")
    
    # --- HELPER: Get Dynamic URL ---
    def _get_api_url(self) -> str:
        """Pick a random server node."""
        base_url = random.choice(self.server_nodes)
        return f"{base_url.rstrip('/')}/vectorize_eval"

    # --- HELPER: Get Embeddings Remotely ---
    def _get_embeddings_remote(self, texts: List[str]) -> List[List[float]]:
        """Fetch embeddings from the Model Server."""
        if not texts: return []
        
        target_url = self._get_api_url()
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(target_url, json={"texts": texts})
                if response.status_code == 200:
                    return response.json().get("embeddings", [])
                else:
                    logger.warning(f"[CONTEXT_EVAL] Server error {response.status_code}: {response.text}")
                    return []
        except Exception as e:
            logger.error(f"[CONTEXT_EVAL] Failed to get remote embeddings: {e}")
            return []

    # --- HELPER: Numpy Cosine Similarity ---
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between vector A (1D) and matrix B (2D),
        or matrix A (2D) and matrix B (2D).
        """
        # Ensure inputs are at least 2D for consistent matrix math
        if a.ndim == 1:
            a = a[np.newaxis, :]
        if b.ndim == 1:
            b = b[np.newaxis, :]

        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True)
        
        # Avoid division by zero
        if np.any(norm_a == 0) or np.any(norm_b == 0):
            return np.zeros((a.shape[0], b.shape[0]))

        return np.dot(a, b.T) / (norm_a * norm_b.T)

    def evaluate_context(
        self,
        query: str,
        context: str,
        source_documents: Optional[List[Dict[str, Any]]] = None,
        context_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate context quality.
        """
        if not context:
            return self._empty_metrics()
        
        metrics = {
            'context_length': len(context),
            'context_tokens': context_tokens or self._estimate_tokens(context),
            'num_source_docs': len(source_documents) if source_documents else 0
        }
        
        # Relevance to query (Uses Remote Model)
        relevance = self._calculate_relevance(query, context)
        metrics.update(relevance)
        
        # Redundancy analysis (Uses Remote Model)
        redundancy = self._analyze_redundancy(context, source_documents)
        metrics.update(redundancy)
        
        # Coverage analysis (Logic only)
        coverage = self._analyze_coverage(query, context)
        metrics.update(coverage)
        
        # Coherence (Uses Remote Model)
        if source_documents and len(source_documents) > 1:
            coherence = self._analyze_coherence(source_documents)
            metrics.update(coherence)
        
        # Token efficiency (Logic only)
        efficiency = self._analyze_token_efficiency(context, metrics)
        metrics.update(efficiency)
        
        # Overall quality score
        overall = self._calculate_overall_quality(metrics)
        metrics['context_quality_score'] = overall
        
        return metrics
    
    def _calculate_relevance(
        self,
        query: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Calculate semantic relevance of context to query.
        """
        try:
            # Split context into chunks for analysis
            context_chunks = self._split_context(context)
            if not context_chunks:
                return {}
            
            # --- REMOTE CALL: Encode Query + Chunks in one batch ---
            all_texts = [query] + context_chunks
            embeddings = self._get_embeddings_remote(all_texts)
            
            if not embeddings:
                return {}

            query_emb = np.array(embeddings[0])
            chunk_embs = np.array(embeddings[1:])
            
            # Calculate similarities using Numpy
            similarities = self._cosine_similarity(query_emb, chunk_embs).flatten()
            
            return {
                'context_relevance_mean': float(np.mean(similarities)),
                'context_relevance_max': float(np.max(similarities)),
                'context_relevance_min': float(np.min(similarities)),
                'context_relevance_std': float(np.std(similarities)),
                'highly_relevant_chunks': int(np.sum(similarities > 0.7)),
                'relevant_chunks': int(np.sum(similarities > 0.5)),
                'irrelevant_chunks': int(np.sum(similarities < 0.3))
            }
        except Exception as e:
            logger.warning(f"[CONTEXT_EVAL] Relevance calculation failed: {e}")
            return {}
    
    def _analyze_redundancy(
        self,
        context: str,
        source_documents: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Analyze redundancy (duplicate information) in context.
        """
        # Method 1: N-gram overlap
        ngram_redundancy = self._calculate_ngram_overlap(context)
        
        # Method 2: Sentence-level similarity
        if source_documents:
            sentence_redundancy = self._calculate_sentence_redundancy(source_documents)
        else:
            sentence_redundancy = {}
        
        return {
            **ngram_redundancy,
            **sentence_redundancy
        }
    
    def _calculate_ngram_overlap(
        self,
        context: str,
        n: int = 3
    ) -> Dict[str, Any]:
        """
        Calculate n-gram overlap to detect repeated phrases.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return {'ngram_redundancy_score': 0.0}
        
        # Extract n-grams from each sentence
        all_ngrams = []
        for sent in sentences:
            words = sent.lower().split()
            if len(words) >= n:
                ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
                all_ngrams.extend(ngrams)
        
        if not all_ngrams:
            return {'ngram_redundancy_score': 0.0}
        
        # Count duplicates
        ngram_counts = Counter(all_ngrams)
        duplicates = sum(count - 1 for count in ngram_counts.values() if count > 1)
        redundancy_ratio = duplicates / len(all_ngrams) if all_ngrams else 0.0
        
        return {
            'ngram_redundancy_score': float(redundancy_ratio),
            'duplicate_ngrams': duplicates,
            'unique_ngrams': len(ngram_counts)
        }
    
    def _calculate_sentence_redundancy(
        self,
        source_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate semantic redundancy between documents.
        """
        if len(source_documents) < 2:
            return {}
        
        try:
            # Extract text from each document
            doc_texts = []
            for doc in source_documents[:5]:  # Limit to 5 for performance
                text = doc.get('page_content', '')[:500]  # First 500 chars
                if text:
                    doc_texts.append(text)
            
            if len(doc_texts) < 2:
                return {}
            
            # --- REMOTE CALL: Encode documents ---
            doc_embs = np.array(self._get_embeddings_remote(doc_texts))
            
            if len(doc_embs) == 0:
                return {}
            
            # Calculate pairwise similarities
            similarities = self._cosine_similarity(doc_embs, doc_embs)
            
            # Get upper triangle (exclude diagonal)
            upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
            
            if len(upper_triangle) == 0:
                return {}
            
            return {
                'semantic_redundancy_mean': float(np.mean(upper_triangle)),
                'semantic_redundancy_max': float(np.max(upper_triangle)),
                'highly_similar_pairs': int(np.sum(upper_triangle > 0.8))
            }
        except Exception as e:
            logger.warning(f"[CONTEXT_EVAL] Semantic redundancy failed: {e}")
            return {}
    
    def _analyze_coverage(
        self,
        query: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Analyze if context has enough information to answer query.
        """
        # Extract key terms from query
        query_terms = self._extract_key_terms(query)
        
        # Check presence in context
        context_lower = context.lower()
        terms_found = sum(1 for term in query_terms if term in context_lower)
        
        coverage_ratio = terms_found / len(query_terms) if query_terms else 0.0
        
        return {
            'coverage_score': float(coverage_ratio),
            'query_terms_found': terms_found,
            'query_terms_total': len(query_terms),
            'coverage_sufficient': coverage_ratio >= 0.7
        }
    
    def _analyze_coherence(
        self,
        source_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze if documents fit together coherently.
        """
        if len(source_documents) < 2:
            return {}
        
        try:
            # Similar to redundancy, but we want SOME overlap (not too much, not too little)
            doc_texts = [
                doc.get('page_content', '')[:500] 
                for doc in source_documents[:5]
            ]
            doc_texts = [t for t in doc_texts if t]
            
            if len(doc_texts) < 2:
                return {}
            
            # --- REMOTE CALL: Encode docs ---
            doc_embs = np.array(self._get_embeddings_remote(doc_texts))
            
            if len(doc_embs) == 0:
                return {}

            similarities = self._cosine_similarity(doc_embs, doc_embs)
            upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
            
            avg_similarity = float(np.mean(upper_triangle))
            
            # Ideal coherence: 0.3-0.6 similarity
            # Too low = unrelated, too high = redundant
            if avg_similarity < 0.3:
                coherence_score = avg_similarity / 0.3  # 0-1 scale
            elif avg_similarity > 0.6:
                coherence_score = 1.0 - (avg_similarity - 0.6) / 0.4  # Penalty for redundancy
            else:
                coherence_score = 1.0  # Perfect range
            
            return {
                'coherence_score': float(max(0, min(1, coherence_score))),
                'inter_doc_similarity': avg_similarity,
                'docs_well_connected': avg_similarity >= 0.3 and avg_similarity <= 0.6
            }
        except Exception as e:
            logger.warning(f"[CONTEXT_EVAL] Coherence analysis failed: {e}")
            return {}
    
    def _analyze_token_efficiency(
        self,
        context: str,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze token efficiency (information density).
        """
        tokens = metrics.get('context_tokens', self._estimate_tokens(context))
        
        # Information density = unique content / total tokens
        words = context.lower().split()
        unique_words = len(set(words))
        
        density = unique_words / len(words) if words else 0.0
        
        # Tokens per document
        num_docs = metrics.get('num_source_docs', 1)
        tokens_per_doc = tokens / num_docs if num_docs > 0 else tokens
        
        return {
            'token_efficiency': float(density),
            'unique_words': unique_words,
            'total_words': len(words),
            'tokens_per_document': float(tokens_per_doc)
        }
    
    def _calculate_overall_quality(
        self,
        metrics: Dict[str, Any]
    ) -> float:
        """
        Calculate overall context quality score (0-1).
        """
        score = 0.0
        weights = []
        
        # Factor 1: Relevance (40%)
        if 'context_relevance_mean' in metrics:
            relevance = metrics['context_relevance_mean']
            score += relevance * 0.4
            weights.append(0.4)
        
        # Factor 2: Coverage (30%)
        if 'coverage_score' in metrics:
            coverage = metrics['coverage_score']
            score += coverage * 0.3
            weights.append(0.3)
        
        # Factor 3: Low redundancy (15%)
        if 'ngram_redundancy_score' in metrics:
            # Invert redundancy (lower is better)
            redundancy = metrics['ngram_redundancy_score']
            non_redundancy = 1.0 - min(redundancy, 1.0)
            score += non_redundancy * 0.15
            weights.append(0.15)
        
        # Factor 4: Coherence (15%)
        if 'coherence_score' in metrics:
            coherence = metrics['coherence_score']
            score += coherence * 0.15
            weights.append(0.15)
        
        # Normalize
        if sum(weights) > 0:
            return score / sum(weights)
        return 0.5  # Neutral
    
    def _split_context(self, context: str, chunk_size: int = 200) -> List[str]:
        """Split context into chunks for analysis."""
        words = context.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query."""
        # Remove common words
        stopwords = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'the', 'a', 'an'}
        words = query.lower().split()
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (1 token ≈ 0.75 words)."""
        words = len(text.split())
        return int(words / 0.75)
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics."""
        return {
            'context_length': 0,
            'context_tokens': 0,
            'context_quality_score': 0.0
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def evaluate_context_quality(
    query: str,
    context: str,
    source_documents: Optional[List[Dict[str, Any]]] = None,
    context_tokens: Optional[int] = None,
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Convenience function to evaluate context quality.
    
    Args:
        query: Search query
        context: Assembled context
        source_documents: Source documents
        context_tokens: Token count
        model_name: Sentence transformer model
        
    Returns:
        Metrics dictionary
    """
    evaluator = ContextQualityEvaluator(model_name=model_name)
    return evaluator.evaluate_context(
        query=query,
        context=context,
        source_documents=source_documents,
        context_tokens=context_tokens
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    query = "What are GDPR data retention requirements?"
    
    context = """
    GDPR Article 5 states that personal data shall be kept for no longer than 
    is necessary for the purposes for which it is processed. Organizations must 
    define clear retention periods.
    
    Article 17 provides the right to erasure, requiring organizations to delete 
    personal data when it is no longer needed.
    
    Data retention policies should be documented and regularly reviewed to ensure
    compliance with GDPR requirements.
    """
    
    source_docs = [
        {'page_content': 'GDPR Article 5 states...'},
        {'page_content': 'Article 17 provides the right to erasure...'},
        {'page_content': 'Data retention policies should be documented...'}
    ]
    
    metrics = evaluate_context_quality(
        query=query,
        context=context,
        source_documents=source_docs,
        context_tokens=100
    )
    
    logger.info("Context Quality Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.3f}")
        elif isinstance(value, bool):
            logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {key}: {value}")