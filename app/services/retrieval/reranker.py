# app/services/retrieval/reranker.py
"""
Layer 4: BGE Reranker (Client)
Sends documents to Model Servers (Load Balanced) for reranking.
"""

import requests  # REQUIRED: To talk to model_server
import logging
import random # <--- NEW: For Load Balancing
from typing import List, Dict, Any
from app.config import settings

# --- FIX: STANDARD OPENTELEMETRY ---
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class BGEReranker:
    """
    Client-Side BGE Reranker.
    Offloads heavy Cross-Encoder processing to the Model Servers (Multi-GPU).
    """
    
    def __init__(self):
        # --- NEW: Load Server List from Config ---
        self.server_nodes = settings.model_server_urls_list
        
        # Fallback if list is empty
        if not self.server_nodes:
            logger.warning("[BGEReranker] No model servers found in config! Defaulting to localhost:8074")
            self.server_nodes = ["http://localhost:8074"]

        logger.info("[BGEReranker] Initializing Reranker Client...")
        logger.info(f"[BGEReranker] Load Balancing Target Nodes: {self.server_nodes}")
        
        # Dummy flag to satisfy "if self.model is None" checks in hybrid_retriever
        self.model = True 
        logger.info("[BGEReranker] Initialization complete (Client Mode).")
    
    def _get_api_url(self) -> str:
        """Pick a random server node and return the rerank endpoint."""
        selected_node = random.choice(self.server_nodes)
        return f"{selected_node.rstrip('/')}/rerank"

    def load_model(self):
        """No-op in Client Mode. Server handles loading."""
        logger.info("[BGEReranker] load_model called. Client mode active - no local model loading required.")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Send documents to a random Model Server for reranking.
        """
        top_k = top_k or settings.RERANK_TOP_K
        
        # --- START TRACE ---
        with tracer.start_as_current_span("BGE Reranking (Client)") as span:
            span.set_attribute("openinference.span.kind", "RERANKER")
            span.set_attribute("input.value", str(documents[:3]))
            
            logger.info(f"[BGEReranker] Starting rerank for query: '{query[:30]}...'")
            
            if not documents:
                logger.warning("[BGEReranker] No documents provided to rerank. Returning empty list.")
                return []
            
            # --- NEW: Get Dynamic URL ---
            target_url = self._get_api_url()
            logger.info(f"[BGEReranker] Preparing to send {len(documents)} documents to {target_url} (Top K target: {top_k}).")
            
            try:
                # 1. Prepare Payload
                # Ensure content exists (fallback to text or page_content if missing)
                cleaned_docs = []
                for i, doc in enumerate(documents):
                    content = doc.get('content') or doc.get('page_content') or ""
                    # We send the whole doc struct so the server can return it back sorted
                    doc['content'] = content 
                    cleaned_docs.append(doc)

                payload = {
                    "query": query,
                    "documents": cleaned_docs,
                    "top_k": top_k
                }
                
                # 2. Send Request
                logger.info(f"[BGEReranker] Sending POST request to {target_url}...")
                # Timeout 60s is usually enough for reranking batches
                response = requests.post(target_url, json=payload, timeout=60)
                
                if response.status_code == 200:
                    logger.info("[BGEReranker] Server responded with 200 OK.")
                    data = response.json()
                    reranked_results = data.get("results", [])
                    
                    logger.info(f"[BGEReranker] Received {len(reranked_results)} reranked docs from server.")
                    
                    # Update ranks locally for stats consistency
                    for rank, doc in enumerate(reranked_results, start=1):
                        doc['final_rank'] = rank
                    
                    top_score = reranked_results[0].get('rerank_score') if reranked_results else 0
                    span.set_attribute("output.value", f"Top score: {top_score}")
                    
                    logger.info("[BGEReranker] Reranking successful.")
                    return reranked_results
                else:
                    logger.error(f"[BGEReranker] Server Error ({target_url}): {response.status_code} - {response.text}")
                    # Fallback: Return original docs sliced if server fails
                    logger.info("[BGEReranker] Falling back to original document order.")
                    return documents[:top_k]
                    
            except requests.exceptions.ConnectionError:
                logger.error(f"[BGEReranker] FAILED TO CONNECT to {target_url}. Is it running?")
                return documents[:top_k]
            except Exception as e:
                logger.error(f"[BGEReranker] Client-side error: {e}", exc_info=True)
                return documents[:top_k]

    def batch_rerank(self, query: str, documents: List[Dict], batch_size: int = 16, top_k: int = None):
        """
        Proxy for rerank (Server handles batching internally if needed).
        """
        logger.info(f"[BGEReranker] batch_rerank called for {len(documents)} docs. Delegating to main rerank method.")
        # For the client, we just send the whole list. The server logic processes it efficiently.
        return self.rerank(query, documents, top_k)

    def get_reranking_stats(self, reranked_results: List[Dict]) -> Dict[str, Any]:
        """Get statistics about reranking."""
        logger.info("[BGEReranker] Calculating reranking statistics...")
        if not reranked_results:
            return {}
        
        rerank_scores = [r.get('rerank_score', 0.0) for r in reranked_results]
        
        # Calculate stats safely
        avg_score = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0.0
        max_score = max(rerank_scores) if rerank_scores else 0.0
        min_score = min(rerank_scores) if rerank_scores else 0.0
        
        stats = {
            'total_reranked': len(reranked_results),
            'avg_rerank_score': avg_score,
            'max_rerank_score': max_score,
            'min_rerank_score': min_score
        }
        logger.info(f"[BGEReranker] Stats calculated: {stats}")
        return stats
    
    def explain_reranking(self, result: Dict[str, Any]) -> str:
        """Generate explanation for reranking."""
        explanation = []
        
        if 'final_rank' in result:
            explanation.append(f"Final Rank: {result['final_rank']}")
        if 'rerank_score' in result:
            explanation.append(f"Rerank Score: {result['rerank_score']:.4f}")
        
        if 'rrf_rank' in result and 'final_rank' in result:
            rank_change = result['rrf_rank'] - result['final_rank']
            if rank_change > 0:
                explanation.append(f"Moved UP {rank_change} positions from RRF")
            elif rank_change < 0:
                explanation.append(f"Moved DOWN {abs(rank_change)} positions from RRF")
            else:
                explanation.append("Rank unchanged from RRF")
        
        return " | ".join(explanation) if explanation else "No reranking info"

# Singleton instance
_reranker = None

def get_reranker() -> BGEReranker:
    global _reranker
    if _reranker is None:
        logger.info("[BGEReranker] Creating new singleton instance.")
        _reranker = BGEReranker()
    else:
        logger.info("[BGEReranker] Returning existing singleton instance.")
    return _reranker