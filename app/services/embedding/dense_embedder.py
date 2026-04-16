# app/services/embedding/dense_embedder.py
"""
Layer 3: Dense Embeddings (Client)
Sends text to Model Servers (Load Balanced) to generate BGE-M3 embeddings.
"""

import requests  # REQUIRED: To talk to model_server
import numpy as np
import logging
import random  # <--- NEW: For Load Balancing
from typing import List
from app.config import settings  # <--- NEW: To get server list

logger = logging.getLogger(__name__)

class DenseEmbedder:
    """
    Client-Side Dense Embedder.
    Offloads heavy BGE-M3 inference to the Model Servers (Multi-GPU).
    """
    
    def __init__(self):
        # --- NEW: Load Server List from Config ---
        self.server_nodes = settings.model_server_urls_list
        
        # Fallback if list is empty
        if not self.server_nodes:
            logger.warning("[DenseEmbedder] No model servers found in config! Defaulting to localhost:8074")
            self.server_nodes = ["http://localhost:8074"]

        logger.info("[DenseEmbedder] Initializing Embedding Client configuration.")
        logger.info(f"[DenseEmbedder] Model Server Targets (Load Balanced): {self.server_nodes}")
    
    def _get_api_url(self) -> str:
        """Pick a random server node and return the embed endpoint."""
        selected_node = random.choice(self.server_nodes)
        return f"{selected_node.rstrip('/')}/embed"

    def load_model(self):
        """
        No-op in Client Mode. The server handles initialization.
        """
        logger.info("[DenseEmbedder] Client mode active. No local model loading required.")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense embeddings for multiple documents via Server.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        # --- NEW: Get Dynamic URL ---
        target_url = self._get_api_url()
        logger.info(f"[DenseEmbedder] Sending {len(texts)} text chunks to server ({target_url})...")
        
        try:
            # Prepare payload
            payload = {"texts": texts}
            
            # Send Request
            # Timeout set to 300s (5 mins) for large batches
            response = requests.post(target_url, json=payload, timeout=300)
            
            if response.status_code == 200:
                data = response.json()
                embeddings = data.get("embeddings", [])
                
                # Check for validity
                if len(embeddings) != len(texts):
                    logger.warning(f"[DenseEmbedder] Mismatch: Sent {len(texts)}, received {len(embeddings)} vectors.")
                
                logger.info(f"[DenseEmbedder] Successfully received {len(embeddings)} embeddings.")
                return embeddings
            else:
                logger.error(f"[DenseEmbedder] Server Error ({target_url}): {response.status_code} - {response.text}")
                return []
                
        except requests.exceptions.ConnectionError:
            logger.error(f"[DenseEmbedder] FAILED TO CONNECT to {target_url}. Is it running?")
            return []
        except Exception as e:
            logger.error(f"[DenseEmbedder] Client-side error during request: {e}", exc_info=True)
            return []
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate dense embedding for a single query.
        """
        # Reuse embed_documents for a single item
        logger.info(f"[DenseEmbedder] Embedding query: '{query[:50]}...'")
        results = self.embed_documents([query])
        
        if results:
            return results[0]
        else:
            logger.error("[DenseEmbedder] Failed to get query embedding.")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        # BGE-M3 produces 1024-dimensional embeddings
        return 1024
    
    def compute_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        (Calculated locally since it is lightweight math)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = float(dot_product / (norm1 * norm2))
            return similarity
            
        except Exception as e:
            logger.info(f"[DenseEmbedder] Error computing similarity: {e}")
            return 0.0
    
    def clear_memory(self):
        """No-op in client mode."""
        logger.info("[DenseEmbedder] Memory clear requested (No-op in Client Mode).")