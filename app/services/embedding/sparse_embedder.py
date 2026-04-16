# app/services/embedding/sparse_embedder.py
"""
Layer 3: Sparse Embeddings using BM25
Provides keyword-based scoring for lexical search.
OPTIMIZED: Uses IDF weighting for queries and removed redundant score calculation.
"""

import pickle
import os
import re
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SparseEmbedder:
    """
    BM25 Sparse Embedding Generator.
    Creates sparse vectors mapping Vocabulary Indices -> IDF Weights.
    """
    
    def __init__(self):
        logger.info("[SparseEmbedder] Initializing SparseEmbedder...")
        self.bm25_index = None
        self.corpus = []
        self.tokenized_corpus = []
        self.vocab = {}
        self.vocab_size = 0
        logger.info("[SparseEmbedder] Initialization complete.")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Robust Regex Tokenization.
        Keeps alphanumeric words, lowercases, ignores short tokens.
        """
        # \w+ matches any word character (alphanumeric + underscore)
        # logger.info(f"[SparseEmbedder] Tokenizing text of length {len(text)}...") # Commented out to reduce spam
        tokens = re.findall(r'\w+', text.lower())
        result = [t for t in tokens if len(t) > 2]
        return result
    
    def build_index(self, texts: List[str]) -> None:
        """
        Build BM25 index from corpus.
        """
        try:
            logger.info(f"[SparseEmbedder] Starting build_index for {len(texts)} documents.")
            
            self.corpus = texts
            logger.info("[SparseEmbedder] Tokenizing corpus...")
            self.tokenized_corpus = [self._tokenize(text) for text in texts]
            logger.info(f"[SparseEmbedder] Tokenization complete. {len(self.tokenized_corpus)} documents processed.")
            
            # Build BM25 index
            logger.info("[SparseEmbedder] Constructing BM25Okapi index...")
            self.bm25_index = BM25Okapi(self.tokenized_corpus)
            logger.info("[SparseEmbedder] BM25Okapi construction finished.")
            
            # Build vocabulary map (Token -> Integer ID)
            logger.info("[SparseEmbedder] Building vocabulary map...")
            self._build_vocabulary()
            
            logger.info(f"[SparseEmbedder] Index built successfully. Vocab size: {self.vocab_size}")
            
        except Exception as e:
            logger.error(f"[SparseEmbedder] Error building index: {e}", exc_info=True)
            raise
    
    def _build_vocabulary(self):
        """Build vocabulary from tokenized corpus."""
        unique_tokens = set()
        for doc_tokens in self.tokenized_corpus:
            unique_tokens.update(doc_tokens)
        
        count = len(unique_tokens)
        logger.info(f"[SparseEmbedder] Found {count} unique tokens. Sorting and assigning IDs...")
        
        # Sort to ensure deterministic IDs
        self.vocab = {token: idx for idx, token in enumerate(sorted(unique_tokens))}
        self.vocab_size = len(self.vocab)
        logger.info(f"[SparseEmbedder] Vocabulary built. Total size: {self.vocab_size}")
    
    def get_sparse_embedding(self, text: str) -> Dict[int, float]:
        """
        Generate sparse embedding for Qdrant Query.
        
        Uses IDF (Inverse Document Frequency) weighting.
        If a word appears in the query, its value is its importance (IDF).
        """
        if self.bm25_index is None:
            logger.warning("[SparseEmbedder] WARN: BM25 index is not loaded. Returning empty embedding.")
            return {}
        
        try:
            # logger.info(f"[SparseEmbedder] Generating sparse embedding for text: '{text[:30]}...'")
            tokens = self._tokenize(text)
            sparse_vec = {}
            
            found_tokens = 0
            for token in tokens:
                if token in self.vocab:
                    found_tokens += 1
                    idx = self.vocab[token]
                    
                    # LOGIC UPDATE: Use IDF instead of raw count
                    # This boosts rare words (like "cheque") and lowers common words (like "the")
                    # We default to 0.5 for tokens found in vocab but somehow missing IDF (rare edge case)
                    idf_score = self.bm25_index.idf.get(token, 0.5)
                    
                    # If word appears twice in query, boost it (IDF * Count)
                    sparse_vec[idx] = sparse_vec.get(idx, 0.0) + idf_score
            
            # logger.info(f"[SparseEmbedder] Sparse vector generated. {found_tokens}/{len(tokens)} tokens found in vocab.")
            return sparse_vec
            
        except Exception as e:
            logger.error(f"[SparseEmbedder] Error generating sparse embedding: {e}", exc_info=True)
            return {}
    
    def save_index(self, filepath: str) -> None:
        """Save BM25 index to disk."""
        try:
            logger.info(f"[SparseEmbedder] Saving index to disk: {filepath}")
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            index_data = {
                'bm25_index': self.bm25_index,
                'corpus': self.corpus,
                'tokenized_corpus': self.tokenized_corpus,
                'vocab': self.vocab,
                'vocab_size': self.vocab_size
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(index_data, f)
            
            logger.info(f"[SparseEmbedder] Index successfully saved to {filepath}")
            
        except Exception as e:
            logger.error(f"[SparseEmbedder] Error saving index: {e}", exc_info=True)
    
    def load_index(self, filepath: str) -> None:
        """Load BM25 index from disk."""
        try:
            logger.info(f"[SparseEmbedder] Attempting to load index from: {filepath}")
            if not os.path.exists(filepath):
                logger.warning(f"[SparseEmbedder] Index file not found: {filepath}")
                # Don't raise, just return. The provider will handle the None check.
                return
            
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
            
            self.bm25_index = index_data['bm25_index']
            self.corpus = index_data['corpus']
            self.tokenized_corpus = index_data['tokenized_corpus']
            self.vocab = index_data['vocab']
            self.vocab_size = index_data['vocab_size']
            
            logger.info(f"[SparseEmbedder] Index loaded successfully from {filepath} (Vocab: {self.vocab_size})")
            
        except Exception as e:
            logger.error(f"[SparseEmbedder] Error loading index: {e}", exc_info=True)
            # Ensure we don't leave half-loaded state
            self.bm25_index = None

    def update_index(self, new_texts: List[str]) -> None:
        """
        Update BM25 index with new documents.
        NOTE: BM25Okapi requires full rebuild to recalculate IDFs correctly.
        """
        if not new_texts:
            logger.info("[SparseEmbedder] update_index called with empty list. Skipping.")
            return
        
        logger.info(f"[SparseEmbedder] Updating index with {len(new_texts)} new documents...")
        
        # Append to corpus
        self.corpus.extend(new_texts)
        
        logger.info("[SparseEmbedder] Tokenizing new documents...")
        new_tokenized = [self._tokenize(text) for text in new_texts]
        self.tokenized_corpus.extend(new_tokenized)
        
        # Rebuild index
        logger.info("[SparseEmbedder] Rebuilding BM25Okapi index...")
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        
        # Rebuild vocabulary (New words might have appeared)
        logger.info("[SparseEmbedder] Rebuilding vocabulary...")
        self._build_vocabulary()
        
        logger.info(f"[SparseEmbedder] Index updated. New vocab size: {self.vocab_size}")