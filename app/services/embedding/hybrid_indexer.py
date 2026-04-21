# app/services/embedding/hybrid_indexer.py
"""
Hybrid Indexer - Indexes documents with both dense and sparse embeddings
Supports both regular user files and global sector files.
Generates bm25_{id}.pkl files automatically.
FIXED: Sparse vectors are generated before Qdrant upload to ensure retrieval works.
"""

import os
import uuid
from typing import List, Dict, Any
from qdrant_client import models
from app.config import settings
from app.core.llm_provider import get_qdrant_client, get_sparse_embedder
from app.services.embedding.sparse_embedder import SparseEmbedder
import logging

logger = logging.getLogger(__name__)


class HybridIndexer:
    """
    Indexes documents with hybrid embeddings (dense + sparse) into Qdrant.
    Also manages the creation and updating of BM25 pickle files.
    """
    
    def __init__(self):
        """Initialize HybridIndexer with Qdrant client and embedders."""
        logger.info("[HybridIndexer] Initializing HybridIndexer...")
        self.qdrant_client = get_qdrant_client()
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        
        # Vector names from config
        self.dense_vector_name = settings.QDRANT_DENSE_VECTOR_NAME  # "dense"
        self.sparse_vector_name = settings.QDRANT_SPARSE_VECTOR_NAME  # "sparse"
        
        logger.info(f"[HybridIndexer] Initialization complete. Collection: {self.collection_name}, Dense Vector: {self.dense_vector_name}, Sparse Vector: {self.sparse_vector_name}")
    
    def index_documents(
        self,
        chunks: List[Dict[str, Any]],
        project_id: str,
        sector: str,
        extra_metadata: Dict[str, Any] = None,
        owner_id: str = None
    ) -> int:
        """
        Index regular private project documents.
        """
        if not chunks:
            logger.warning("[HybridIndexer] No chunks provided for indexing.")
            return 0
        
        logger.info(f"[HybridIndexer] Starting private document indexing. Project ID: {project_id}, Sector: {sector}, Chunk Count: {len(chunks)}")
        
        # --- CHANGED: Step 1 is now Updating BM25 (Generate Sparse Vectors) ---
        logger.info("[HybridIndexer] Step 1: Updating BM25 sparse index & Enriching Chunks...")
        self._update_sparse_index_and_enrich(chunks, index_id=sector)
        logger.info(f"[HybridIndexer] Processing {len(chunks)} chunks for project {project_id}")
        
        # 1. STEP ONE: Update BM25 Index & Generate Sparse Vectors
        # We must do this BEFORE uploading so Qdrant gets the vectors
        # Private files use Project ID as the BM25 index identifier
        self._enrich_chunks_with_sparse_vectors(chunks, index_id=project_id)

        # 2. STEP TWO: Upload to Qdrant
        points_count = self._upload_chunks_to_qdrant(
            chunks=chunks,
            source_id=project_id, # Private files use Project ID as source
            project_id=project_id,
            sector=sector,
            owner_id=owner_id,
            is_global=False,
            extra_metadata=extra_metadata
        )
        logger.info(f"[HybridIndexer] Qdrant upload complete. {points_count} points processed.")
        
        logger.info("[HybridIndexer] Private indexing workflow finished.")
        return points_count

    def index_global_documents(
        self,
        chunks: List[Dict[str, Any]],
        sector: str,
        extra_metadata: Dict[str, Any] = None,
        owner_id: str = "system"
    ) -> int:
        """
        Index GLOBAL sector documents.
        """
        if not chunks:
            logger.warning("[HybridIndexer] No chunks provided for global indexing.")
            return 0
        
        logger.info(f"[HybridIndexer] Processing {len(chunks)} GLOBAL chunks for {sector}")
        
        # 1. STEP ONE: Update BM25 Index & Generate Sparse Vectors
        # Global files use Sector as the BM25 index identifier
        self._enrich_chunks_with_sparse_vectors(chunks, index_id=sector)
        
        # 2. STEP TWO: Upload to Qdrant
        points_count = self._upload_chunks_to_qdrant(
            chunks=chunks,
            source_id=sector, # Global files use Sector as source
            project_id="GLOBAL",
            sector=sector,
            owner_id=owner_id,
            is_global=True,
            extra_metadata=extra_metadata
        )
        
        logger.info("[HybridIndexer] Global indexing workflow finished.")
        return points_count

    def _enrich_chunks_with_sparse_vectors(self, chunks: List[Any], index_id: str):
        """
        Updates the local BM25 index with new text and attaches sparse vectors to chunks.
        SAFE MODE: Loads existing PKL if available to prevent overwriting.
        """
        try:
            # A. Extract Text from Chunks
            texts = []
            for c in chunks:
                if hasattr(c, 'page_content'):
                    texts.append(c.page_content)
                else:
                    txt = c.get('text_content') or c.get('content') or ''
                    texts.append(txt)

            if not texts:
                return

            # Define path
            filename = f"bm25_{index_id}.pkl"
            save_path = os.path.join(settings.PROCESSED_DIR, filename)

            # B. Get or Create Embedder
            # 1. Try Memory Cache
            embedder = get_sparse_embedder(index_id)
            
            if embedder is None:
                embedder = SparseEmbedder()
                # 2. Try Disk Load (CRITICAL FIX: Don't overwrite if exists!)
                if os.path.exists(save_path):
                    logger.info(f"[HybridIndexer] Loading EXISTING BM25 index from disk: {save_path}")
                    embedder.load_index(save_path)
                else:
                    logger.info(f"[HybridIndexer] Creating NEW BM25 index for '{index_id}'")
            else:
                logger.info(f"[HybridIndexer] Using CACHED BM25 index for '{index_id}'")

            # C. Update Index with NEW text
            # This appends new text to the existing corpus and re-calculates IDF
            embedder.update_index(texts)
            
            # D. Save Index to Disk
            os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
            embedder.save_index(save_path)
            logger.info(f"[HybridIndexer] Saved updated index to {save_path}")

            # E. Generate Vectors for current chunks
            logger.info(f"[HybridIndexer] Generating sparse vectors for {len(chunks)} chunks...")
            for i, chunk in enumerate(chunks):
                text = texts[i]
                sparse_vec = embedder.get_sparse_embedding(text)
                
                # Attach to metadata
                if hasattr(chunk, 'metadata'):
                    chunk.metadata['sparse_embedding'] = sparse_vec
                else:
                    if 'metadata' not in chunk: chunk['metadata'] = {}
                    chunk['metadata']['sparse_embedding'] = sparse_vec
            
            logger.info("[HybridIndexer] Sparse vectors attached successfully.")

        except Exception as e:
            logger.error(f"[HybridIndexer] Failed to generate sparse vectors: {e}")
            import traceback
            traceback.print_exc()

    def _upload_chunks_to_qdrant(
        self,
        chunks: List[Dict[str, Any]],
        source_id: str,
        project_id: str,
        sector: str,
        owner_id: str,
        is_global: bool,
        extra_metadata: Dict[str, Any] = None
    ) -> int:
        """Helper to construct points and upsert to Qdrant."""
        logger.info(f"[HybridIndexer] Constructing Qdrant points. Source: {source_id}, Is Global: {is_global}")
        points = []
        
        for chunk in chunks:
            # 1. Unpack Chunk (Handle both dict and Document objects)
            is_dict = False
            if hasattr(chunk, 'page_content'):
                content = chunk.page_content
                chunk_metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
            else:
                is_dict = True
                content = chunk.get('content', '')
                chunk_metadata = chunk.get('metadata', {})
            
            # 2. Get Vectors
            dense_vector = chunk_metadata.get('dense_embedding')
            # Fallback for dicts if embedding is at root
            if not dense_vector and is_dict:
                dense_vector = chunk.get('dense_embedding')
                
            sparse_vector = chunk_metadata.get('sparse_embedding')
            if not sparse_vector and is_dict:
                sparse_vector = chunk.get('sparse_embedding')
            
            if not dense_vector:
                # If Layer 2 failed to generate dense vector, we skip this chunk
                logger.warning("[HybridIndexer] Chunk missing dense embedding, skipping")
                continue

            # 3. Build Vector Payload
            vectors = {self.dense_vector_name: dense_vector}
            
            # Only add sparse if it exists and is not empty
            if sparse_vector and isinstance(sparse_vector, dict) and len(sparse_vector) > 0:
                vectors[self.sparse_vector_name] = models.SparseVector(
                    indices=list(sparse_vector.keys()),
                    values=list(sparse_vector.values())
                )
            
            # 4. Resolve IDs and Metadata
            # --- FIX: Safe ID Resolution ---
            chunk_id = chunk_metadata.get('chunk_id')
            if not chunk_id and is_dict:
                chunk_id = chunk.get('chunk_id')
            
            if not chunk_id:
                chunk_id = str(uuid.uuid4())
            # -------------------------------
            
            # Resolve File Name with Priority
            file_name = (
                chunk_metadata.get('file_name') or 
                (extra_metadata.get('file_name') if extra_metadata else None) or 
                'unknown'
            )

            # Resolve Sector with Priority
            final_sector = chunk_metadata.get('sector') or sector or "N/A"

            # Handle 'page' vs 'page_number' mismatch
            raw_page = chunk_metadata.get('page')
            if raw_page is None:
                raw_page = chunk_metadata.get('page_number')
            
            # Build final metadata dict
            metadata = {
                "source": source_id,
                "project_id": project_id,
                "sector": final_sector,
                "owner_id": owner_id,
                "file_name": file_name,
                "chunk_id": chunk_id,
                "page_number": raw_page,
                "chunk_type": chunk_metadata.get('chunk_type', 'text'),
                "is_global": is_global
            }
            
            # Merge extra metadata safely
            if extra_metadata:
                clean_extra = {k:v for k,v in extra_metadata.items() if k not in metadata}
                metadata.update(clean_extra)
            
            # 5. Create Point
            points.append(models.PointStruct(
                id=chunk_id,
                vector=vectors,
                payload={
                    "page_content": content,
                    "metadata": metadata
                }
            ))
        
        # 6. Upsert Batch
        if points:
            try:
                logger.info(f"[HybridIndexer] Sending upsert request to Qdrant collection '{self.collection_name}'...")
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
                logger.info(f"[HybridIndexer] Qdrant upsert successful for {len(points)} points.")
                return len(points)
            except Exception as e:
                logger.error(f"[HybridIndexer] Qdrant Upsert Error: {e}", exc_info=True)
                raise
        else:
            logger.warning("[HybridIndexer] No valid points created. Skipping upsert.")
            return 0

    # --- RENAMED AND UPDATED METHOD ---
    def _update_sparse_index_and_enrich(self, chunks: List[Dict], index_id: str):
        """
        Deprecated: Logic moved to _enrich_chunks_with_sparse_vectors.
        Kept for backward compatibility if needed, but redirects to new method.
        """
        self._enrich_chunks_with_sparse_vectors(chunks, index_id)

    def delete_by_filter(self, project_id: str, filename: str, owner_id: str) -> bool:
        """Delete user documents by filter."""
        logger.info(f"[HybridIndexer] Deleting documents. Project: {project_id}, File: {filename}, Owner: {owner_id}")
        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(key="metadata.project_id", match=models.MatchValue(value=project_id)),
                            models.FieldCondition(key="metadata.file_name", match=models.MatchValue(value=filename)),
                            models.FieldCondition(key="metadata.owner_id", match=models.MatchValue(value=owner_id)),
                        ]
                    )
                )
            )
            logger.info("[HybridIndexer] Deletion successful.")
            return True
        except Exception as e:
            logger.error(f"[HybridIndexer] Delete error: {e}", exc_info=True)
            return False

    def delete_global_by_filter(self, sector: str, filename: str) -> bool:
        """Delete global documents."""
        logger.info(f"[HybridIndexer] Deleting global documents. Sector: {sector}, File: {filename}")
        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(key="metadata.is_global", match=models.MatchValue(value=True)),
                            models.FieldCondition(key="metadata.sector", match=models.MatchValue(value=sector)),
                            models.FieldCondition(key="metadata.file_name", match=models.MatchValue(value=filename)),
                        ]
                    )
                )
            )
            logger.info("[HybridIndexer] Global deletion successful.")
            return True
        except Exception as e:
            logger.error(f"[HybridIndexer] Global delete error: {e}", exc_info=True)
            return False

    def update_sector(self, project_id: str, filename: str, new_sector: str, owner_id: str) -> bool:
        """Update sector metadata."""
        logger.info(f"[HybridIndexer] Updating sector to '{new_sector}' for File: {filename} in Project: {project_id}")
        try:
            # Scroll to find points
            points_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="metadata.project_id", match=models.MatchValue(value=project_id)),
                        models.FieldCondition(key="metadata.file_name", match=models.MatchValue(value=filename)),
                        models.FieldCondition(key="metadata.owner_id", match=models.MatchValue(value=owner_id)),
                    ]
                ),
                limit=1000
            )
            
            points = points_result[0]
            
            if not points:
                logger.warning("[HybridIndexer] No points found to update.")
                return False
                
            # Batch update payload
            point_ids = [p.id for p in points]
            self.qdrant_client.set_payload(
                collection_name=self.collection_name,
                payload={"metadata": {"sector": new_sector}}, 
                points=point_ids
            )
            return True
        except Exception as e:
            logger.error(f"[HybridIndexer] Update error: {e}", exc_info=True)
            return False


# Singleton instance
_hybrid_indexer = None

def get_hybrid_indexer() -> HybridIndexer:
    """Get or create singleton HybridIndexer instance."""
    global _hybrid_indexer
    if _hybrid_indexer is None:
        logger.info("[HybridIndexer] Creating new singleton instance.")
        _hybrid_indexer = HybridIndexer()
    else:
        logger.info("[HybridIndexer] Returning existing singleton instance.")
    return _hybrid_indexer
