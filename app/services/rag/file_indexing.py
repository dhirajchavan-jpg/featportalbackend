# app/services/rag/file_indexing.py
import uuid
import time
from typing import Optional, Dict, Any
import logging
from qdrant_client import QdrantClient, models
from app.config import settings
client = QdrantClient(url=settings.QDRANT_URL)
COLLECTION_NAME = settings.QDRANT_COLLECTION_NAME

from app.database import chat_history_collection
from app.dependencies import UserPayload

from app.services.document_processing.json_builder import get_document_processor
from app.core.llm_provider import get_dense_embedder
from app.services.chunking.hybrid_chunker import get_hybrid_chunker
from app.services.embedding.hybrid_indexer import get_hybrid_indexer

logger = logging.getLogger(__name__)

def process_and_index_file(
    file_path: str,
    project_id: str,
    sector: str,
    current_user: UserPayload,
    doc_type: str = "general",
    original_filename: Optional[str] = None,
    file_id: Optional[str] = None,
    ocr_engine: str = "paddleocr" 
):
    """
    Complete document processing and indexing pipeline for Private Files.
    UPDATED: Now strictly attaches filename and project context to all chunks.
    """
    start_time = time.time()
    
    if not file_id:
        file_id = str(uuid.uuid4())
        logger.info(f"No file_id provided. Generated new ID: {file_id}")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"STARTING PIPELINE: {original_filename or file_path}")
    logger.info(f"Project ID: {project_id}")
    logger.info(f"File ID: {file_id}")
    logger.info(f"OCR Strategy: {ocr_engine}") 
    logger.info(f"{'='*70}\n")
    
    try:
        # LAYER 1: Document Processing
        logger.info("[LAYER 1] Document Processing...")
        layer1_start = time.time()
        
        processor = get_document_processor()
        
        # --- PASSING OCR PREFERENCE DOWN ---
        document_json = processor.process_document(
            file_path=file_path,
            file_id=file_id,
            project_id=project_id,
            sector=sector,
            ocr_engine_name=ocr_engine 
        )
        
        # --- FIX START: Inject Filename into Metadata ---
        # This ensures the retrieval layer can cite the source file correctly
        if 'metadata' not in document_json:
            document_json['metadata'] = {}
        
        if original_filename:
            document_json['metadata']['file_name'] = original_filename
            document_json['metadata']['project_id'] = project_id
            document_json['metadata']['sector'] = sector
            logger.info(f"[LAYER 1] Attached metadata: {original_filename}")
        # --- FIX END ---
        
        logger.info(f"[LAYER 1] Completed in {time.time() - layer1_start:.2f}s")
        
        # LAYER 2: Chunking
        logger.info("\n[LAYER 2] Hybrid Chunking...")
        layer2_start = time.time()
        
        embeddings = get_dense_embedder()
        chunker = get_hybrid_chunker(embeddings)
        chunks = chunker.chunk_document(document_json)
        
        # --- FIX START: Propagate Filename to Chunks ---
        # Ensure every single chunk knows which file it belongs to
        if original_filename:
            for chunk in chunks:
                if hasattr(chunk, 'metadata'):
                    chunk.metadata['file_name'] = original_filename
                    chunk.metadata['project_id'] = project_id
                    chunk.metadata['sector'] = sector
                elif isinstance(chunk, dict):
                    if 'metadata' not in chunk:
                        chunk['metadata'] = {}
                    chunk['metadata']['file_name'] = original_filename
                    chunk['metadata']['project_id'] = project_id
                    chunk['metadata']['sector'] = sector
        # --- FIX END ---
        
        chunk_stats = chunker.get_chunk_statistics(chunks)
        logger.info(f"[LAYER 2] Created {len(chunks)} chunks in {time.time() - layer2_start:.2f}s")
        
        # LAYER 3: Hybrid Indexing
        logger.info("\n[LAYER 3] Hybrid Embedding & Indexing...")
        layer3_start = time.time()
        
        # Prepare metadata for indexing
        extra_meta = {
            "doc_type": doc_type,
            "project_id": project_id,
            "file_name": original_filename,
            "sector": sector
        }

        indexer = get_hybrid_indexer()
        
        indexed_count = indexer.index_documents(
            chunks=chunks,
            project_id=project_id,
            sector=sector,
            extra_metadata=extra_meta,
            owner_id=current_user.user_id
        )
        
        logger.info(f"[LAYER 3] Indexed {indexed_count} chunks in {time.time() - layer3_start:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"\n{'='*70}")
        logger.info(f"PIPELINE COMPLETE: {total_time:.2f}s total")
        logger.info(f"{'='*70}\n")
        
        return {
            "success": True,
            "file_id": file_id,
            "chunks_created": len(chunks),
            "chunks_indexed": indexed_count,
            "processing_time": total_time
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to process and index document: {e}")


def process_and_index_global_file(
    file_path: str,
    sector: str,
    file_id: str,
    original_filename: str,
    extra_metadata: Dict[str, Any] = None,
    ocr_engine: str = "paddleocr" 
):
    """Process and index a GLOBAL sector file."""
    start_time = time.time()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"GLOBAL FILE PIPELINE: {original_filename}")
    logger.info(f"Sector: {sector}")
    logger.info(f"File ID: {file_id}")
    logger.info(f"OCR Strategy: {ocr_engine}") 
    logger.info(f"{'='*70}\n")
    
    try:
        # LAYER 1: Document Processing
        logger.info("[LAYER 1] Document Processing...")
        processor = get_document_processor()
        
        # --- PASSING OCR PREFERENCE DOWN ---
        document_json = processor.process_document(
            file_path=file_path,
            file_id=file_id,
            project_id="GLOBAL",
            sector=sector,
            ocr_engine_name=ocr_engine 
        )
        
        if 'metadata' not in document_json:
            document_json['metadata'] = {}
        document_json['metadata']['file_name'] = original_filename
        
        logger.info(f"[LAYER 1] Set file_name in document metadata: {original_filename}")
        
        # LAYER 2: Chunking
        logger.info("[LAYER 2] Hybrid Chunking...")
        embeddings = get_dense_embedder()
        chunker = get_hybrid_chunker(embeddings)
        chunks = chunker.chunk_document(document_json)
        
        logger.info(f"[LAYER 2] Created {len(chunks)} chunks")
        
        # Ensure filename propagates
        for chunk in chunks:
            if hasattr(chunk, 'metadata'):
                if 'file_name' not in chunk.metadata:
                    chunk.metadata['file_name'] = original_filename
            elif isinstance(chunk, dict):
                if 'metadata' not in chunk:
                    chunk['metadata'] = {}
                if 'file_name' not in chunk['metadata']:
                    chunk['metadata']['file_name'] = original_filename
        
        # LAYER 3: Generate embeddings
        logger.info("[LAYER 3] Generating embeddings...")
        
        dense_embedder = get_dense_embedder()
        
        for chunk in chunks:
            if hasattr(chunk, 'page_content'):
                content = chunk.page_content
                chunk.metadata['dense_embedding'] = dense_embedder.embed_query(content)
            else:
                content = chunk.get('content', '')
                chunk['dense_embedding'] = dense_embedder.embed_query(content)
        
        logger.info("[LAYER 3] Embeddings generated")
        
        # LAYER 4: Index
        logger.info("[LAYER 4] Indexing to Qdrant...")
        indexer = get_hybrid_indexer()
        
        global_metadata = {
            "doc_type": "global_regulation",
            "is_global": True,
            "file_name": original_filename
        }
        if extra_metadata:
            global_metadata.update(extra_metadata)
        
        if 'file_name' in extra_metadata:
            global_metadata['file_name'] = original_filename
        
        indexed_count = indexer.index_global_documents(
            chunks=chunks,
            sector=sector,
            extra_metadata=global_metadata,
            owner_id="system"
        )
        
        logger.info(f"[LAYER 4] Indexed {indexed_count} chunks with file_name={original_filename}")
        
        total_time = time.time() - start_time
        logger.info(f"\n{'='*70}")
        logger.info(f"GLOBAL PIPELINE COMPLETE: {total_time:.2f}s")
        logger.info(f"{'='*70}\n")
        
        return {
            "success": True,
            "file_id": file_id,
            "chunks_created": len(chunks),
            "chunks_indexed": indexed_count,
            "processing_time": total_time
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Global pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to process global file: {e}")

async def delete_document_by_source(filename: str, project_id: str, current_user: UserPayload):
    """Delete document from Qdrant and nullify chat history."""
    try:
        indexer = get_hybrid_indexer()
        success = indexer.delete_by_filter(project_id, filename, current_user.user_id)
        
        if success:
            chat_history_filter = {
                "user_id": current_user.user_id,
                "project_id": project_id,
                "file_name": filename,
                "message_type": "file"
            }
            update_result = await chat_history_collection.update_many(
                chat_history_filter,
                {"$set": {"file_id": None, "file_name": None}}
            )
            return {
                "status": "deleted",
                "chat_history_updated": update_result.modified_count
            }
        return {"status": "error", "message": "Deletion failed"}
    except Exception as e:
        logger.error(f"[ERROR] Delete failed: {e}")
        return {"status": "error", "message": str(e)}
    
async def update_document_sector(filename: str, project_id: str, new_sector: str, current_user: UserPayload):
    """Update document sector in Qdrant and MongoDB."""
    try:
        indexer = get_hybrid_indexer()
        success = indexer.update_sector(project_id, filename, new_sector, current_user.user_id)
        
        if success:
            mongo_filter = {
                "project_id": project_id,
                "user_id": current_user.user_id,
                "file_name": filename
            }
            mongo_result = await chat_history_collection.update_many(
                mongo_filter,
                {"$set": {"sector": new_sector}}
            )
            return {
                "status": "updated",
                "mongo_history_update": f"Matched: {mongo_result.matched_count}, Modified: {mongo_result.modified_count}"
            }
        return {"status": "error", "message": "Update failed"}
    except Exception as e:
        logger.error(f"[ERROR] Update failed: {e}")
        return {"status": "error", "message": str(e)}

async def delete_global_document_by_source(filename: str, sector: str):
    """
    Delete a GLOBAL document from Qdrant using filename + sector filter.
    """

    try:
        logger.info(
            f"[QDRANT][GLOBAL] Deleting chunks | File: {filename} | Sector: {sector}"
        )

        delete_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.file_name",
                    match=models.MatchValue(value=filename)
                ),
                models.FieldCondition(
                    key="metadata.sector",
                    match=models.MatchValue(value=sector)
                )
            ]
        )

        result = client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(filter=delete_filter),
            wait=True
        )

        logger.info(f"[QDRANT][GLOBAL] Delete result: {result}")

        return {
            "status": "deleted",
            "sector": sector,
            "filename": filename,
            "message": f"Successfully deleted {filename} from {sector}"
        }

    except Exception as e:
        logger.error(f"[ERROR] Global Qdrant delete failed: {e}")
        return {
            "status": "error",
            "sector": sector,
            "filename": filename,
            "message": str(e)
        }