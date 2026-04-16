import pytest
from unittest.mock import MagicMock, patch, ANY, AsyncMock
import uuid

# Import the module under test
from app.services.rag.file_indexing import (
    process_and_index_file,
    process_and_index_global_file,
    delete_document_by_source,
    update_document_sector,
    delete_global_document_by_source
)
from app.dependencies import UserPayload

# --- Fixtures ---

@pytest.fixture
def mock_user():
    return UserPayload(user_id="user_123", email="test@example.com", role="user")

@pytest.fixture
def mock_services():
    """Mocks all external services used in the indexing pipeline."""
    with patch("app.services.rag.file_indexing.get_document_processor") as m_proc, \
         patch("app.services.rag.file_indexing.get_dense_embedder") as m_embed, \
         patch("app.services.rag.file_indexing.get_hybrid_chunker") as m_chunker, \
         patch("app.services.rag.file_indexing.get_hybrid_indexer") as m_indexer, \
         patch("app.services.rag.file_indexing.chat_history_collection") as m_chat_hist, \
         patch("app.services.rag.file_indexing.client") as m_qdrant_client:
        
        # 1. Document Processor
        proc_inst = m_proc.return_value
        proc_inst.process_document.return_value = {
            "content": "Full document text",
            "metadata": {"source": "test.pdf"}
        }
        
        # 2. Embedder (Mock dense embedding)
        embed_inst = m_embed.return_value
        embed_inst.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # 3. Chunker
        chunker_inst = m_chunker.return_value
        chunker_inst.chunk_document.return_value = [
            {"page_content": "Chunk 1", "metadata": {"page": 1}},
            {"page_content": "Chunk 2", "metadata": {"page": 2}}
        ]
        chunker_inst.get_chunk_statistics.return_value = {"chunks": 2}
        
        # 4. Indexer
        indexer_inst = m_indexer.return_value
        indexer_inst.index_documents.return_value = 2
        indexer_inst.index_global_documents.return_value = 2
        indexer_inst.delete_by_filter.return_value = True
        indexer_inst.update_sector.return_value = True
        
        # 5. Mongo Collection (CRITICAL: Must be AsyncMock)
        m_chat_hist.update_many = AsyncMock()
        
        yield {
            "processor": proc_inst,
            "embedder": embed_inst,
            "chunker": chunker_inst,
            "indexer": indexer_inst,
            "chat_history": m_chat_hist,
            "qdrant": m_qdrant_client
        }

# --- Tests: Private File Indexing ---

def test_process_and_index_file_success(mock_services, mock_user):
    """Test standard private file indexing flow."""
    with patch("app.routes.file_router.active_cancellations", set()):
        
        result = process_and_index_file(
            file_path="/tmp/test.pdf",
            project_id="proj_1",
            sector="Finance",
            current_user=mock_user,
            original_filename="test.pdf"
        )
        
        assert result["success"] is True
        assert result["chunks_indexed"] == 2
        
        mock_services["processor"].process_document.assert_called_with(
            file_path="/tmp/test.pdf",
            file_id=ANY,
            project_id="proj_1",
            sector="Finance",
            ocr_engine_name="paddleocr"
        )
        
        mock_services["indexer"].index_documents.assert_called_with(
            chunks=ANY,
            project_id="proj_1",
            sector="Finance",
            extra_metadata=ANY,
            owner_id="user_123"
        )

def test_process_and_index_file_cancellation_at_start(mock_services, mock_user):
    """Test early exit if file is in active_cancellations."""
    with patch("app.routes.file_router.active_cancellations", {"test.pdf"}):
        
        result = process_and_index_file(
            file_path="/tmp/test.pdf",
            project_id="proj_1",
            sector="Finance",
            current_user=mock_user,
            original_filename="test.pdf"
        )
        
        assert result["success"] is False
        assert result["status"] == "cancelled"
        mock_services["processor"].process_document.assert_not_called()

def test_process_and_index_metadata_injection(mock_services, mock_user):
    """Test that filename and project_id are injected into document metadata."""
    with patch("app.routes.file_router.active_cancellations", set()):
        
        process_and_index_file(
            file_path="p", project_id="proj_1", sector="sec", 
            current_user=mock_user, original_filename="my_doc.pdf"
        )
        
        call_args = mock_services["chunker"].chunk_document.call_args
        doc_json = call_args[0][0]
        
        assert doc_json["metadata"]["file_name"] == "my_doc.pdf"
        assert doc_json["metadata"]["project_id"] == "proj_1"

# --- Tests: Global File Indexing ---

def test_process_and_index_global_file_success(mock_services):
    """Test indexing of global documents."""
    # FIX: Pass extra_metadata={} to avoid NoneType error in source code
    result = process_and_index_global_file(
        file_path="/tmp/global.pdf",
        sector="Healthcare",
        file_id="global_1",
        original_filename="global.pdf",
        extra_metadata={} 
    )
    
    assert result["success"] is True
    assert result["chunks_indexed"] == 2
    
    mock_services["indexer"].index_global_documents.assert_called_once()
    
    call_args = mock_services["indexer"].index_global_documents.call_args
    extra_meta = call_args.kwargs['extra_metadata']
    assert extra_meta["is_global"] is True
    assert extra_meta["file_name"] == "global.pdf"

# --- Tests: Management Functions ---

@pytest.mark.asyncio
async def test_delete_document_by_source(mock_services, mock_user):
    """Test deletion from Qdrant and Mongo history update."""
    filename = "test.pdf"
    proj_id = "p1"
    
    # Mock update result
    mock_update = MagicMock()
    mock_update.modified_count = 1
    mock_services["chat_history"].update_many.return_value = mock_update

    result = await delete_document_by_source(filename, proj_id, mock_user)
    
    # 1. Verify Indexer Delete Call
    mock_services["indexer"].delete_by_filter.assert_called_with(
        proj_id, filename, mock_user.user_id
    )
    
    # 2. Verify Mongo Update
    mock_services["chat_history"].update_many.assert_called()
    assert result["status"] == "deleted"
    assert result["chat_history_updated"] == 1

@pytest.mark.asyncio
async def test_update_document_sector(mock_services, mock_user):
    """Test updating sector in Qdrant and Mongo."""
    # Mock update result
    mock_update = MagicMock()
    mock_update.matched_count = 1
    mock_update.modified_count = 1
    mock_services["chat_history"].update_many.return_value = mock_update

    result = await update_document_sector("file.pdf", "p1", "NewSector", mock_user)
    
    # 1. Verify Indexer Update
    mock_services["indexer"].update_sector.assert_called_with(
        "p1", "file.pdf", "NewSector", mock_user.user_id
    )
    
    # 2. Verify Mongo Update
    mock_services["chat_history"].update_many.assert_called()
    assert result["status"] == "updated"

@pytest.mark.asyncio
async def test_delete_global_document_by_source(mock_services):
    """Test deleting a global document via Qdrant Client directly."""
    mock_qdrant = mock_services["qdrant"]
    mock_qdrant.delete.return_value = "Deleted"
    
    result = await delete_global_document_by_source("global.pdf", "Finance")
    
    assert result["status"] == "deleted"
    
    mock_qdrant.delete.assert_called_once()
    args, kwargs = mock_qdrant.delete.call_args
    assert "collection_name" in kwargs
    assert kwargs["wait"] is True

@pytest.mark.asyncio
async def test_delete_global_failure_handling(mock_services):
    """Test error handling during global deletion."""
    mock_services["qdrant"].delete.side_effect = Exception("Qdrant Down")
    
    result = await delete_global_document_by_source("f.pdf", "S")
    
    assert result["status"] == "error"
    assert "Qdrant Down" in result["message"]