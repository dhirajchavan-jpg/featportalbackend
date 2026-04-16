import pytest
import os
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock, ANY
from fastapi import HTTPException, UploadFile
from io import BytesIO
from bson import ObjectId

# Import the module under test
from app.middleware.files_middleware import (
    sanitize_filename,
    calculate_sha256,
    validate_token,
    verify_project,
    save_file,
    run_qdrant_indexing,
    get_all_files_for_project,
    handle_file_deletion,
    stream_to_temp_file,
    calculate_file_hash,
    store_file_metadata
)
from app.dependencies import UserPayload
from app.models.Files import FileModel

# --- Fixtures ---

@pytest.fixture
def mock_user():
    # Use a valid ObjectId string for user_id
    valid_id = str(ObjectId())
    return UserPayload(user_id=valid_id, role="user", email="test@example.com")

@pytest.fixture
def mock_admin():
    valid_id = str(ObjectId())
    return UserPayload(user_id=valid_id, role="admin", email="admin@example.com")

@pytest.fixture
def mock_upload_file():
    return UploadFile(
        filename="test_document.pdf",
        file=BytesIO(b"dummy content for testing")
    )

# --- 1. Utility Tests ---

def test_sanitize_filename():
    """Test filename cleaning regex."""
    assert sanitize_filename("File! @Name #Test.pdf") == "File_Name_Testpdf"
    assert sanitize_filename("  spaces  .txt") == "_spaces_txt"

def test_calculate_sha256_deterministic():
    """Test hash consistency."""
    content = b"sensitive data"
    assert calculate_sha256(content) == calculate_sha256(content)

# --- 2. Auth & Project Validation Tests ---

@pytest.mark.asyncio
@patch("app.middleware.files_middleware.decode_access_token")
async def test_validate_token_success(mock_decode):
    mock_decode.return_value = {"user_id": "123", "sub": "test@test.com"}
    payload = await validate_token("valid_token")
    assert payload["user_id"] == "123"

@pytest.mark.asyncio
@patch("app.middleware.files_middleware.decode_access_token")
async def test_validate_token_failure(mock_decode):
    mock_decode.return_value = None
    with pytest.raises(HTTPException) as exc:
        await validate_token("invalid_token")
    assert exc.value.status_code == 401

@pytest.mark.asyncio
@patch("app.middleware.files_middleware.project_collection.find_one", new_callable=AsyncMock)
async def test_verify_project_forbidden(mock_find, mock_user):
    # Simulate project not found or not owned
    mock_find.return_value = None
    valid_oid = str(ObjectId())
    
    with pytest.raises(HTTPException) as exc:
        await verify_project(valid_oid, mock_user)
    assert exc.value.status_code == 403

# --- 3. File Processing & Saving Tests ---

@pytest.mark.asyncio
async def test_stream_to_temp_file_success(mock_upload_file):
    """Test the temp file streaming logic (integration with tempfile)."""
    with patch("app.middleware.files_middleware.asyncio.to_thread") as mock_to_thread:
        with patch("app.middleware.files_middleware.tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.return_value.name = "/tmp/mock_temp_file"
            
            path = await stream_to_temp_file(mock_upload_file)
            
            assert path == "/tmp/mock_temp_file"
            assert mock_to_thread.call_count > 0

@pytest.mark.asyncio
@patch("app.middleware.files_middleware.file_collection", new_callable=AsyncMock)
@patch("app.middleware.files_middleware.os.path.getsize")
@patch("app.middleware.files_middleware.stream_to_temp_file", new_callable=AsyncMock)
@patch("app.middleware.files_middleware.calculate_file_hash", new_callable=AsyncMock)
@patch("app.middleware.files_middleware.asyncio.to_thread", new_callable=AsyncMock)
async def test_save_file_success(
    mock_to_thread, mock_calc_hash, mock_stream, mock_getsize, mock_db, mock_upload_file
):
    mock_stream.return_value = "/tmp/temp_123"
    mock_getsize.return_value = 1024 
    mock_calc_hash.return_value = "dummy_sha256_hash"
    mock_db.find_one.return_value = None 
    
    def side_effect_handler(*args, **kwargs):
        if args and "magic" in str(args[0]): 
             return "application/pdf"
        return None 

    mock_to_thread.side_effect = ["application/pdf", None]

    filename, path, file_hash = await save_file(mock_upload_file, "project_123")

    assert filename == "test_document"
    assert file_hash == "dummy_sha256_hash"
    assert "test_document.pdf" in path

@pytest.mark.asyncio
@patch("app.middleware.files_middleware.stream_to_temp_file", new_callable=AsyncMock)
@patch("app.middleware.files_middleware.os.path.getsize")
async def test_save_file_size_limit_exceeded(mock_getsize, mock_stream, mock_upload_file):
    mock_stream.return_value = "/tmp/big_file"
    mock_getsize.return_value = 1024 * 1024 * 1024 
    
    with pytest.raises(HTTPException) as exc:
        await save_file(mock_upload_file, "project_123")
    
    assert exc.value.status_code == 400
    assert "limit" in exc.value.detail

@pytest.mark.asyncio
@patch("app.middleware.files_middleware.file_collection", new_callable=AsyncMock)
@patch("app.middleware.files_middleware.stream_to_temp_file", new_callable=AsyncMock)
@patch("app.middleware.files_middleware.os.path.getsize")
@patch("app.middleware.files_middleware.calculate_file_hash", new_callable=AsyncMock)
async def test_save_file_duplicate_hash(
    mock_calc_hash, mock_getsize, mock_stream, mock_db, mock_upload_file
):
    mock_stream.return_value = "/tmp/temp_dup"
    mock_getsize.return_value = 500
    mock_calc_hash.return_value = "existing_hash"
    mock_db.find_one.return_value = {"filename": "old.pdf"}
    
    with pytest.raises(HTTPException) as exc:
        await save_file(mock_upload_file, "project_123")
    
    assert exc.value.status_code == 409
    assert "Duplicate file content" in exc.value.detail

# --- 4. Redis & Indexing Tests ---

@pytest.mark.asyncio
@patch("app.middleware.files_middleware.redis_service.enqueue_job", new_callable=AsyncMock)
async def test_run_qdrant_indexing_enqueues_redis(mock_enqueue, mock_user):
    mock_enqueue.return_value = "task_uuid_123"

    task_id = await run_qdrant_indexing(
        file_path="/docs/file.pdf",
        project_id="proj_1",
        sector="IT",
        current_user=mock_user,
        category="Invoices",
        original_filename="orig.pdf"
    )

    assert task_id == "task_uuid_123"
    mock_enqueue.assert_called_once()
    call_kwargs = mock_enqueue.call_args[1]
    assert call_kwargs["job_type"] == "file_upload"
    assert call_kwargs["user_data"]["email"] == "test@example.com"

# --- 5. Metadata & Permission Tests ---

@pytest.mark.asyncio
@patch("app.middleware.files_middleware.file_collection", new_callable=AsyncMock)
async def test_store_file_metadata(mock_db):
    mock_db.insert_one.return_value.inserted_id = ObjectId()
    
    result = await store_file_metadata(
        project_id="p1", 
        filename="f1", 
        file_path="/p/f1", 
        file_hash="h1",
        category="Legal",
        user_id="u1"
    )
    
    assert result.category == "Legal"
    assert result.user_id == "u1"
    assert mock_db.insert_one.called

@pytest.mark.asyncio
@patch("app.middleware.files_middleware.file_collection")
@patch("app.middleware.files_middleware.User_collection")
@patch("app.middleware.files_middleware.verify_project", new_callable=AsyncMock)
async def test_get_all_files_for_project_user_logic(
    mock_verify, mock_user_coll, mock_file_coll, mock_user
):
    """
    Test that a regular user sees:
    1. Their own files
    2. Files uploaded by their Admin (if admin_id exists)
    """
    # Setup User Collection to return an admin_id
    admin_id = str(ObjectId())
    mock_user_coll.find_one = AsyncMock(return_value={
        "_id": ObjectId(mock_user.user_id),
        "admin_id": ObjectId(admin_id) 
    })

    # Setup File Collection Cursor
    mock_cursor = MagicMock()
    mock_cursor.__aiter__.return_value = [
        {
            "_id": "f1", 
            "project_id": "p1", 
            "filename": "doc1.pdf", 
            "user_id": mock_user.user_id,
            "file_url": "/docs/doc1.pdf" # <-- ADDED THIS
        },
        {
            "_id": "f2", 
            "project_id": "p1", 
            "filename": "admin_doc.pdf", 
            "user_id": admin_id,
            "file_url": "/docs/admin_doc.pdf" # <-- ADDED THIS
        }
    ].__iter__()
    
    mock_file_coll.find.return_value = mock_cursor

    files = await get_all_files_for_project("p1", mock_user)

    call_args = mock_file_coll.find.call_args[0][0]
    assert "user_id" in call_args
    assert "$in" in call_args["user_id"]
    assert mock_user.user_id in call_args["user_id"]["$in"]
    assert admin_id in call_args["user_id"]["$in"]

# --- 6. Deletion Tests ---

@pytest.mark.asyncio
@patch("app.middleware.files_middleware.delete_document_by_source", new_callable=AsyncMock)
@patch("app.middleware.files_middleware.file_collection", new_callable=AsyncMock)
@patch("app.middleware.files_middleware.os.remove")
async def test_handle_file_deletion_soft_delete(
    mock_os_remove, mock_file_coll, mock_qdrant_del, mock_user
):
    """
    Test that deletion:
    1. Calls Qdrant deletion
    2. Deletes DB metadata
    3. DOES NOT delete physical file (os.remove should NOT be called)
    """
    # Ensure all required Pydantic fields are present
    file_doc = FileModel(
        file_id=str(ObjectId()), 
        project_id="p1", 
        filename="test.pdf", 
        file_url="/docs/test.pdf",
        sector="General",
        category="Other",
        compliance_type="General",
        user_id=mock_user.user_id
    )
    mock_file_coll.delete_one.return_value.deleted_count = 1

    await handle_file_deletion(file_doc, "p1", mock_user)

    mock_qdrant_del.assert_called_once() 
    mock_file_coll.delete_one.assert_called_once() 
    
    mock_os_remove.assert_not_called()