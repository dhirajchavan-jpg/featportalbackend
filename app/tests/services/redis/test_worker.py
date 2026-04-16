import pytest
import json
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock, ANY
from bson import ObjectId

# Import the module under test
from app.services.redis.worker import process_job, worker_loop, RUNNING

# --- FIXTURES ---

@pytest.fixture
def mock_redis():
    with patch("app.services.redis.worker.redis_service") as mock:
        # Setup async methods
        mock.update_job_result = AsyncMock()
        mock.is_job_cancelled = AsyncMock(return_value=False)
        mock.client.brpoplpush = AsyncMock()
        mock.client.lrem = AsyncMock()
        mock.client.rpop = AsyncMock()
        yield mock

@pytest.fixture
def mock_mongo():
    with patch("app.services.redis.worker.file_collection", new_callable=AsyncMock) as mock_file_col, \
         patch("app.services.redis.worker.Global_file_collection", new_callable=AsyncMock) as mock_global_col:
        
        mock_file_col.insert_one.return_value.inserted_id = ObjectId()
        yield mock_file_col, mock_global_col

@pytest.fixture
def mock_rag_pipeline():
    with patch("app.services.redis.worker.query_rag_pipeline", new_callable=AsyncMock) as mock:
        mock.return_value = {
            "result": "Test Answer",
            "source_documents": [],
            "retrieval_stats": {},
            "_eval_data": {"some": "data"} 
        }
        yield mock

@pytest.fixture
def mock_file_indexing():
    with patch("app.services.redis.worker.process_and_index_file", new_callable=MagicMock) as mock_idx, \
         patch("app.services.redis.worker.process_and_index_global_file", new_callable=MagicMock) as mock_global_idx:
        
        mock_idx.return_value = {"status": "indexed", "vectors": 10}
        mock_global_idx.return_value = {"status": "global_indexed"}
        yield mock_idx, mock_global_idx

@pytest.fixture
def mock_eval_runner():
    with patch("app.services.redis.worker._run_comprehensive_evaluation_background", new_callable=AsyncMock) as mock:
        yield mock

# --- TESTS: JOB PROCESSING (Query) ---

@pytest.mark.asyncio
async def test_process_job_query_success(mock_redis, mock_rag_pipeline, mock_eval_runner):
    """Test standard RAG query processing flow."""
    
    # 1. Setup Job Data
    job_data = {
        "task_id": "task_123",
        "job_type": "query",
        "job_data": {
            "query": "Hello?", 
            "project_id": "p1"
        },
        "user_data": {"user_id": "u1", "email": "test@test.com", "role": "user"}
    }
    job_json = json.dumps(job_data)

    # 2. Run
    await process_job(job_json)

    # 3. Assertions
    # Check Pipeline Call
    mock_rag_pipeline.assert_called_once()
    assert mock_rag_pipeline.call_args[1]["skip_evaluation"] is True 
    
    # Check Redis Updates (Processing -> Completed)
    assert mock_redis.update_job_result.call_count == 2
    mock_redis.update_job_result.assert_any_call("task_123", {}, status="processing")
    
    # Verify Final Success Call
    success_call = mock_redis.update_job_result.call_args_list[-1]
    assert success_call[0][0] == "task_123"
    assert success_call[1]["status"] == "completed"
    assert success_call[0][1]["result"] == "Test Answer"
    
    # Check Evaluation Triggered
    mock_eval_runner.assert_called_once_with({"some": "data"})

@pytest.mark.asyncio
async def test_process_job_query_pipeline_failure(mock_redis, mock_rag_pipeline):
    """Test error handling when pipeline fails."""
    mock_rag_pipeline.side_effect = ValueError("Pipeline Exploded")
    
    # FIX: Added 'query' and 'project_id' so the worker doesn't crash with KeyError
    job_json = json.dumps({
        "task_id": "task_fail", 
        "job_type": "query", 
        "job_data": {
            "query": "Must exist", 
            "project_id": "p1"
        },
        "user_data": {"user_id": "u1", "email": "fail@test.com", "role": "user"}
    })
    
    await process_job(job_json)
    
    # Should update Redis with "failed" status
    mock_redis.update_job_result.assert_called_with(
        "task_fail", 
        {"error": "Pipeline Exploded"}, 
        status="failed"
    )

# --- TESTS: JOB PROCESSING (File Upload) ---

@pytest.mark.asyncio
async def test_process_job_file_upload_success(mock_redis, mock_file_indexing, mock_mongo):
    """Test file upload: indexing -> Mongo Save -> Redis Success."""
    mock_file_col, _ = mock_mongo
    
    job_data = {
        "task_id": "file_task_1",
        "job_type": "file_upload",
        "job_data": {
            "file_path": "/tmp/f1.pdf",
            "project_id": "p1",
            "sector": "IT",
            "original_filename": "f1.pdf",
            "file_id": str(ObjectId()),
            "file_hash": "hash123"
        },
        "user_data": {"user_id": str(ObjectId()), "role": "user", "email": "user@test.com"}
    }
    
    await process_job(json.dumps(job_data))
    
    # 1. Verify Indexing called
    mock_file_indexing[0].assert_called_once()
    
    # 2. Verify Mongo Insert (Crucial!)
    mock_file_col.insert_one.assert_called_once()
    inserted_doc = mock_file_col.insert_one.call_args[0][0]
    assert inserted_doc["filename"] == "f1.pdf"
    assert inserted_doc["file_hash"] == "hash123"
    
    # 3. Verify Redis Success
    mock_redis.update_job_result.assert_called_with(
        "file_task_1", 
        ANY, # Result dict 
        status="completed"
    )

@pytest.mark.asyncio
async def test_process_job_file_upload_cancellation(mock_redis, mock_file_indexing, mock_mongo):
    """Test file upload: Indexing runs, but User Cancels -> Skip DB Save."""
    mock_file_col, _ = mock_mongo
    
    # Simulate User Cancelled
    mock_redis.is_job_cancelled.return_value = True
    
    job_data = {
        "task_id": "cancel_task", 
        "job_type": "file_upload", 
        "job_data": {
            "file_path": "x", "project_id": "p", "sector": "s", "file_id": "123", "original_filename": "x.pdf"
        },
        "user_data": {"user_id": "u1", "email": "cancel@test.com"}
    }
    
    await process_job(json.dumps(job_data))
    
    # Indexing still happened (it's heavy/blocking usually)
    mock_file_indexing[0].assert_called_once()
    
    # CRITICAL: Mongo Insert should be SKIPPED
    mock_file_col.insert_one.assert_not_called()
    
    # Redis status should be 'cancelled'
    mock_redis.update_job_result.assert_called_with(
        "cancel_task", 
        {"error": "Cancelled by user"}, 
        status="cancelled"
    )

# --- TESTS: JOB PROCESSING (Global File) ---

@pytest.mark.asyncio
async def test_process_job_global_upload_success(mock_redis, mock_file_indexing, mock_mongo):
    """Test global file upload success path."""
    _, mock_global_col = mock_mongo
    
    job_data = {
        "task_id": "global_1",
        "job_type": "global_file_upload",
        "job_data": {
            "file_path": "/g/doc.pdf",
            "sector": "Finance",
            "file_id": "global_id_123",
            "original_filename": "doc.pdf"
        },
        "user_data": {"user_id": "admin1", "email": "admin@test.com", "role": "admin"}
    }
    
    await process_job(json.dumps(job_data))
    
    # Verify Global Indexing Function
    mock_file_indexing[1].assert_called_once()
    
    # Verify Success Status
    mock_redis.update_job_result.assert_called_with("global_1", ANY, status="completed")
    
    # Verify NO Delete called (Rollback logic)
    mock_global_col.delete_one.assert_not_called()

@pytest.mark.asyncio
async def test_process_job_global_upload_failure_rollback(mock_redis, mock_file_indexing, mock_mongo):
    """Test global file upload failure triggers Mongo Rollback."""
    _, mock_global_col = mock_mongo
    
    # Simulate Indexing Failure
    mock_file_indexing[1].side_effect = Exception("Indexing Failed")
    
    # FIX: Added 'sector' so worker doesn't crash with KeyError
    job_data = {
        "task_id": "global_fail",
        "job_type": "global_file_upload",
        "job_data": {
            "file_id": "bad_id_999",
            "original_filename": "fail.pdf",
            "file_path": "/path",
            "sector": "Finance"
        },
        "user_data": {"user_id": "admin1", "email": "admin@test.com", "role": "admin"}
    }
    
    await process_job(json.dumps(job_data))
    
    # Verify Rollback (delete_one) was called
    mock_global_col.delete_one.assert_called_with({"file_id": "bad_id_999"})
    
    # Verify Redis Failure
    mock_redis.update_job_result.assert_called_with(
        "global_fail", 
        {"error": "Indexing Failed"}, 
        status="failed"
    )

# --- TESTS: WORKER LOOP & RECOVERY ---

@pytest.mark.asyncio
@patch("app.services.redis.worker.load_models")
@patch("app.services.redis.worker.verify_connection", new_callable=AsyncMock)
@patch("app.services.redis.worker.create_indexes", new_callable=AsyncMock)
@patch("app.services.redis.worker.close_mongo_connection", new_callable=AsyncMock)
async def test_worker_loop_recovery(mock_close, mock_indexes, mock_verify, mock_load, mock_redis):
    """
    Test that the worker:
    1. Checks the processing queue on startup (Recovery).
    2. Processes found items.
    3. Enters main loop (simulated by breaking loop via side_effect).
    """
    
    # Setup Recovery Data with valid user_data
    stranded_job = json.dumps({
        "task_id": "stranded_1", 
        "job_type": "query", 
        "job_data": {"query": "q", "project_id": "p"},
        "user_data": {"user_id": "u1", "email": "test@test.com"}
    })
    
    # Mock RPOP to return 1 stranded job, then None (end of recovery)
    mock_redis.client.rpop.side_effect = [stranded_job, None]
    
    # Mock BRPOPLPUSH to raise Exception to break the infinite Main Loop for testing
    async def side_effect_stop(*args, **kwargs):
        raise SystemExit("Stop Worker")

    mock_redis.client.brpoplpush.side_effect = side_effect_stop

    with patch("app.services.redis.worker.RUNNING", True):
        with patch("app.services.redis.worker.process_job", new_callable=AsyncMock) as mock_process:
            try:
                await worker_loop()
            except SystemExit:
                pass

            # ASSERTIONS
            
            # 1. Verify Recovery Processing
            # process_job called for stranded_job?
            assert mock_process.call_count >= 1
            call_args = mock_process.call_args_list[0]
            assert "stranded_1" in call_args[0][0]

            # 2. Verify Startup Sequence
            mock_verify.assert_called()
            mock_load.assert_called()

@pytest.mark.asyncio
@patch("app.services.redis.worker.process_job", new_callable=AsyncMock)
async def test_worker_main_loop_flow(mock_process, mock_redis):
    """
    Test the reliable queue flow:
    BRPOPLPUSH (Main->Proc) -> Process -> LREM (Proc)
    """
    # Setup: No recovery items
    mock_redis.client.rpop.return_value = None
    
    # Main Loop: Return 1 job, then Stop
    job_json = json.dumps({"task_id": "t1", "user_data": {"user_id": "u", "email": "e"}})
    
    async def one_job_then_stop(*args, **kwargs):
        # First call returns job
        if mock_redis.client.brpoplpush.call_count == 1:
            return job_json
        # Second call stops loop
        raise SystemExit()

    mock_redis.client.brpoplpush.side_effect = one_job_then_stop
    
    try:
        await worker_loop()
    except SystemExit:
        pass
    
    # Assertions
    # 1. Verify Process Called
    mock_process.assert_called_with(job_json)
    
    # 2. Verify LREM called (Ack)
    # LREM(queue, count, value)
    mock_redis.client.lrem.assert_called_with(ANY, 1, job_json)