import pytest
from unittest.mock import AsyncMock, MagicMock
import logging

# Import the module under test
import app.database as database

# ---------------------------------------------------------
# get_database
# ---------------------------------------------------------

def test_get_database_returns_db():
    """
    Test that get_database returns the module-level db object.
    """
    result = database.get_database()
    assert result is database.db


# ---------------------------------------------------------
# create_indexes
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_create_indexes_success(mocker, caplog):
    """
    Test that create_indexes calls create_index on all expected collections.
    """
    # 1. Mock the main db object for dynamic collection access (e.g., db.rate_limits_uploads)
    mock_db = MagicMock()
    # Mock the specific dynamic collections accessed via db.collection_name
    mock_db.rate_limits_uploads.create_index = AsyncMock()
    mock_db.rate_limits_chat.create_index = AsyncMock()

    # 2. Mock the specific module-level collection variables
    mock_cache = MagicMock()
    mock_cache.create_index = AsyncMock()

    mock_project = MagicMock()
    mock_project.create_index = AsyncMock()

    mock_chat_history = MagicMock()
    mock_chat_history.create_index = AsyncMock()

    # 3. Patch the module objects
    mocker.patch.object(database, "db", mock_db)
    mocker.patch.object(database, "cache_collection", mock_cache)
    mocker.patch.object(database, "project_collection", mock_project)
    mocker.patch.object(database, "chat_history_collection", mock_chat_history)

    # 4. Execute
    with caplog.at_level(logging.INFO):
        await database.create_indexes()

    # 5. Assertions
    # Rate Limits (accessed via db.*)
    mock_db.rate_limits_uploads.create_index.assert_any_await("expires_at", expireAfterSeconds=0)
    mock_db.rate_limits_uploads.create_index.assert_any_await("key", unique=True)
    mock_db.rate_limits_chat.create_index.assert_awaited_once()

    # Global Collections
    mock_cache.create_index.assert_awaited_once()
    mock_project.create_index.assert_awaited_once()
    mock_chat_history.create_index.assert_awaited_once()

    # Verify Success Log
    assert "MongoDB indexes created successfully!" in caplog.text


@pytest.mark.asyncio
async def test_create_indexes_failure(mocker, caplog):
    """
    Test that create_indexes handles exceptions gracefully and logs an error.
    """
    # Simulate an error on one of the index creations
    mock_db = MagicMock()
    mock_db.rate_limits_uploads.create_index = AsyncMock(side_effect=Exception("Index Error"))

    mocker.patch.object(database, "db", mock_db)

    # Execute - should not raise exception, but log it
    with caplog.at_level(logging.ERROR):
        await database.create_indexes()

    # Verify Error Log
    assert "Index creation failed" in caplog.text
    assert "Index Error" in caplog.text


# ---------------------------------------------------------
# verify_connection
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_connection_success(mocker, caplog):
    """
    Test that verify_connection pings the database and logs success.
    """
    mock_client = MagicMock()
    # Mock successful ping response
    mock_client.admin.command = AsyncMock(return_value={"ok": 1})

    mocker.patch.object(database, "client", mock_client)
    # Patch pooling options since they are accessed in the log message
    mocker.patch.object(database, "pooling_options", {"minPoolSize": 10, "maxPoolSize": 50})

    with caplog.at_level(logging.INFO):
        await database.verify_connection()

    mock_client.admin.command.assert_awaited_once_with("ping")
    assert "Connected to MongoDB successfully!" in caplog.text
    assert "Pool Config: minPoolSize=10" in caplog.text


@pytest.mark.asyncio
async def test_verify_connection_failure(mocker, caplog):
    """
    Test that verify_connection re-raises exception on failure.
    """
    mock_client = MagicMock()
    # Mock failure
    mock_client.admin.command = AsyncMock(side_effect=Exception("Connection Refused"))

    mocker.patch.object(database, "client", mock_client)

    # Verify exception is raised
    with pytest.raises(Exception, match="Connection Refused"):
        with caplog.at_level(logging.ERROR):
            await database.verify_connection()
    
    assert "MongoDB connection failed" in caplog.text


# ---------------------------------------------------------
# close_mongo_connection
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_close_mongo_connection(mocker):
    """
    Test that close_mongo_connection closes the client.
    """
    mock_client = MagicMock()
    mocker.patch.object(database, "client", mock_client)

    await database.close_mongo_connection()

    mock_client.close.assert_called_once()