import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Import the module under test
# Note: This import assumes you are running pytest from the project root.
from app.services.rag.sector_registry import get_available_sectors, _SECTOR_CACHE, SECTOR_CACHE_TTL

# --- Fixtures ---

@pytest.fixture
def mock_qdrant_client():
    """Mocks the QdrantClient class."""
    with patch("app.services.rag.sector_registry.QdrantClient") as MockClient:
        instance = MockClient.return_value
        yield instance

@pytest.fixture
def reset_cache():
    """Resets the global _SECTOR_CACHE before and after each test."""
    _SECTOR_CACHE["data"] = []
    _SECTOR_CACHE["last_updated"] = None
    yield
    _SECTOR_CACHE["data"] = []
    _SECTOR_CACHE["last_updated"] = None

# --- Tests ---

def test_get_sectors_from_cache(reset_cache):
    """Test that data is returned from cache if valid."""
    # 1. Populate Cache
    _SECTOR_CACHE["data"] = ["Finance", "Healthcare"]
    _SECTOR_CACHE["last_updated"] = datetime.utcnow() # Just updated

    # 2. Call Function
    # We patch QdrantClient to ensure it is NOT called (strict check)
    with patch("app.services.rag.sector_registry.QdrantClient") as MockClient:
        sectors = get_available_sectors()
        
        assert sectors == ["Finance", "Healthcare"]
        MockClient.assert_not_called()

def test_get_sectors_cache_expired(reset_cache, mock_qdrant_client):
    """Test that data is fetched from DB if cache is expired."""
    # 1. Setup Expired Cache
    _SECTOR_CACHE["data"] = ["Old Data"]
    _SECTOR_CACHE["last_updated"] = datetime.utcnow() - timedelta(minutes=11) # TTL is 10 mins

    # 2. Setup Mock DB Response
    # Point object structure simulation
    p1 = MagicMock()
    p1.payload = {"metadata": {"sector": "New Sector"}}
    
    # Scroll returns (points, offset). offset=None breaks the loop.
    mock_qdrant_client.scroll.return_value = ([p1], None)

    # 3. Call Function
    sectors = get_available_sectors()

    assert sectors == ["New Sector"]
    mock_qdrant_client.scroll.assert_called_once()
    
    # Verify Cache Updated
    assert _SECTOR_CACHE["data"] == ["New Sector"]
    assert _SECTOR_CACHE["last_updated"] is not None

def test_get_sectors_paginated_scroll(reset_cache, mock_qdrant_client):
    """Test that it loops through all pages (scrolls) until offset is None."""
    # Page 1: Sector A, Offset="next_page"
    p1 = MagicMock()
    p1.payload = {"metadata": {"sector": "Sector A"}}
    
    # Page 2: Sector B, Offset=None (End)
    p2 = MagicMock()
    p2.payload = {"metadata": {"sector": "Sector B"}}

    # Configure side_effect for multiple calls
    mock_qdrant_client.scroll.side_effect = [
        ([p1], "next_page_token"), # First call
        ([p2], None)               # Second call
    ]

    sectors = get_available_sectors()

    assert sorted(sectors) == ["Sector A", "Sector B"]
    assert mock_qdrant_client.scroll.call_count == 2

def test_get_sectors_metadata_fallback(reset_cache, mock_qdrant_client):
    """Test fallback logic: try 'source' if 'sector' is missing."""
    p1 = MagicMock()
    p1.payload = {"metadata": {"sector": "Sector X"}} # Has sector
    
    p2 = MagicMock()
    p2.payload = {"metadata": {"source": "Sector Y"}} # Has source only
    
    p3 = MagicMock()
    p3.payload = {"metadata": {}} # Empty metadata (should be ignored)

    mock_qdrant_client.scroll.return_value = ([p1, p2, p3], None)

    sectors = get_available_sectors()

    assert sorted(sectors) == ["Sector X", "Sector Y"]

def test_get_sectors_db_error(reset_cache, mock_qdrant_client):
    """Test robust error handling when DB connection fails."""
    # 1. Setup Stale Cache (to verify fallback behavior)
    _SECTOR_CACHE["data"] = ["Stale Data"]
    _SECTOR_CACHE["last_updated"] = None # Force refresh attempt

    # 2. Simulate DB Crash
    mock_qdrant_client.scroll.side_effect = Exception("Connection Failed")

    # 3. Call Function
    with patch("app.services.rag.sector_registry.logger") as mock_logger:
        sectors = get_available_sectors()
        
        # Should return stale cache instead of crashing
        assert sectors == ["Stale Data"]
        # Should log error
        mock_logger.error.assert_called()