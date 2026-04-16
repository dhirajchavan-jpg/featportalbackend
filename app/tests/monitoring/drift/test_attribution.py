# app/tests/monitoring/test_drift_attribution.py
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from app.monitoring.drift.attribution import attribute_drift, DRIFT_ATTRIBUTION_RULES


# -------------------------------
# Test: attribute_drift returns correct counts and top_examples
# -------------------------------
@pytest.mark.asyncio
async def test_attribute_drift_returns_correct_counts(monkeypatch):
    """
    Summary:
        Validates that `attribute_drift` correctly counts documents exceeding the drift threshold
        and extracts top examples with scores and reasons.

    Explanation:
        - Mocks MongoDB collection methods: count_documents, find, insert_one.
        - Uses an async generator to simulate documents returned from find().
        - Verifies that the count, total, and top_examples fields are returned correctly.
        - Ensures that insert_one is called to save drift references.
    """
    since = datetime.now(timezone.utc) - timedelta(days=7)
    until = datetime.now(timezone.utc)
    report_reference_id = "test_report_123"
    drift_type = "HALLUCINATION"
    threshold = 0.5

    # Mock count_documents to return drifted count and total
    mock_collection = MagicMock()
    mock_collection.count_documents = AsyncMock(side_effect=[5, 20])

    # Mock find() to yield example documents
    async def async_gen():
        yield {"query": "Test query 1", "generation": {"hallucination_score": 0.7}}
        yield {"query": "Test query 2", "generation": {"hallucination_score": 0.8}}

    mock_collection.find = MagicMock(return_value=async_gen())
    mock_collection.insert_one = AsyncMock()

    # Mock database
    mock_db = MagicMock()
    def db_getitem(key):
        if key == "comprehensive_evaluations":
            return mock_collection
        if key == "drift_query_references":
            return mock_collection
    mock_db.__getitem__.side_effect = db_getitem

    monkeypatch.setattr("app.monitoring.drift.attribution.db", mock_db)

    # Call function
    result = await attribute_drift(
        drift_type=drift_type,
        threshold=threshold,
        since=since,
        until=until,
        report_reference_id=report_reference_id
    )

    # Assertions
    assert result["count"] == 5
    assert result["total"] == 20
    assert len(result["top_examples"]) == 2
    assert result["top_examples"][0]["score"] == 0.7
    assert result["top_examples"][0]["reason"] == f"{drift_type} definition threshold breached"
    mock_collection.insert_one.assert_awaited_once()


# -------------------------------
# Test: attribute_drift handles empty results
# -------------------------------
@pytest.mark.asyncio
async def test_attribute_drift_empty_results(monkeypatch):
    """
    Summary:
        Ensures `attribute_drift` gracefully handles cases where no documents match
        the drift threshold.

    Explanation:
        - count_documents returns 0 for both drifted and total counts.
        - find() yields no documents via an async generator.
        - top_examples list should be empty.
        - insert_one should still be called to record the empty reference.
    """
    since = datetime.now(timezone.utc) - timedelta(days=1)
    until = datetime.now(timezone.utc)
    report_reference_id = "report_empty"
    drift_type = "RETRIEVAL"
    threshold = 0.2

    mock_collection = MagicMock()
    mock_collection.count_documents = AsyncMock(return_value=0)

    async def empty_async_gen():
        if False:
            yield

    mock_collection.find = MagicMock(return_value=empty_async_gen())
    mock_collection.insert_one = AsyncMock()

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    monkeypatch.setattr("app.monitoring.drift.attribution.db", mock_db)

    result = await attribute_drift(
        drift_type=drift_type,
        threshold=threshold,
        since=since,
        until=until,
        report_reference_id=report_reference_id
    )

    assert result["count"] == 0
    assert result["total"] == 0
    assert result["top_examples"] == []
    mock_collection.insert_one.assert_awaited_once()


# -------------------------------
# Test: attribute_drift extracts nested example scores correctly
# -------------------------------
@pytest.mark.asyncio
async def test_attribute_drift_example_scores(monkeypatch):
    """
    Summary:
        Confirms that `attribute_drift` extracts scores from nested generation fields correctly.

    Explanation:
        - Simulates a document with a nested score (e.g., response_toxicity_score).
        - Verifies that the top_examples list contains the correct score.
        - Ensures that function logic for accessing nested fields is correct.
    """
    since = datetime.now(timezone.utc) - timedelta(days=7)
    until = datetime.now(timezone.utc)
    report_reference_id = "test_nested"
    drift_type = "RESPONSE_TOXICITY"
    threshold = 0.1

    # Simulate one document
    doc = {"query": "Q1", "generation": {"response_toxicity_score": 0.3}}
    async def async_gen():
        yield doc

    mock_collection = MagicMock()
    mock_collection.count_documents = AsyncMock(side_effect=[1, 1])
    mock_collection.find = MagicMock(return_value=async_gen())
    mock_collection.insert_one = AsyncMock()

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    monkeypatch.setattr("app.monitoring.drift.attribution.db", mock_db)

    result = await attribute_drift(
        drift_type=drift_type,
        threshold=threshold,
        since=since,
        until=until,
        report_reference_id=report_reference_id
    )

    assert result["top_examples"][0]["score"] == 0.3
