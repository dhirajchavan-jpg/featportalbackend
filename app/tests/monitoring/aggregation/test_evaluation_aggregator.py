# app/tests/monitoring/test_evaluation_aggregator.py
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from app.monitoring.aggregation.evaluation_aggregator import aggregate_evaluations


# -------------------------------
# Test: Aggregation returns data when MongoDB has results
# -------------------------------
@pytest.mark.asyncio
async def test_aggregate_evaluations_returns_data(monkeypatch):
    """
    Summary:
        Validates that aggregate_evaluations returns correct aggregated metrics
        when the MongoDB aggregation returns data.
    
    Explanation:
        - Mocks the MongoDB collection and cursor to return sample aggregated metrics.
        - Verifies key output fields: total_queries, overall_score_avg, knowledge_gap_rate.
        - Demonstrates that aggregation works and output is correctly parsed.
    """
    since = datetime.now(timezone.utc) - timedelta(days=7)
    until = datetime.now(timezone.utc)

    mock_result = [{
        "faithfulness_avg": 0.82,
        "hallucination_avg": 0.18,
        "overall_score_avg": 0.79,
        "retrieval_avg": 0.74,
        "query_toxicity_avg": 0.02,
        "response_toxicity_avg": 0.01,
        "pipeline_health_avg": 0.85,
        "total_queries": 120,
        "knowledge_gap_rate": 0.12,
        "knowledge_coverage_gap_rate": 0.08
    }]

    # Mock Mongo cursor
    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(return_value=mock_result)

    # Mock collection
    mock_collection = MagicMock()
    mock_collection.aggregate.return_value = mock_cursor

    # Mock database
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    # Patch the db in the module
    monkeypatch.setattr(
        "app.monitoring.aggregation.evaluation_aggregator.db",
        mock_db
    )

    result = await aggregate_evaluations(since, until)

    assert result["total_queries"] == 120
    assert result["overall_score_avg"] == 0.79
    assert result["knowledge_gap_rate"] == 0.12


# -------------------------------
# Test: Aggregation returns empty dict when no data exists
# -------------------------------
@pytest.mark.asyncio
async def test_aggregate_evaluations_returns_empty_when_no_data(monkeypatch):
    """
    Summary:
        Ensures that aggregate_evaluations returns an empty dict when no evaluations exist.
    
    Explanation:
        - Mocks the MongoDB cursor to return an empty list.
        - Verifies that the function handles empty data gracefully.
    """
    since = datetime.now(timezone.utc) - timedelta(days=1)
    until = datetime.now(timezone.utc)

    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(return_value=[])

    mock_collection = MagicMock()
    mock_collection.aggregate.return_value = mock_cursor

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    monkeypatch.setattr(
        "app.monitoring.aggregation.evaluation_aggregator.db",
        mock_db
    )

    result = await aggregate_evaluations(since, until)

    assert result == {}


# -------------------------------
# Test: Ensure aggregation pipeline is executed
# -------------------------------
@pytest.mark.asyncio
async def test_aggregate_evaluations_pipeline_called(monkeypatch):
    """
    Summary:
        Confirms that the aggregation pipeline is called exactly once.
    
    Explanation:
        - Mocks MongoDB collection and cursor.
        - Checks that the aggregate() method on the collection is invoked.
        - Checks that the cursor's to_list() coroutine is awaited exactly once.
        - Ensures the pipeline execution mechanism functions correctly.
    """
    since = datetime.now(timezone.utc) - timedelta(days=30)
    until = datetime.now(timezone.utc)

    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(return_value=[{"total_queries": 1}])

    mock_collection = MagicMock()
    mock_collection.aggregate.return_value = mock_cursor

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    monkeypatch.setattr(
        "app.monitoring.aggregation.evaluation_aggregator.db",
        mock_db
    )

    await aggregate_evaluations(since, until)

    assert mock_collection.aggregate.called
    mock_cursor.to_list.assert_awaited_once_with(1)
