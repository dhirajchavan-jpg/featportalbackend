# tests/test_drift_query_repository.py
import pytest
from unittest.mock import AsyncMock, patch
from app.monitoring.repositories import drift_query_repository as repo
from datetime import datetime

# -------------------------------
# Test: get_drift_queries when there are no drift references
# -------------------------------
@pytest.mark.asyncio
async def test_get_drift_queries_no_refs():
    """
    Summary:
        Tests the behavior of `get_drift_queries` when there are no drift query references
        stored in the database for a given report reference ID.
    
    Explanation:
        - Simulates an empty database query by returning an empty async generator.
        - Ensures that the function returns a dictionary with:
            - report_reference_id set correctly
            - total count = 0
            - items list is empty
        - Confirms that the function handles the "no drift references" scenario gracefully.
    """
    report_id = "RPT-TEST-123"

    # Create an async generator that yields nothing
    async def empty_cursor():
        for _ in []:
            yield _

    with patch("app.monitoring.repositories.drift_query_repository.db") as mock_db:
        mock_db.__getitem__.return_value.find.return_value = empty_cursor()

        result = await repo.get_drift_queries(report_id)

        # Assertions
        assert result["report_reference_id"] == report_id
        assert result["total"] == 0
        assert result["items"] == []


# -------------------------------
# Test: get_drift_queries when there are drift references with evaluated results
# -------------------------------
@pytest.mark.asyncio
async def test_get_drift_queries_with_refs():
    """
    Summary:
        Tests `get_drift_queries` with a report reference that has associated drift references
        and evaluated documents in the database.
    
    Explanation:
        - Simulates a database returning:
            1. One drift query reference
            2. One evaluation document that matches the reference filter
        - Ensures the function correctly:
            - Maps evaluation results to the drift reference type
            - Returns total count and top items
            - Includes fields like query, score, drift_type, and evaluated_at
        - Validates that nested score extraction (e.g., hallucination_score) works correctly.
    """
    report_id = "RPT-TEST-456"

    # Mock drift reference document
    ref_doc = {
        "report_reference_id": report_id,
        "drift_type": "HALLUCINATION",
        "query_filter": {"field": "generation.hallucination_score", "operator": "$gt", "threshold": 0.5},
        "since": datetime(2026, 1, 1),
        "until": datetime(2026, 1, 2)
    }

    # Mock evaluated document that matches the filter
    eval_doc = {
        "query": "What is AI?",
        "generation": {"hallucination_score": 0.7},
        "evaluated_at": datetime(2026, 1, 1, 12, 0)
    }

    # Async generators for mocking DB cursors
    async def refs_cursor():
        yield ref_doc

    async def eval_cursor():
        yield eval_doc

    with patch("app.monitoring.repositories.drift_query_repository.db") as mock_db:
        # Patch DB find() to return references first, then evaluation documents
        mock_db.__getitem__.return_value.find.side_effect = [refs_cursor(), eval_cursor()]

        result = await repo.get_drift_queries(report_id)

        # Assertions
        assert result["report_reference_id"] == report_id
        assert result["total"] == 1
        assert result["items"][0]["query"] == "What is AI?"
        assert result["items"][0]["score"] == 0.7
        assert result["items"][0]["drift_type"] == "HALLUCINATION"
        assert isinstance(result["items"][0]["evaluated_at"], datetime)
