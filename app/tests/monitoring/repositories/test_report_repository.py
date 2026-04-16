# tests/test_report_repository.py
import pytest
from unittest.mock import AsyncMock, patch
from app.monitoring.repositories import report_repository as repo
from bson import ObjectId

# -------------------------------
# Test: save report to DB
# -------------------------------
@pytest.mark.asyncio
async def test_save_report():
    """
    Summary:
        Tests the `ReportRepository.save` method to ensure that a report dictionary
        is correctly inserted into the MongoDB collection and the inserted `_id`
        is returned as a string.
    
    Explanation:
        - Creates a sample report dict with minimal required fields.
        - Mocks MongoDB `insert_one` to simulate insertion and return a fake ObjectId.
        - Asserts that the inserted document contains the correct keys (`report_type`, `metrics`).
        - Ensures the returned `_id` from the repository is converted from ObjectId to string.
    """
    test_report = {"report_type": "DAILY", "metrics": {}}
    fake_id = ObjectId()

    mock_insert_result = AsyncMock()
    mock_insert_result.inserted_id = fake_id

    with patch.object(repo.ReportRepository, "collection") as mock_collection:
        mock_collection.insert_one = AsyncMock(return_value=mock_insert_result)

        result = await repo.ReportRepository.save(test_report.copy())

        # Ensure insert_one was called with correct data
        called_arg = mock_collection.insert_one.call_args[0][0]
        assert called_arg["report_type"] == "DAILY"
        assert "metrics" in called_arg

        # Check that the returned _id is string
        assert result["_id"] == str(fake_id)


# -------------------------------
# Test: fetch multiple reports from DB
# -------------------------------
@pytest.mark.asyncio
async def test_fetch_reports():
    """
    Summary:
        Tests the `ReportRepository.fetch` method to retrieve a list of reports
        filtered by report_type and limited by count.
    
    Explanation:
        - Mocks MongoDB `find().sort().limit().to_list()` chain to return multiple fake reports.
        - Each report has a Mongo ObjectId that should be converted to string.
        - Asserts:
            - find called with correct filter
            - to_list called with correct limit
            - each returned report contains _id as string
            - the total number of reports matches the limit
    """
    fake_reports = [
        {"_id": ObjectId(), "report_type": "DAILY", "metrics": {}},
        {"_id": ObjectId(), "report_type": "DAILY", "metrics": {}},
    ]

    mock_cursor = AsyncMock()
    mock_cursor.to_list = AsyncMock(return_value=fake_reports)

    with patch.object(repo.ReportRepository, "collection") as mock_collection:
        mock_collection.find.return_value.sort.return_value.limit.return_value = mock_cursor

        result = await repo.ReportRepository.fetch("DAILY", limit=2)

        # Ensure DB calls executed correctly
        mock_collection.find.assert_called_once_with({"report_type": "DAILY"})
        mock_cursor.to_list.assert_called_once_with(length=2)

        # Validate _id conversion and number of reports
        for r in result:
            assert isinstance(r["_id"], str)
        assert len(result) == 2


# -------------------------------
# Test: fetch reports when DB returns empty
# -------------------------------
@pytest.mark.asyncio
async def test_fetch_reports_empty():
    """
    Summary:
        Tests `ReportRepository.fetch` when no reports exist for the given report_type.
    
    Explanation:
        - Mocks MongoDB cursor to return an empty list via `to_list`.
        - Ensures that the repository correctly returns an empty list.
        - Validates that the function handles "no reports found" scenario gracefully
          without raising exceptions.
    """
    mock_cursor = AsyncMock()
    mock_cursor.to_list = AsyncMock(return_value=[])

    with patch.object(repo.ReportRepository, "collection") as mock_collection:
        mock_collection.find.return_value.sort.return_value.limit.return_value = mock_cursor

        result = await repo.ReportRepository.fetch("DAILY", limit=5)

        # Validate empty result
        assert result == []
