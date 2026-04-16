
import pytest
from unittest.mock import patch, AsyncMock
from datetime import datetime, timedelta
from app.monitoring.reports import reports_service  # assuming your file is named reports_service.py

@pytest.mark.asyncio
async def test_generate_report_daily():
    report_type = "DAILY"

    # Patch all external dependencies
    with patch("app.monitoring.reports.reports_service.aggregate_evaluations", new_callable=AsyncMock) as mock_agg, \
         patch("app.monitoring.reports.reports_service.detect_drifts", new_callable=AsyncMock) as mock_drifts, \
         patch("app.monitoring.reports.reports_service.generate_narrative") as mock_narrative, \
         patch("app.monitoring.reports.reports_service.generate_recommendations") as mock_recs, \
         patch("app.monitoring.reports.reports_service.rp.ReportRepository.save", new_callable=AsyncMock) as mock_save:

        # Setup return values
        mock_agg.return_value = {
            "faithfulness_avg": 0.8,
            "hallucination_avg": 0.1,
            "overall_score_avg": 0.75,
            "retrieval_avg": 0.6,
            "query_toxicity_avg": 0,
            "response_toxicity_avg": 0,
            "pipeline_health_avg": 0.9,
            "total_queries": 5,
            "knowledge_gap_rate": 0.2,
            "knowledge_coverage_gap_rate": 0.1
        }

        mock_drifts.return_value = [
            {"drift_type": "HALLUCINATION", "status": "OK"}
        ]
        mock_narrative.return_value = "Narrative text"
        mock_recs.return_value = ["Recommendation 1"]

        report = await reports_service.generate_report(report_type)

        # Assertions
        assert report["report_type"] == report_type
        assert report["window_days"] == 1
        assert "metrics" in report
        assert "drifts" in report
        assert report["narrative"] == "Narrative text"
        assert report["recommendations"] == []  # DAILY report should not generate recommendations
        assert report["report_reference_id"].startswith(f"RPT-{report_type}-")
        mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_report_weekly():
    report_type = "WEEKLY"

    with patch("app.monitoring.reports.reports_service.aggregate_evaluations", new_callable=AsyncMock) as mock_agg, \
         patch("app.monitoring.reports.reports_service.detect_drifts", new_callable=AsyncMock) as mock_drifts, \
         patch("app.monitoring.reports.reports_service.generate_narrative") as mock_narrative, \
         patch("app.monitoring.reports.reports_service.generate_recommendations") as mock_recs, \
         patch("app.monitoring.reports.reports_service.rp.ReportRepository.save", new_callable=AsyncMock) as mock_save:

        mock_agg.return_value = {}
        mock_drifts.return_value = []
        mock_narrative.return_value = "Weekly narrative"
        mock_recs.return_value = ["Weekly recommendation"]

        report = await reports_service.generate_report(report_type)

        assert report["window_days"] == 7
        assert report["recommendations"] == ["Weekly recommendation"]
        mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_report_invalid_type():
    with pytest.raises(ValueError):
        await reports_service.generate_report("INVALID")
