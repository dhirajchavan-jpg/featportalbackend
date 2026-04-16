# tests/test_drift_detector.py
import pytest
from unittest.mock import patch, AsyncMock
from datetime import datetime, timedelta
from app.monitoring.drift import drift_detector


# -------------------------------
# Test: detect_drifts when no drift occurs
# -------------------------------
@pytest.mark.asyncio
async def test_detect_drifts_no_drift():
    """
    Summary:
        Validates that `detect_drifts` correctly handles metrics that do not exceed any drift thresholds.
    
    Explanation:
        - All metric averages are below drift thresholds.
        - The function should mark all drifts as OK with severity NONE.
        - attribute_drift (which fetches affected queries) should not be called.
        - Mocks resolve_severity, resolve_status, and calculate_confidence to control output.
    """
    metrics = {
        "total_queries": 10,
        "hallucination_avg": 0.1,
        "retrieval_avg": 0.9,
        "knowledge_gap_rate": 0.0,
        "knowledge_coverage_gap_rate": 0.0,
        "query_toxicity_avg": 0.0,
        "response_toxicity_avg": 0.0
    }

    since = datetime.utcnow() - timedelta(days=1)
    until = datetime.utcnow()
    report_reference_id = "test-report"

    # Patch dependencies
    with patch("app.monitoring.drift.drift_detector.attribute_drift", new_callable=AsyncMock) as mock_attr, \
         patch("app.monitoring.drift.drift_detector.resolve_severity") as mock_sev, \
         patch("app.monitoring.drift.drift_detector.resolve_status") as mock_status, \
         patch("app.monitoring.drift.drift_detector.calculate_confidence") as mock_conf:

        # Mocks return default values indicating no drift
        mock_sev.return_value = None
        mock_status.return_value = "OK"
        mock_conf.return_value = "HIGH"

        result = await drift_detector.detect_drifts(metrics, since, until, report_reference_id)

        # Validate that all drift types are OK
        for drift in result:
            assert drift["status"] == "OK"
            assert drift["severity"] == "NONE"
            assert drift["scope_pct"] == 0
            assert drift["confidence"] == "HIGH"
            assert drift["affected_queries"]["count"] == 0

        # attribute_drift should not be called for metrics below threshold
        mock_attr.assert_not_called()


# -------------------------------
# Test: detect_drifts when drift occurs
# -------------------------------
@pytest.mark.asyncio
async def test_detect_drifts_with_drift():
    """
    Summary:
        Ensures `detect_drifts` correctly identifies metrics exceeding drift thresholds
        and populates severity, status, scope, confidence, and affected queries.

    Explanation:
        - Metrics are set to exceed thresholds for some drift types (e.g., hallucination_avg > 0.7).
        - Mocks attribute_drift to simulate affected query extraction.
        - Mocks resolve_severity, resolve_status, and calculate_confidence to produce expected values.
        - Checks that the resulting drift object has correct ALERT status and severity.
        - Validates that the scope percentage is calculated correctly.
    """
    metrics = {
        "total_queries": 10,
        "hallucination_avg": 0.8,  # triggers drift
        "retrieval_avg": 0.1,
        "knowledge_gap_rate": 0.5,
        "knowledge_coverage_gap_rate": 0.2,
        "query_toxicity_avg": 0.5,
        "response_toxicity_avg": 0.3
    }

    since = datetime.utcnow() - timedelta(days=1)
    until = datetime.utcnow()
    report_reference_id = "test-report"

    with patch("app.monitoring.drift.drift_detector.attribute_drift", new_callable=AsyncMock) as mock_attr, \
         patch("app.monitoring.drift.drift_detector.resolve_severity") as mock_sev, \
         patch("app.monitoring.drift.drift_detector.resolve_status") as mock_status, \
         patch("app.monitoring.drift.drift_detector.calculate_confidence") as mock_conf:

        # Setup mocks
        mock_sev.side_effect = lambda value, thresholds: "HIGH" if value > 0.7 else None
        mock_status.return_value = "ALERT"
        mock_conf.return_value = "STRONG"
        mock_attr.return_value = {
            "count": 5,
            "total": 10,
            "top_examples": [{"query": "example", "score": 0.8, "reason": "test"}]
        }

        result = await drift_detector.detect_drifts(metrics, since, until, report_reference_id)

        # Validate that HALLUCINATION drift is correctly detected
        hallu_drift = next(d for d in result if d["drift_type"] == "HALLUCINATION")
        assert hallu_drift["status"] == "ALERT"
        assert hallu_drift["severity"] == "HIGH"
        assert hallu_drift["confidence"] == "STRONG"
        assert hallu_drift["scope_pct"] == 50.0  # 5 affected / 10 total * 100
        assert hallu_drift["affected_queries"]["count"] == 5

        # Ensure attribute_drift was called to get affected queries
        mock_attr.assert_awaited()
