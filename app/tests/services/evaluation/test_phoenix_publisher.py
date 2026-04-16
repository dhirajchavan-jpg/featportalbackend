import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import logging

# --- FIX: Updated import to match your filename 'phoenix_publisher.py' ---
from app.services.evaluation.phoenix_publisher import push_evaluation_to_phoenix

@pytest.fixture
def mock_phoenix_client():
    """
    Mocks the global 'client' object inside the phoenix_publisher module.
    This prevents real network calls to localhost:6006.
    """
    # --- FIX: Patch the client specifically in the publisher module ---
    with patch("app.services.evaluation.phoenix_publisher.client") as mock_client:
        yield mock_client

def test_push_evaluation_success(mock_phoenix_client, caplog):
    """Test that valid data is converted to a DataFrame and logged."""
    
    # 1. Setup Input Data
    eval_data = {
        "span_id": "span_123",
        "input": "User Query",
        "output": "AI Response",
        "hallucination_label": "clean",
        "hallucination_score": 0.0,
        "qa_correctness_label": "correct",
        "qa_correctness_score": 1.0,
        "model": "gpt-4",
        "timestamp": "2024-01-01T12:00:00"
    }

    # 2. Call the function
    with caplog.at_level(logging.INFO):
        push_evaluation_to_phoenix(eval_data)

    # 3. Verify client.log_evaluations was called
    mock_phoenix_client.log_evaluations.assert_called_once()

    # 4. Deep Inspection: Verify the DataFrame passed to the client
    call_args = mock_phoenix_client.log_evaluations.call_args
    # The argument 'evaluations' is passed as a keyword argument
    df_passed = call_args.kwargs['evaluations']

    # Assertions on the DataFrame content
    assert isinstance(df_passed, pd.DataFrame)
    assert len(df_passed) == 1
    assert df_passed.iloc[0]["span_id"] == "span_123"
    assert df_passed.iloc[0]["hallucination_score"] == 0.0
    assert df_passed.iloc[0]["model"] == "gpt-4"

    # 5. Verify Success Log
    assert "[PHOENIX] Evaluation pushed successfully" in caplog.text

def test_push_evaluation_missing_keys(mock_phoenix_client):
    """Test behavior when the input dict is missing optional keys."""
    # Only provide span_id (minimal valid record for tracing)
    eval_data = {"span_id": "span_456"}

    push_evaluation_to_phoenix(eval_data)

    # Extract the DataFrame
    df_passed = mock_phoenix_client.log_evaluations.call_args.kwargs['evaluations']

    # Assert keys exist but are None (NaN in pandas usually)
    assert df_passed.iloc[0]["span_id"] == "span_456"
    assert pd.isna(df_passed.iloc[0]["hallucination_score"]) 
    assert pd.isna(df_passed.iloc[0]["output"])

def test_push_evaluation_failure_handling(mock_phoenix_client, caplog):
    """Test that exceptions during logging are caught and logged, preventing crash."""
    
    # 1. Simulate an error in the Phoenix client
    mock_phoenix_client.log_evaluations.side_effect = Exception("Connection Refused")

    eval_data = {"span_id": "test"}

    # 2. Call function (should NOT raise exception)
    with caplog.at_level(logging.ERROR):
        push_evaluation_to_phoenix(eval_data)

    # 3. Verify Error Log
    assert "[PHOENIX] Push failed: Connection Refused" in caplog.text