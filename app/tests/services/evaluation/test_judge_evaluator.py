import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json
from datetime import datetime

# --- FIX: Updated import path to match your filename 'gemini_judge_evaluator.py' ---
from app.services.evaluation.gemini_judge_evaluator import (
    GeminiJudgeEvaluator,
    RetrievalQualityEvaluator,
    get_gemini_judge_evaluator,
    get_retrieval_judge_evaluator
)

# --- Fixtures ---

@pytest.fixture
def mock_genai():
    """Mocks the google.generativeai module."""
    # We patch the module where it is imported IN THE SOURCE FILE
    with patch("app.services.evaluation.gemini_judge_evaluator.genai") as mock:
        # Mock the GenerativeModel constructor and instance
        mock_model_instance = MagicMock()
        mock.GenerativeModel.return_value = mock_model_instance
        yield mock, mock_model_instance

@pytest.fixture
def gemini_judge(mock_genai):
    """Returns an instance of GeminiJudgeEvaluator with mocked model."""
    return GeminiJudgeEvaluator(api_key="test_key")

@pytest.fixture
def retrieval_judge(mock_genai):
    """Returns an instance of RetrievalQualityEvaluator with mocked model."""
    return RetrievalQualityEvaluator(api_key="test_key")

# --- GeminiJudgeEvaluator Tests ---

@pytest.mark.asyncio
async def test_evaluate_response_success(gemini_judge, mock_genai):
    """Test successful evaluation with valid JSON response."""
    _, mock_model = mock_genai
    
    # Mock Gemini response text
    mock_response_json = {
        "faithfulness": {"score": 90, "explanation": "Good"},
        "relevance": {"score": 85, "explanation": "Relevant"},
        "completeness": {"score": 80, "explanation": "Mostly complete"},
        "hallucination": {"score": 0, "explanation": "None"},
        "query_toxicity": {"score": 0, "explanation": "Safe"},
        "response_toxicity": {"score": 5, "explanation": "Professional"}
    }
    
    # Configure mock to return an object with .text attribute
    mock_response_obj = MagicMock()
    mock_response_obj.text = json.dumps(mock_response_json)
    mock_model.generate_content.return_value = mock_response_obj

    # Inputs
    query = "What is X?"
    response = "X is Y."
    chunks = [{"page_content": "X is Y", "metadata": {"file_name": "doc1"}}]

    result = await gemini_judge.evaluate_response(query, response, chunks)

    # Verify basic parsing
    assert result["faithfulness_score"] == 90
    assert result["hallucination_score"] == 0
    assert result["query_toxicity_score"] == 0
    
    # Verify Overall Score Calculation
    # Scores: Faith(90), Rel(85), Comp(80) -> Positive
    # Scores: Hallucination(0) -> Inverted to 100
    # Scores: RespToxicity(5) -> Inverted to 95
    # Avg: (90 + 85 + 80 + 100 + 95) / 5 = 450 / 5 = 90
    assert result["overall_score"] == 90.0
    
    mock_model.generate_content.assert_called_once()

@pytest.mark.asyncio
async def test_evaluate_response_parsing_markdown(gemini_judge, mock_genai):
    """Test parsing when Gemini wraps JSON in markdown code blocks."""
    _, mock_model = mock_genai
    
    # JSON wrapped in ```json ... ```
    raw_text = """
    Here is the evaluation:
    ```json
    {
        "faithfulness": {"score": 100}
    }
    ```
    """
    mock_response_obj = MagicMock()
    mock_response_obj.text = raw_text
    mock_model.generate_content.return_value = mock_response_obj

    result = await gemini_judge.evaluate_response("q", "a", [])
    
    assert result["faithfulness_score"] == 100

@pytest.mark.asyncio
async def test_evaluate_response_api_failure(gemini_judge, mock_genai):
    """Test handling of Gemini API errors (e.g., connection issues)."""
    _, mock_model = mock_genai
    
    # Simulate API crashing
    mock_model.generate_content.side_effect = Exception("API Down")

    # Should not crash, but return None scores
    result = await gemini_judge.evaluate_response("q", "a", [])
    
    assert result["faithfulness_score"] is None
    assert "Error: API Down" in result["faithfulness_explanation"]
    assert result["overall_score"] is None

@pytest.mark.asyncio
async def test_evaluate_response_malformed_json(gemini_judge, mock_genai):
    """Test handling of invalid JSON returned by Gemini."""
    _, mock_model = mock_genai
    
    mock_response_obj = MagicMock()
    mock_response_obj.text = "This is not JSON."
    mock_model.generate_content.return_value = mock_response_obj

    result = await gemini_judge.evaluate_response("q", "a", [])
    
    # Should result in None scores because parsing failed
    assert result["faithfulness_score"] is None
    assert result["overall_score"] is None

def test_format_retrieved_chunks(gemini_judge):
    """Test string formatting of chunks."""
    chunks = [
        {"page_content": "Content A", "metadata": {"file_name": "Doc1", "page_number": 1}},
        {"content": "Content B", "metadata": {}} # Test fallback keys
    ]
    
    formatted = gemini_judge._format_retrieved_chunks(chunks)
    
    assert "[Chunk 1] Source: Doc1, Page: 1" in formatted
    assert "Content A" in formatted
    assert "[Chunk 2]" in formatted
    assert "Content B" in formatted

# --- RetrievalQualityEvaluator Tests ---

@pytest.mark.asyncio
async def test_evaluate_retrieved_chunks_success(retrieval_judge, mock_genai):
    """Test retrieval evaluation success."""
    _, mock_model = mock_genai
    
    mock_data = {
        "overall_score": 75,
        "chunk_scores": [100, 50],
        "num_relevant_chunks": 2,
        "explanation": "Decent"
    }
    
    mock_response_obj = MagicMock()
    mock_response_obj.text = json.dumps(mock_data)
    mock_model.generate_content.return_value = mock_response_obj

    chunks = [{"content": "c1"}, {"content": "c2"}]
    result = await retrieval_judge.evaluate_retrieved_chunks("q", chunks)
    
    assert result["overall_score"] == 75
    assert result["num_relevant_chunks"] == 2
    assert result["num_chunks_evaluated"] == 2

@pytest.mark.asyncio
async def test_evaluate_retrieved_chunks_error(retrieval_judge, mock_genai):
    """Test retrieval evaluation API error."""
    _, mock_model = mock_genai
    mock_model.generate_content.side_effect = Exception("Fail")
    
    chunks = [{"content": "c1"}]
    result = await retrieval_judge.evaluate_retrieved_chunks("q", chunks)
    
    assert result["overall_score"] is None
    assert "error" in result

# --- Retry Logic Tests ---

@pytest.mark.asyncio
async def test_call_gemini_retry_logic(gemini_judge, mock_genai):
    """Test that _call_gemini retries on failure."""
    _, mock_model = mock_genai
    
    # Fail twice, then succeed
    mock_success = MagicMock()
    mock_success.text = "{}"
    
    mock_model.generate_content.side_effect = [
        Exception("Fail 1"),
        Exception("Fail 2"),
        mock_success
    ]
    
    # We patch asyncio.sleep to speed up tests
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await gemini_judge._call_gemini("prompt")
        
        # Should have slept twice
        assert mock_sleep.call_count == 2
        # Should have called generate_content 3 times
        assert mock_model.generate_content.call_count == 3

@pytest.mark.asyncio
async def test_call_gemini_max_retries_exceeded(gemini_judge, mock_genai):
    """Test that it raises exception after max retries."""
    _, mock_model = mock_genai
    mock_model.generate_content.side_effect = Exception("Persistent Failure")
    
    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(Exception, match="Persistent Failure"):
            await gemini_judge._call_gemini("prompt", max_retries=2)
            
    assert mock_model.generate_content.call_count == 2

# --- Singleton Tests ---

def test_singleton_getters():
    """Verify singleton pattern works."""
    # Reset instances if they exist from previous tests (hack for testing)
    if hasattr(get_gemini_judge_evaluator, "_instance"):
        del get_gemini_judge_evaluator._instance
        
    g1 = get_gemini_judge_evaluator()
    g2 = get_gemini_judge_evaluator()
    
    assert g1 is g2
    assert isinstance(g1, GeminiJudgeEvaluator)

    if hasattr(get_retrieval_judge_evaluator, "_instance"):
        del get_retrieval_judge_evaluator._instance

    r1 = get_retrieval_judge_evaluator()
    r2 = get_retrieval_judge_evaluator()
    assert r1 is r2
    assert isinstance(r1, RetrievalQualityEvaluator)