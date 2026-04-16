from app.core.phoenix_utils import _parse_eval
from unittest.mock import patch
from app.core.phoenix_utils import _evaluate_hallucination, _evaluate_qa_correctness


def test_parse_eval_json():
    """
    Summary:
        Verify that evaluator output provided as valid JSON is parsed correctly.

    Explanation:
        The evaluator may return structured JSON responses. This test ensures
        that the parser correctly extracts the evaluation label from JSON input.
    """
    text = '{"label": "factual"}'
    assert _parse_eval(text)["label"] == "factual"


def test_parse_eval_fallback_factual():
    """
    Summary:
        Ensure fallback parsing correctly identifies factual responses.

    Explanation:
        When the evaluator returns plain text instead of JSON, the system must
        infer the label based on keywords indicating factual correctness.
    """
    text = "This answer is factual and correct."
    assert _parse_eval(text)["label"] == "factual"


def test_parse_eval_fallback_hallucinated():
    """
    Summary:
        Ensure fallback parsing correctly identifies hallucinated responses.

    Explanation:
        This test validates that non-JSON evaluator output indicating hallucination
        is correctly classified to prevent false factual scoring.
    """
    text = "This response is hallucinated."
    assert _parse_eval(text)["label"] == "hallucinated"


def test_parse_eval_unknown():
    """
    Summary:
        Unknown or unrecognized evaluator output must be classified as unknown.

    Explanation:
        This prevents the system from making unsafe assumptions when the evaluator
        response does not match any known patterns.
    """
    assert _parse_eval("nonsense")["label"] == "unknown"



@patch("app.core.phoenix_utils._call_ollama")
def test_evaluate_hallucination_factual(mock_call):
    """
    Summary:
        Factual hallucination evaluation must return a factual label with full score.

    Explanation:
        When the underlying evaluation model returns a factual judgment, the
        hallucination evaluator must propagate the label and assign the maximum score.
    """
    mock_call.return_value = '{"label": "factual"}'

    result = _evaluate_hallucination("q", "r", "ctx")
    assert result["label"] == "factual"
    assert result["score"] == 1.0


@patch("app.core.phoenix_utils._call_ollama")
def test_evaluate_qa_incorrect(mock_call):
    """
    Summary:
        Incorrect QA evaluation must return an incorrect label with zero score.

    Explanation:
        This test ensures that QA correctness evaluation assigns a zero score when
        the answer is judged to be incorrect, preventing misleading quality signals.
    """
    mock_call.return_value = '{"label": "incorrect"}'

    result = _evaluate_qa_correctness("q", "r", "ctx")
    assert result["label"] == "incorrect"
    assert result["score"] == 0.0
