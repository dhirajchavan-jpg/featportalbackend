import pytest
from unittest.mock import MagicMock, patch

# CORRECT IMPORT: Pointing to app/middleware/
from app.middleware.moderation import is_compliance_related


class FakeChain:
    """
    Summary:
        Minimal fake LangChain runnable used for testing moderation logic.
    """
    def __init__(self, response=None, raise_exc=False):
        self.response = response
        self.raise_exc = raise_exc

    def invoke(self, _):
        if self.raise_exc:
            raise RuntimeError("LLM failure")
        return self.response


@pytest.fixture
def fake_llm():
    """
    Summary:
        Provides a mocked LLM dependency.
    """
    return MagicMock(name="FakeLLM")


# CORRECT PATCH PATH: Pointing to app.middleware.moderation
@patch("app.middleware.moderation.get_llm")
@patch("app.middleware.moderation.PromptTemplate.from_template")
def test_returns_true_when_llm_says_yes(mock_prompt, mock_get_llm, fake_llm):
    """
    Summary:
        Compliance-related queries should be accepted when LLM responds 'Yes'.
    """
    fake_chain = FakeChain("Yes")

    # Mock the chain execution: prompt | llm | parser
    mock_prompt.return_value.__or__.return_value.__or__.return_value = fake_chain
    mock_get_llm.return_value = fake_llm

    result = is_compliance_related("Explain RBI compliance rules")

    assert result is True
    mock_get_llm.assert_called_once()


# CORRECT PATCH PATH: Pointing to app.middleware.moderation
@patch("app.middleware.moderation.get_llm")
@patch("app.middleware.moderation.PromptTemplate.from_template")
def test_returns_false_when_llm_says_no(mock_prompt, mock_get_llm, fake_llm):
    """
    Summary:
        Non-compliance queries should be rejected by moderation.
    """
    fake_chain = FakeChain("No")

    mock_prompt.return_value.__or__.return_value.__or__.return_value = fake_chain
    mock_get_llm.return_value = fake_llm

    result = is_compliance_related("Tell me a joke")

    assert result is False


# CORRECT PATCH PATH: Pointing to app.middleware.moderation
@patch("app.middleware.moderation.get_llm")
@patch("app.middleware.moderation.PromptTemplate.from_template")
def test_strips_and_normalizes_llm_response(mock_prompt, mock_get_llm, fake_llm):
    """
    Summary:
        LLM responses must be normalized before evaluation.
    """
    fake_chain = FakeChain("  yes  ")

    mock_prompt.return_value.__or__.return_value.__or__.return_value = fake_chain
    mock_get_llm.return_value = fake_llm

    result = is_compliance_related("Explain GDPR penalties")

    assert result is True


# CORRECT PATCH PATH: Pointing to app.middleware.moderation
@patch("app.middleware.moderation.get_llm")
@patch("app.middleware.moderation.PromptTemplate.from_template")
def test_returns_false_on_llm_exception(mock_prompt, mock_get_llm, fake_llm):
    """
    Summary:
        Moderation must fail safely when the LLM errors.
    """
    fake_chain = FakeChain(raise_exc=True)

    mock_prompt.return_value.__or__.return_value.__or__.return_value = fake_chain
    mock_get_llm.return_value = fake_llm

    result = is_compliance_related("Explain compliance")

    assert result is False