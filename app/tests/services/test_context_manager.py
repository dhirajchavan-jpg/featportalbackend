import pytest
from unittest.mock import MagicMock, patch, mock_open
import os
from typing import List, Dict, Any

# Adjust this import based on your actual folder structure
from app.services.context_manager import ContextManager, get_context_manager

# --- Fixtures ---

@pytest.fixture
def mock_settings():
    """Mock application settings."""
    with patch("app.services.context_manager.settings") as settings:
        settings.MAX_CONTEXT_TOKENS = 1000
        settings.ENABLE_DEDUPLICATION = True
        settings.PROMPTS_DIR = "/mock/prompts"
        yield settings

@pytest.fixture
def manager(mock_settings):
    """
    Initialize ContextManager with mocked file loading.
    We use a side_effect for open to differentiate between simple and complex files if needed,
    or just return static content.
    """
    with patch("app.services.context_manager.os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data="Mock System Prompt")):
        return ContextManager()

# --- Tests ---

def test_singleton_instance(mock_settings):
    """Verify that get_context_manager returns a singleton instance."""
    # Reset singleton manually for test isolation
    import app.services.context_manager as cm
    cm._context_manager = None
    
    with patch("app.services.context_manager.os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data="Prompt")):
        m1 = get_context_manager()
        m2 = get_context_manager()
        assert m1 is m2
        assert isinstance(m1, ContextManager)

def test_initialization_prompt_loading_success(mock_settings):
    """Test that prompts are loaded correctly from files."""
    with patch("app.services.context_manager.os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data="Loaded Prompt content")) as mock_file:
        
        cm = ContextManager()
        
        assert cm.prompt_simple == "Loaded Prompt content"
        assert cm.prompt_complex == "Loaded Prompt content"
        assert mock_file.call_count == 2 

def test_initialization_prompt_loading_failure(mock_settings):
    """Test fallback behavior when prompt files are missing."""
    with patch("app.services.context_manager.os.path.exists", return_value=False):
        cm = ContextManager()
        # Should rely on defaults defined in __init__
        assert cm.prompt_simple == "You are a helpful assistant." 
        assert cm.prompt_complex == "You are a senior analyst."

def test_build_context_standard_flow(manager):
    """Test standard context assembly with history and docs."""
    query = "test query"
    docs = [
        {"page_content": "Doc 1 content", "metadata": {"file_name": "f1.pdf", "page_number": 1}},
        {"page_content": "Doc 2 content", "metadata": {"file_name": "f2.pdf", "page_number": 2}}
    ]
    history = [{"user_query": "hi", "llm_answer": "hello"}]
    
    context = manager.build_context(
        query, 
        docs, 
        chat_history=history, 
        include_history=True,
        chat_history_limit=2
    )
    
    # Verify Structure
    assert "## Previous Conversation" in context
    assert "User: hi" in context
    assert "Assistant: hello" in context
    assert "## Relevant Information" in context
    assert "SOURCE: [File: f1.pdf, Page: 1]" in context
    assert "Doc 1 content" in context
    assert "SOURCE: [File: f2.pdf, Page: 2]" in context

def test_build_context_no_history(manager):
    """Test context assembly when history is disabled."""
    context = manager.build_context("q", [{"content": "c"}], include_history=False)
    assert "## Previous Conversation" not in context
    assert "## Relevant Information" in context

def test_build_context_empty_docs(manager):
    """Test handling of empty retrieved documents list."""
    context = manager.build_context("q", [])
    assert "## Relevant Information" not in context

def test_build_comparative_context(manager):
    """Test comparative context assembly (sector grouping)."""
    sector_results = [
        {"sector": "Finance", "chunks": [{"content": "Finance Rule 1", "metadata": {}}]},
        {"sector": "Healthcare", "chunks": []}, # Empty, should be skipped
        {"sector": "Energy", "chunks": [{"content": "Energy Rule 1", "metadata": {}}]}
    ]
    
    context = manager.build_comparative_context("q", sector_results)
    
    assert "# Cross-Sector Regulatory Context" in context
    assert "## Finance Sector" in context
    assert "Finance Rule 1" in context
    assert "## Energy Sector" in context
    assert "Energy Rule 1" in context
    assert "## Healthcare Sector" not in context

def test_build_comparative_context_all_empty(manager):
    """Test fallback when no sectors return data."""
    empty_results = [{"sector": "A", "chunks": []}]
    context = manager.build_comparative_context("q", empty_results)
    
    assert "No relevant information found" in context

def test_prepare_prompt_structure(manager):
    """Test construction of the messages array."""
    messages = manager.prepare_prompt(
        query="test query", 
        context="test context", 
        complexity="simple",
        style="Detailed" # Explicitly set default
    )
    
    assert isinstance(messages, list)
    assert len(messages) == 3
    
    # Check System Message
    assert messages[0]["role"] == "system"
    # FIXED: The source code appends the style overlay, so we check using 'in' or 'startswith'
    assert manager.prompt_simple in messages[0]["content"]
    assert "RESPONSE MODE: DETAILED" in messages[0]["content"]
    
    # Check Context Message
    assert messages[1]["role"] == "user"
    assert "=== CONTEXT DOCUMENTS ===" in messages[1]["content"]
    assert "test context" in messages[1]["content"]
    
    # Check Query Message
    assert messages[2]["role"] == "user"
    assert "=== USER QUESTION ===" in messages[2]["content"]
    assert "test query" in messages[2]["content"]

def test_prepare_prompt_with_style_overlay(manager):
    """Test that style overlays (Formal/Simple) are correctly appended."""
    # Test Formal
    messages = manager.prepare_prompt("q", "c", complexity="complex", style="Formal")
    content = messages[0]["content"]
    
    assert manager.prompt_complex in content
    assert "RESPONSE MODE: FORMAL" in content
    assert "Use bullet points only" in content

    # Test Simple (Should use prompt_simple + Simple overlay)
    messages_simple = manager.prepare_prompt("q", "c", complexity="simple", style="Simple")
    content_simple = messages_simple[0]["content"]
    
    assert manager.prompt_simple in content_simple
    assert "RESPONSE MODE: SIMPLE" in content_simple

def test_prepare_prompt_custom_system_message(manager):
    """Test override with custom system message."""
    custom_msg = "Custom Instruction"
    # Even with custom message, style overlay applies if 'style' arg is passed
    messages = manager.prepare_prompt("q", "c", system_message=custom_msg, style="Detailed")
    
    assert custom_msg in messages[0]["content"]
    assert "RESPONSE MODE: DETAILED" in messages[0]["content"]

def test_format_doc_helper(manager):
    """Test formatting of a single document chunk."""
    doc = {
        "page_content": "Content",
        "metadata": {"file_name": "Test.pdf", "page": 10}
    }
    
    # With Index
    fmt1 = manager._format_doc(doc, index=1)
    assert "[1] SOURCE: [File: Test.pdf, Page: 10]" in fmt1
    assert "Content" in fmt1
    
    # Without Metadata (Robustness check)
    doc_missing = {"content": "Content"}
    fmt2 = manager._format_doc(doc_missing)
    assert "SOURCE: [File: Unknown File, Page: ?]" in fmt2

def test_filter_history_by_sector(manager):
    """Test filtering chat history by sector."""
    history = [
        {"user_query": "q1", "sector": "Finance"},
        {"user_query": "q2", "sector": "Energy"},
        {"user_query": "q3", "sector": "Finance"}
    ]
    
    filtered = manager._filter_history_by_sector(history, "Finance")
    assert len(filtered) == 2
    assert filtered[0]["user_query"] == "q1"
    
    filtered_none = manager._filter_history_by_sector(history, "Tech")
    assert len(filtered_none) == 0

def test_format_chat_history_limit(manager):
    """Test that chat history respects the turn limit."""
    history = [
        {"user_query": f"q{i}", "llm_answer": f"a{i}"} 
        for i in range(10)
    ]
    
    # Limit to 3 recent turns
    formatted = manager._format_chat_history(history, limit=3)
    
    # Should contain q7, q8, q9
    assert "User: q9" in formatted
    assert "User: q8" in formatted
    assert "User: q7" in formatted
    assert "User: q6" not in formatted

def test_deduplicate_content_logic(manager):
    """Test paragraph deduplication."""
    raw_context = "Para 1.\n\nPara 1.\n\nPara 2."
    deduped = manager._deduplicate_content(raw_context)
    
    assert deduped.count("Para 1.") == 1
    assert "Para 2." in deduped

def test_smart_truncate_context(manager):
    """Test context truncation based on length."""
    manager.max_tokens = 10 
    long_text = "A" * 200 
    
    truncated = manager._smart_truncate_context(long_text, multiplier=2)
    
    assert "[...Context truncated...]" in truncated
    assert len(truncated) < 200

def test_estimate_tokens(manager):
    text = "one two three"
    assert manager.estimate_tokens(text) == 3

def test_get_context_stats(manager):
    stats = manager.get_context_stats("word word word")
    assert stats["length"] == 14
    assert stats["estimated_tokens"] == 3