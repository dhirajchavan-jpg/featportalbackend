# app/tests/models/test_project_config.py
import pytest
from datetime import datetime
from app.models.project_config import ProjectConfig, ProjectConfigPatch, IST


# -------------------------------
# Tests for ProjectConfig model
# -------------------------------

def test_project_config_defaults():
    """
    Summary:
        Tests that ProjectConfig initializes with correct default values.

    Explanation:
        ProjectConfig defines configuration options for a project, such as:
        - router_model, simple_model, complex_model
        - search_strategy, retrieval_depth, enable_reranking
        - ocr_engine, chat_history_limit
        - created_at, updated_at timestamps with IST timezone

        This test ensures that all defaults are correctly assigned and
        that timestamps are timezone-aware.
    """
    config = ProjectConfig()

    assert config.router_model == "qwen2.5:0.5b"
    assert config.simple_model == "qwen2.5:1.5b-instruct"
    assert config.complex_model == "gemma:7b"
    assert config.search_strategy == "hybrid"
    assert config.retrieval_depth == 5
    assert config.enable_reranking is True
    assert config.ocr_engine == "easyocr"
    assert config.chat_history_limit == 5
    assert isinstance(config.created_at, datetime)
    assert isinstance(config.updated_at, datetime)
    assert config.created_at.tzinfo == IST
    assert config.updated_at.tzinfo == IST


def test_project_config_custom_values():
    """
    Summary:
        Tests that ProjectConfig can be initialized with custom values.

    Explanation:
        Users can override default configuration parameters during initialization.
        This test ensures that custom values are correctly applied.
    """
    config = ProjectConfig(router_model="custom-router", chat_history_limit=10)

    assert config.router_model == "custom-router"
    assert config.chat_history_limit == 10


def test_project_config_invalid_chat_history_limit():
    """
    Summary:
        Ensures ProjectConfig enforces valid chat_history_limit values.

    Explanation:
        chat_history_limit must be between 1 and 50 inclusive.
        Values outside this range should raise ValueError.
    """
    with pytest.raises(ValueError):
        ProjectConfig(chat_history_limit=0)  # Less than minimum allowed

    with pytest.raises(ValueError):
        ProjectConfig(chat_history_limit=100)  # Greater than maximum allowed


# -------------------------------
# Tests for ProjectConfigPatch model
# -------------------------------

def test_project_config_patch_all_optional():
    """
    Summary:
        Ensures ProjectConfigPatch initializes with all fields optional.

    Explanation:
        ProjectConfigPatch is used for patch updates. All fields should default
        to None so that only explicitly provided fields are updated.
    """
    patch = ProjectConfigPatch()
    for field in patch.__fields__:
        assert getattr(patch, field) is None


def test_project_config_patch_accepts_values():
    """
    Summary:
        Ensures ProjectConfigPatch accepts provided values.

    Explanation:
        Even though all fields are optional, when values are provided,
        they should be correctly assigned.
    """
    patch = ProjectConfigPatch(router_model="x", retrieval_depth=10, chat_history_limit=20)
    assert patch.router_model == "x"
    assert patch.retrieval_depth == 10
    assert patch.chat_history_limit == 20
