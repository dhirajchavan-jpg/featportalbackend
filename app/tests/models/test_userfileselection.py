# app/tests/models/test_user_file_selection.py
import pytest
from datetime import datetime
from pydantic import ValidationError
from app.models.UserFileSelection import UserFileSelectionModel  # adjust import as needed


# -------------------------------
# Test required fields in UserFileSelectionModel
# -------------------------------
def test_user_file_selection_required_fields():
    """
    Summary:
        Ensures that the UserFileSelectionModel enforces all required fields: file_id, filename, and source.
    
    Explanation:
        - Omitting any required field should raise a Pydantic ValidationError.
        - Verifies default values:
            - deselected defaults to True
            - project_id defaults to None
        - created_at is automatically set and is a datetime instance.
    """
    # Missing file_id
    with pytest.raises(ValidationError):
        UserFileSelectionModel(filename="file.txt", source="admin")

    # Missing filename
    with pytest.raises(ValidationError):
        UserFileSelectionModel(file_id="123", source="admin")

    # Missing source
    with pytest.raises(ValidationError):
        UserFileSelectionModel(file_id="123", filename="file.txt")

    # All required fields provided
    selection = UserFileSelectionModel(file_id="123", filename="file.txt", source="admin")
    assert selection.file_id == "123"
    assert selection.filename == "file.txt"
    assert selection.source == "admin"
    assert selection.deselected is True  # default value
    assert selection.project_id is None
    assert isinstance(selection.created_at, datetime)


# -------------------------------
# Test optional project_id field
# -------------------------------
def test_user_file_selection_optional_project_id():
    """
    Summary:
        Validates that project_id is optional and can be set if provided.
    
    Explanation:
        - project_id links the selection to a specific project.
        - If provided, it should be correctly stored in the model.
    """
    selection = UserFileSelectionModel(
        file_id="123",
        filename="file.txt",
        source="admin",
        project_id="proj_001"
    )
    assert selection.project_id == "proj_001"


# -------------------------------
# Test deselected flag
# -------------------------------
def test_user_file_selection_deselected_flag():
    """
    Summary:
        Ensures that the deselected field can be overridden.
    
    Explanation:
        - Default value is True.
        - Can be explicitly set to False.
    """
    selection = UserFileSelectionModel(
        file_id="123",
        filename="file.txt",
        source="admin",
        deselected=False
    )
    assert selection.deselected is False


# -------------------------------
# Test JSON serialization
# -------------------------------
def test_user_file_selection_json_serialization():
    """
    Summary:
        Validates that the model can be serialized to JSON correctly.
    
    Explanation:
        - Ensures all fields are present in the JSON string.
        - Uses Pydantic v2 method `model_dump_json()`.
    """
    selection = UserFileSelectionModel(file_id="123", filename="file.txt", source="admin")
    json_data = selection.model_dump_json()  # Pydantic v2 method for JSON serialization
    assert "123" in json_data
    assert "file.txt" in json_data
    assert "admin" in json_data
    assert "created_at" in json_data
