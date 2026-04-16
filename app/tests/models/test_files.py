# app/tests/models/test_files.py
import pytest
from datetime import datetime, timedelta
from app.models.Files import FileModel
from bson import ObjectId
from pydantic import ValidationError


# -------------------------------
# Tests for FileModel
# -------------------------------

def test_file_model_required_fields():
    """
    Summary:
        Ensures that a FileModel instance can be created with all required fields
        and that optional fields are correctly initialized.

    Explanation:
        FileModel represents a file uploaded for a project. Required fields include:
        - project_id
        - filename
        - file_url
        - sector
        - category
        - compliance_type

        Optional fields such as file_id, file_hash, and user_id should be None if not provided.
        created_at should be automatically set to the current timestamp.
    """
    file = FileModel(
        project_id="proj_001",
        filename="document",
        file_url="/files/document.pdf",
        sector="RBI",
        category="Regulatory",
        compliance_type="Circular"
    )

    # Check that required fields are correctly set
    assert file.project_id == "proj_001"
    assert file.filename == "document"
    assert file.file_url == "/files/document.pdf"
    assert file.sector == "RBI"
    assert file.category == "Regulatory"
    assert file.compliance_type == "Circular"

    # Optional fields should be None or auto-generated
    assert file.file_id is None
    assert file.file_hash is None
    assert file.user_id is None
    assert isinstance(file.created_at, datetime)
    # Ensure timestamp is "recent"
    assert datetime.utcnow() - file.created_at < timedelta(seconds=5)


def test_file_model_missing_required_fields():
    """
    Summary:
        Validates that missing required fields in FileModel raise a ValidationError.

    Explanation:
        Pydantic ensures that all required fields must be present. If any of the following
        fields are missing: project_id, filename, file_url, sector, category, or compliance_type,
        a ValidationError should be raised. This prevents creating invalid file objects.
    """
    # Missing project_id
    with pytest.raises(ValidationError):
        FileModel(
            filename="doc",
            file_url="/files/doc.pdf",
            sector="RBI",
            category="Regulatory",
            compliance_type="Circular"
        )

    # Missing filename
    with pytest.raises(ValidationError):
        FileModel(
            project_id="proj_001",
            file_url="/files/doc.pdf",
            sector="RBI",
            category="Regulatory",
            compliance_type="Circular"
        )

    # Missing file_url
    with pytest.raises(ValidationError):
        FileModel(
            project_id="proj_001",
            filename="doc",
            sector="RBI",
            category="Regulatory",
            compliance_type="Circular"
        )

    # Missing sector
    with pytest.raises(ValidationError):
        FileModel(
            project_id="proj_001",
            filename="doc",
            file_url="/files/doc.pdf",
            category="Regulatory",
            compliance_type="Circular"
        )

    # Missing category
    with pytest.raises(ValidationError):
        FileModel(
            project_id="proj_001",
            filename="doc",
            file_url="/files/doc.pdf",
            sector="RBI",
            compliance_type="Circular"
        )

    # Missing compliance_type
    with pytest.raises(ValidationError):
        FileModel(
            project_id="proj_001",
            filename="doc",
            file_url="/files/doc.pdf",
            sector="RBI",
            category="Regulatory"
        )


def test_file_model_custom_values():
    """
    Summary:
        Ensures that optional fields and custom values can be correctly set
        when creating a FileModel instance.

    Explanation:
        This test verifies that attributes like file_id, file_hash, user_id,
        and created_at can be explicitly provided and are preserved correctly
        in the model instance.
    """
    custom_dt = datetime(2026, 1, 1, 12, 0, 0)
    file = FileModel(
        file_id="file_001",
        project_id="proj_001",
        filename="doc",
        file_url="/files/doc.pdf",
        sector="RBI",
        category="Regulatory",
        compliance_type="Circular",
        file_hash="abcd1234",
        user_id="user_001",
        created_at=custom_dt
    )

    assert file.file_id == "file_001"
    assert file.file_hash == "abcd1234"
    assert file.user_id == "user_001"
    assert file.created_at == custom_dt


def test_file_model_json_serialization():
    """
    Summary:
        Verifies that FileModel can be correctly serialized to JSON.

    Explanation:
        Pydantic models provide `model_dump_json` for converting to JSON strings.
        This test ensures that fields such as file_id and created_at are serialized
        as strings and that JSON serialization does not fail.
    """
    file = FileModel(
        project_id="proj_001",
        filename="doc",
        file_url="/files/doc.pdf",
        sector="RBI",
        category="Regulatory",
        compliance_type="Circular",
        file_id=str(ObjectId()),  # Convert ObjectId to string
    )

    json_str = file.model_dump_json()
    import json
    json_dict = json.loads(json_str)
    assert isinstance(json_dict["file_id"], str)
    assert isinstance(json_dict["created_at"], str)
