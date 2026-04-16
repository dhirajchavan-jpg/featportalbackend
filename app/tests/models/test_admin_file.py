# app/tests/models/test_admin_file.py
import pytest
from datetime import datetime
from pydantic import ValidationError
from app.models.Admin_file import GlobalFileUploadForm, GlobalFileDB


# -------------------------------
# Tests for GlobalFileUploadForm
# -------------------------------

def test_global_file_upload_form_valid_data():
    """
    Summary:
        Validates that a properly filled GlobalFileUploadForm instance
        is created successfully.

    Explanation:
        GlobalFileUploadForm is a Pydantic model representing a file upload form.
        This test ensures that providing all required fields (sector, document_type,
        category, description, effective_date, version) results in a valid object
        with correct attributes.
    """
    form = GlobalFileUploadForm(
        sector="RBI",
        document_type="circular",
        category="Cybersecurity",
        description="Test description",
        effective_date="2026-01-01",
        version="v1.0"
    )

    assert form.sector == "RBI"
    assert form.document_type == "circular"
    assert form.category == "Cybersecurity"
    assert form.description == "Test description"
    assert form.effective_date == "2026-01-01"
    assert form.version == "v1.0"


def test_global_file_upload_form_required_fields():
    """
    Summary:
        Ensures that omitting required fields raises ValidationError.

    Explanation:
        Pydantic enforces required fields in models. Missing any of the
        required fields (sector, document_type, category) should trigger
        a ValidationError.
    """
    # Missing sector should raise ValidationError
    with pytest.raises(ValidationError):
        GlobalFileUploadForm(
            document_type="circular",
            category="Cybersecurity"
        )

    # Missing document_type should raise ValidationError
    with pytest.raises(ValidationError):
        GlobalFileUploadForm(
            sector="RBI",
            category="Cybersecurity"
        )

    # Missing category should raise ValidationError
    with pytest.raises(ValidationError):
        GlobalFileUploadForm(
            sector="RBI",
            document_type="circular"
        )


# -------------------------------
# Tests for GlobalFileDB
# -------------------------------

def test_global_file_db_valid_data():
    """
    Summary:
        Validates that a properly filled GlobalFileDB instance
        is created successfully.

    Explanation:
        GlobalFileDB is a Pydantic model representing stored files.
        This test ensures that when all required and optional fields
        are provided, the instance is valid and attributes match.
        The default field `is_active` is checked to ensure it defaults to True.
    """
    now = datetime.utcnow()
    file_db = GlobalFileDB(
        sector="GDPR",
        document_type="guidelines",
        category="Compliance",
        description="Some description",
        effective_date="2026-01-01",
        version="v2.0",
        filename="file.pdf",
        uploaded_by="user1",
        file_hash="abc123",
        file_id="file_001",
        file_path="/path/to/file.pdf",
        file_url="http://example.com/file.pdf",
        created_at=now,
        updated_at=now
    )

    assert file_db.sector == "GDPR"
    assert file_db.filename == "file.pdf"
    assert file_db.is_active is True  # default value
    assert file_db.created_at == now
    assert file_db.updated_at == now


def test_global_file_db_missing_required_fields():
    """
    Summary:
        Ensures that omitting required fields in GlobalFileDB raises ValidationError.

    Explanation:
        Pydantic enforces required fields like `filename` and `file_id`.
        Omitting them should trigger a ValidationError to prevent invalid database records.
    """
    now = datetime.utcnow()
    # Missing filename should raise ValidationError
    with pytest.raises(ValidationError):
        GlobalFileDB(
            sector="GDPR",
            document_type="guidelines",
            category="Compliance",
            uploaded_by="user1",
            file_hash="abc123",
            file_id="file_001",
            file_path="/path/to/file.pdf",
            created_at=now,
            updated_at=now
        )

    # Missing file_id should raise ValidationError
    with pytest.raises(ValidationError):
        GlobalFileDB(
            sector="GDPR",
            document_type="guidelines",
            category="Compliance",
            filename="file.pdf",
            uploaded_by="user1",
            file_hash="abc123",
            file_path="/path/to/file.pdf",
            created_at=now,
            updated_at=now
        )


def test_global_file_db_optional_fields_default():
    """
    Summary:
        Checks default values for optional fields in GlobalFileDB.

    Explanation:
        Optional fields like description, effective_date, version, and file_url
        should default to None if not provided. The `is_active` field should default
        to True to indicate active files.
    """
    now = datetime.utcnow()
    file_db = GlobalFileDB(
        sector="GDPR",
        document_type="guidelines",
        category="Compliance",
        filename="file.pdf",
        uploaded_by="user1",
        file_hash="abc123",
        file_id="file_001",
        file_path="/path/to/file.pdf",
        created_at=now,
        updated_at=now
    )

    assert file_db.description is None
    assert file_db.effective_date is None
    assert file_db.version is None
    assert file_db.file_url is None
    assert file_db.is_active is True
