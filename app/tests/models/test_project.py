# app/tests/models/test_project.py
import pytest
from datetime import datetime, timezone
from app.models.project import ProjectModel, ProjectUpdateSchema

# -------------------------------
# Tests for ProjectModel
# -------------------------------

def test_project_model_defaults():
    """
    Summary:
        Tests that ProjectModel initializes required and default fields correctly.

    Explanation:
        - Required fields: project_name, industry, description
        - Default values:
            - has_organizational_compliance -> False
            - third_party_framework -> empty list
            - sectors -> empty list
            - regulatory_framework -> None
            - organization_sector -> None
            - user_id -> None
        - Timestamps (created_at, updated_at) should be datetime objects with UTC timezone.
        This test ensures a new project instance has sensible defaults for optional fields.
    """
    project = ProjectModel(
        project_name="My Project",
        industry="Finance",
        description="Project description"
    )

    # Defaults
    assert project.has_organizational_compliance is False
    assert project.third_party_framework == []
    assert project.sectors == []
    assert project.regulatory_framework is None
    assert project.organization_sector is None
    assert project.user_id is None

    # Required fields
    assert project.project_name == "My Project"
    assert project.industry == "Finance"
    assert project.description == "Project description"

    # Timestamps
    assert isinstance(project.created_at, datetime)
    assert isinstance(project.updated_at, datetime)
    assert project.created_at.tzinfo == timezone.utc
    assert project.updated_at.tzinfo == timezone.utc


def test_project_model_all_fields():
    """
    Summary:
        Tests that ProjectModel correctly assigns all fields when provided.

    Explanation:
        - Ensures that optional fields can be set during initialization.
        - Verifies that all lists and boolean fields behave as expected.
        - This is useful for creating fully-specified project objects for testing or database insertion.
    """
    project = ProjectModel(
        project_name="Project X",
        industry="Tech",
        description="Desc",
        regulatory_framework="GDPR",
        third_party_framework=["ISO27001", "SOC2"],
        has_organizational_compliance=True,
        organization_sector="IT",
        sectors=["Regulatory", "Organization"],
        user_id="user123"
    )

    assert project.project_name == "Project X"
    assert project.industry == "Tech"
    assert project.description == "Desc"
    assert project.regulatory_framework == "GDPR"
    assert project.third_party_framework == ["ISO27001", "SOC2"]
    assert project.has_organizational_compliance is True
    assert project.organization_sector == "IT"
    assert project.sectors == ["Regulatory", "Organization"]
    assert project.user_id == "user123"


# -------------------------------
# Tests for ProjectUpdateSchema
# -------------------------------

def test_project_update_schema_all_optional():
    """
    Summary:
        Tests that ProjectUpdateSchema initializes all fields as optional (None by default).

    Explanation:
        - ProjectUpdateSchema is used for patch/update requests.
        - All fields should be optional to allow partial updates.
        - This test ensures no default values are set unless explicitly provided.
    """
    patch = ProjectUpdateSchema()
    assert patch.project_name is None
    assert patch.description is None
    assert patch.industry is None


def test_project_update_schema_with_values():
    """
    Summary:
        Tests that ProjectUpdateSchema correctly assigns provided values.

    Explanation:
        - Even though fields are optional, when provided, they should be assigned.
        - This is useful for updating only specific project attributes without affecting others.
    """
    patch = ProjectUpdateSchema(project_name="Updated Name", industry="Healthcare")
    assert patch.project_name == "Updated Name"
    assert patch.industry == "Healthcare"
    assert patch.description is None
