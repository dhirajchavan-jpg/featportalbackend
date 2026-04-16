import pytest
from fastapi import HTTPException

from app.middleware.project_middleware import (
    validate_project_name,
    validate_industry,
    validate_description,
    validate_sectors,
    ALLOWED_SECTORS,
)

@pytest.mark.parametrize("name", [
    "My Project",
    "Project-123",
    "Compliance_RAG v2",
    "SOC2 & GDPR Analysis",
])
def test_validate_project_name_valid(name):
    """
    Summary:
        Valid project names should pass validation.

    Explanation:
        This test ensures that project names containing allowed characters,
        reasonable length, and valid formatting do not raise validation errors.
        These names reflect realistic compliance and enterprise use cases.
    """
    # Should not raise
    validate_project_name(name)


@pytest.mark.parametrize("name, expected_msg", [
    ("", "Project name must be at least 3 characters long."),
    ("  ", "Project name must be at least 3 characters long."),
    ("ab", "Project name must be at least 3 characters long."),
])
def test_validate_project_name_too_short(name, expected_msg):
    """
    Summary:
        Project names that are too short must be rejected.

    Explanation:
        This test validates minimum length enforcement and ensures
        user-facing error messages clearly explain why validation failed.
    """
    with pytest.raises(HTTPException) as exc:
        validate_project_name(name)
    assert exc.value.status_code == 422
    assert expected_msg in exc.value.detail


def test_validate_project_name_too_long():
    """
    Summary:
        Project names exceeding the maximum allowed length must fail.

    Explanation:
        This test confirms that overly long project names are rejected
        to prevent storage abuse, UI issues, or downstream processing errors.
    """
    name = "a" * 101
    with pytest.raises(HTTPException):
        validate_project_name(name)


@pytest.mark.parametrize("name", [
    "Project@123",
    "Test#!",
    "<script>alert(1)</script>",
])
def test_validate_project_name_invalid_characters(name):
    """
    Summary:
        Project names containing invalid or unsafe characters must be rejected.

    Explanation:
        This test enforces strict character validation to prevent
        injection attacks, malformed data, and unsafe input storage.
    """
    with pytest.raises(HTTPException) as exc:
        validate_project_name(name)
    assert exc.value.status_code == 422
    assert "invalid characters" in exc.value.detail



@pytest.mark.parametrize("input_value, expected", [
    ("Finance", "finance"),
    ("  Healthcare  ", "healthcare"),
    ("Banking & Finance", "banking & finance"),
])
def test_validate_industry_valid(input_value, expected):
    """
    Summary:
        Valid industry names should be normalized and accepted.

    Explanation:
        This test ensures that industry names are:
        - Trimmed of whitespace
        - Lowercased for consistency
        - Allowed to include '&' where appropriate
    """
    result = validate_industry(input_value)
    assert result == expected


@pytest.mark.parametrize("industry, msg", [
    ("", "Industry cannot be empty."),
    (" ", "Industry cannot be empty."),
])
def test_validate_industry_empty(industry, msg):
    """
    Summary:
        Empty or whitespace-only industry values must be rejected.

    Explanation:
        This test validates strict non-empty enforcement to ensure
        meaningful metadata is always provided.
    """
    with pytest.raises(HTTPException) as exc:
        validate_industry(industry)
    assert "Industry cannot be empty." in exc.value.detail




def test_validate_industry_too_short():
    """
    Summary:
        Industry names below minimum length must be rejected.

    Explanation:
        This ensures that industry values remain descriptive
        and avoids meaningless single-character inputs.
    """
    with pytest.raises(HTTPException):
        validate_industry("a")


def test_validate_industry_too_long():
    """
    Summary:
        Industry names exceeding maximum length must be rejected.

    Explanation:
        This test enforces upper bounds to prevent excessive or malformed input.
    """
    with pytest.raises(HTTPException):
        validate_industry("a" * 51)


@pytest.mark.parametrize("industry", [
    "Finance123",
    "Health-care",
    "IT!",
])
def test_validate_industry_invalid_characters(industry):
    """
    Summary:
        Industry names containing invalid characters must fail validation.

    Explanation:
        This test ensures that only letters, spaces, and '&' are allowed,
        maintaining clean, predictable industry metadata.
    """
    with pytest.raises(HTTPException) as exc:
        validate_industry(industry)
    assert "only letters, spaces, and &" in exc.value.detail


def test_validate_description_valid():
    """
    Summary:
        Valid project descriptions should pass validation.

    Explanation:
        This test confirms that sufficiently long and meaningful descriptions
        are accepted without errors.
    """
    validate_description("This is a valid project description.")


@pytest.mark.parametrize("desc", [
    "",
    "short",
    "   too small   ",
])
def test_validate_description_too_short(desc):
    """
    Summary:
        Project descriptions that are too short must be rejected.

    Explanation:
        This enforces a minimum content requirement to ensure
        project descriptions are meaningful.
    """
    with pytest.raises(HTTPException):
        validate_description(desc)


def test_validate_description_too_long():
    """
    Summary:
        Overly long project descriptions must be rejected.

    Explanation:
        This test enforces an upper bound to prevent excessive
        text storage and potential performance issues.
    """
    with pytest.raises(HTTPException):
        validate_description("a" * 1001)


def test_validate_sectors_valid():
    """
    Summary:
        Valid sector lists should be normalized and deduplicated.

    Explanation:
        This test ensures:
        - Case-insensitive matching
        - Whitespace trimming
        - Deduplication
        - Canonical formatting of sector names
    """
    sectors = ["rbi", "SEBI", "  rbi ", "GDPR"]
    result = validate_sectors(sectors)

    assert result == ["RBI", "SEBI", "GDPR"]


def test_validate_sectors_empty():
    """
    Summary:
        Empty sector lists should be allowed.

    Explanation:
        This test confirms that sectors are optional and
        an empty list is a valid input.
    """
    assert validate_sectors([]) == []


def test_validate_sectors_not_list():
    """
    Summary:
        Sector input must be a list.

    Explanation:
        This test ensures type safety by rejecting non-list inputs,
        preventing unexpected runtime errors.
    """
    with pytest.raises(HTTPException):
        validate_sectors("RBI")


def test_validate_sectors_invalid():
    """
    Summary:
        Invalid sector values must be rejected with clear errors.

    Explanation:
        This test verifies that:
        - Only allowed sectors are accepted
        - Invalid values are reported explicitly in the error message
    """
    with pytest.raises(HTTPException) as exc:
        validate_sectors(["RBI", "INVALID", "GDPR"])

    assert exc.value.status_code == 422
    assert "Invalid sectors" in exc.value.detail
    assert "INVALID" in exc.value.detail


@pytest.mark.parametrize("sector", ALLOWED_SECTORS)
def test_validate_sectors_all_allowed(sector):
    """
    Summary:
        Every allowed sector should validate successfully.

    Explanation:
        This test ensures full coverage of the ALLOWED_SECTORS list
        and validates correct normalization behavior for each sector.
    """
    result = validate_sectors([sector.lower()])
    assert result == [sector]
