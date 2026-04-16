# app/tests/models/test_global_response.py
import pytest
from pydantic import ValidationError
from app.models.global_response import GlobalResponse, ErrorItem, Meta


# -------------------------------
# Tests for ErrorItem model
# -------------------------------

def test_error_item_creation():
    """
    Summary:
        Tests the creation of ErrorItem instances, which represent individual
        errors in API responses.

    Explanation:
        ErrorItem has the following fields:
        - message: required, the error message
        - field: optional, the field associated with the error
        - rejected_value: optional, the value that caused the error

        This test verifies that:
        1. Required fields are correctly set.
        2. Optional fields default to None if not provided.
    """
    error = ErrorItem(message="Invalid field", field="name", rejected_value="")
    
    assert error.message == "Invalid field"
    assert error.field == "name"
    assert error.rejected_value == ""

    # Optional fields can be None
    error2 = ErrorItem(message="Required")
    assert error2.field is None
    assert error2.rejected_value is None


# -------------------------------
# Tests for Meta model
# -------------------------------

def test_meta_creation():
    """
    Summary:
        Tests creation of Meta objects, which provide pagination info in API responses.

    Explanation:
        Meta fields:
        - page: current page number (optional)
        - per_page: items per page (optional)
        - total: total number of items (optional)

        This test ensures that optional fields can be provided or omitted.
    """
    meta = Meta(page=1, per_page=10, total=100)
    
    assert meta.page == 1
    assert meta.per_page == 10
    assert meta.total == 100

    # Optional fields can be omitted
    meta2 = Meta()
    assert meta2.page is None
    assert meta2.per_page is None
    assert meta2.total is None


# -------------------------------
# Tests for GlobalResponse model
# -------------------------------

def test_global_response_success_creation():
    """
    Summary:
        Tests creation of a successful GlobalResponse with all fields populated.

    Explanation:
        GlobalResponse represents standard API responses:
        - status: "success" or "error"
        - status_code: HTTP status code
        - message: informational message
        - data: response payload (optional)
        - process_time: optional timing info
        - meta: optional pagination info

        This test ensures that all fields, including nested Meta, are correctly assigned.
    """
    response = GlobalResponse(
        status="success",
        status_code=200,
        message="OK",
        data={"key": "value"},
        process_time="0.123s",
        meta=Meta(page=1, per_page=10, total=50)
    )

    assert response.status == "success"
    assert response.status_code == 200
    assert response.message == "OK"
    assert response.data == {"key": "value"}
    assert response.process_time == "0.123s"
    assert isinstance(response.meta, Meta)
    assert response.meta.page == 1


def test_global_response_error_creation_with_errors():
    """
    Summary:
        Tests creation of an error GlobalResponse with a list of ErrorItem objects.

    Explanation:
        This ensures that GlobalResponse correctly handles API errors
        with multiple error items. Each ErrorItem should have required and optional fields set.
    """
    errors = [
        ErrorItem(message="Invalid", field="name", rejected_value="")
    ]
    response = GlobalResponse(
        status="error",
        status_code=400,
        message="Bad Request",
        errors=errors
    )

    assert response.status == "error"
    assert response.status_code == 400
    assert response.message == "Bad Request"
    assert isinstance(response.errors, list)
    assert response.errors[0].message == "Invalid"
    assert response.errors[0].field == "name"


def test_global_response_invalid_status():
    """
    Summary:
        Ensures that GlobalResponse enforces valid status values.

    Explanation:
        The 'status' field must be either "success" or "error".
        Providing any other value should raise a Pydantic ValidationError.
    """
    with pytest.raises(ValidationError):
        GlobalResponse(
            status="fail",  # invalid
            status_code=400,
            message="Fail"
        )


def test_global_response_optional_fields_can_be_none():
    """
    Summary:
        Tests that optional fields in GlobalResponse can be omitted.

    Explanation:
        Optional fields include:
        - data
        - errors
        - meta
        - process_time

        When omitted, they should default to None without raising errors.
    """
    response = GlobalResponse(
        status="success",
        status_code=200,
        message="OK"
    )
    assert response.data is None
    assert response.errors is None
    assert response.meta is None
    assert response.process_time is None
