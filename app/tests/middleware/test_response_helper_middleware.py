import pytest
from fastapi.responses import JSONResponse
import json

from app.middleware.response_helper_middleware import (
    success_response,
    error_response,
    error_response_dict,
)




def test_success_response_default():
    response = success_response(data={"a": 1})

    assert isinstance(response, JSONResponse)
    assert response.status_code == 200

    body = response.body.decode()
    assert '"status":"success"' in body
    assert '"status_code":200' in body
    assert '"data":{"a":1}' in body
    assert '"errors":null' in body


def test_success_response_custom_values():
    response = success_response(
        data=[1, 2, 3],
        message="Created",
        status_code=201,
        process_time="0.123s"
    )

    body = json.loads(response.body)

    assert body["status"] == "success"
    assert body["status_code"] == 201
    assert body["message"] == "Created"
    assert body["data"] == [1, 2, 3]
    assert body["process_time"] == "0.123s"


def test_error_response_default():
    response = error_response(message="Bad Request", status_code=400)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400

    body = json.loads(response.body)
    assert body["status"] == "error"
    assert body["status_code"] == 400
    assert body["message"] == "Bad Request"
    assert body["data"] is None
    assert body["errors"] is None


def test_error_response_with_errors_list():
    errors = [
        {
            "message": "Invalid field",
            "field": "name",
            "rejected_value": ""
        }
    ]

    response = error_response(
        message="Validation failed",
        status_code=422,
        errors=errors
    )

    body = json.loads(response.body)

    assert body["status"] == "error"
    assert body["status_code"] == 422
    assert body["errors"][0]["message"] == "Invalid field"
    assert body["errors"][0]["field"] == "name"


def test_error_response_dict_returns_plain_dict():
    result = error_response_dict(
        message="Unauthorized",
        status_code=401
    )

    assert isinstance(result, dict)
    assert result["status"] == "error"
    assert result["status_code"] == 401
    assert result["message"] == "Unauthorized"
    assert result["data"] is None
    assert result["errors"] is None


def test_error_response_dict_with_errors():
    errors = [
        {
            "message": "Missing token",
            "field": "Authorization",
            "rejected_value": None
        }
    ]

    result = error_response_dict(
        message="Auth failed",
        status_code=401,
        errors=errors
    )

    assert result["errors"][0]["message"] == "Missing token"
    assert result["errors"][0]["field"] == "Authorization"


def test_json_serializable():
    response = success_response(data={"ok": True})
    json.loads(response.body)  # Should not raise
