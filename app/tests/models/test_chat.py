# app/tests/models/test_chat.py
import pytest
from datetime import datetime, timezone
from pydantic import ValidationError
from app.models.Chat import MessageModel


# -------------------------------
# Tests for MessageModel
# -------------------------------

def test_message_model_valid_data():
    """
    Summary:
        Validates that a MessageModel instance is correctly created
        when all required fields are provided.

    Explanation:
        MessageModel represents a chat message in the system.
        Required fields include message_id, project_id, user_id, and chat_txt.
        This test ensures that when all required fields are provided, the instance
        is created successfully with correct attribute values.
    """
    msg = MessageModel(
        message_id="msg_001",
        project_id="proj_001",
        user_id="user_001",
        chat_txt="Hello world"
    )

    assert msg.message_id == "msg_001"
    assert msg.project_id == "proj_001"
    assert msg.user_id == "user_001"
    assert msg.chat_txt == "Hello world"


def test_message_model_missing_required_fields():
    """
    Summary:
        Ensures that missing required fields in MessageModel raise ValidationError.

    Explanation:
        Pydantic models enforce required fields. If any of the required fields
        (message_id, project_id, user_id, chat_txt) are missing, a ValidationError
        should be raised. This prevents creating invalid message objects.
    """
    # Missing message_id
    with pytest.raises(ValidationError):
        MessageModel(
            project_id="proj_001",
            user_id="user_001",
            chat_txt="Hello"
        )

    # Missing project_id
    with pytest.raises(ValidationError):
        MessageModel(
            message_id="msg_001",
            user_id="user_001",
            chat_txt="Hello"
        )

    # Missing user_id
    with pytest.raises(ValidationError):
        MessageModel(
            message_id="msg_001",
            project_id="proj_001",
            chat_txt="Hello"
        )

    # Missing chat_txt
    with pytest.raises(ValidationError):
        MessageModel(
            message_id="msg_001",
            project_id="proj_001",
            user_id="user_001"
        )


def test_message_model_custom_timestamp():
    """
    Summary:
        Validates that a custom timestamp can be assigned to MessageModel.

    Explanation:
        MessageModel has an optional `timestamp` field, which defaults to the current time
        if not provided. This test ensures that a user-provided timestamp is correctly
        assigned and not overridden by the default value.
    """
    custom_ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    msg = MessageModel(
        message_id="msg_002",
        project_id="proj_002",
        user_id="user_002",
        chat_txt="Custom timestamp",
        timestamp=custom_ts
    )

    assert msg.timestamp == custom_ts
