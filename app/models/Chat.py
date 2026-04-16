from pydantic import BaseModel
from datetime import datetime, timezone

class MessageModel(BaseModel):
    message_id: str              # Unique ID for the message
    project_id: str              # Reference to the project
    user_id: str                 # Reference to the user who sent the message
    chat_txt: str                # The chat text content
    timestamp: datetime = datetime.now(timezone.utc)