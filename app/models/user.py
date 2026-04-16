from pydantic import BaseModel, Field, EmailStr
from typing import Optional
class User(BaseModel):
    # user_id: str = Field(default_factory=str)  
    name: str                                  
    email: EmailStr                             
    password: str                               
    role: str = Field(default="user")
    admin_id: Optional[str] = Field(default=None)