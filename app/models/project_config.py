from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

def ist_now():
    return datetime.now(IST)

class ProjectConfig(BaseModel):
    """
    MongoDB Model for the 'project_configs' collection.
    """
    #  CHANGE: Make this Optional. 
    # It allows the body to be empty of ID, but the Router will fill it in.
    project_id: Optional[str] = Field(default=None, description="Reference to the parent Project ID")
    
    # --- AI Model Selection ---
    router_model: str = Field(default="qwen2.5:0.5b", description="Model used for routing")
    simple_model: str = Field(default="qwen2.5:1.5b-instruct", description="Model for simple queries")
    complex_model: str = Field(default="qwen2.5:14b", description="Model for complex queries")
    
    # --- Search Strategy ---
    search_strategy: str = Field(default="hybrid", description="hybrid, vector, or keyword")
    retrieval_depth: int = Field(default=5, description="Top K chunks to retrieve")
    enable_reranking: bool = Field(default=True, description="Use BGE Reranker")
    
    # --- Processing ---
    ocr_engine: str = Field(default="easyocr", description="none, easyocr, or paddleocr")
    chat_history_limit: int = Field(default=5, ge=1, le=50, description="Number of past messages to retain in context")

    hallucination_threshold: float = Field(default=100.0, ge=0.0, le=100.0, description="Score above which response is NOT cached")
    
    # --- Metadata (IST) ---
    created_at: datetime = Field(default_factory=ist_now)
    updated_at: datetime = Field(default_factory=ist_now)

    class Config:
        json_schema_extra = {
            "example": {
                "router_model": "qwen2.5:0.5b",
                "search_strategy": "hybrid",
                "retrieval_depth": 5,
                "enable_reranking": True
            }
        }

class ProjectConfigPatch(BaseModel):
    """
    Validation schema for updating AI settings. 
    All fields are Optional so we can update just one at a time.
    """
    router_model: Optional[str] = None
    simple_model: Optional[str] = None
    complex_model: Optional[str] = None
    search_strategy: Optional[str] = None
    retrieval_depth: Optional[int] = None
    enable_reranking: Optional[bool] = None
    ocr_engine: Optional[str] = None
    chat_history_limit: Optional[int] = Field(default=None, ge=1, le=50)
    hallucination_threshold: Optional[float] = Field(default=None, ge=0.0, le=100.0)