# app/schemas.py

from pydantic import BaseModel, Field, EmailStr, validator, field_validator
from typing import List, Optional, Any, TypeVar, Generic, Dict
from datetime import datetime

# --- NEW MODELS FOR STANDARDIZED RESPONSE ---

# Define a generic type for the 'data' field
T = TypeVar('T')

# --- MODIFIED ---
class ChatHistoryEntry(BaseModel):
    created_at: datetime
    
    # --- NEW ---
    # This field tells the frontend what to render:
    # "text" = a user_query/llm_answer pair
    # "file" = a file upload component
    message_type: str = Field(default="text", description="Type of message ('text' or 'file')")

    # --- Fields for "text" messages (now optional) ---
    user_query: Optional[str] = Field(None, description="The user's text query")
    llm_answer: Optional[str] = Field(None, description="The AI's text response")
    source_documents: Optional[List["SourceDocument"]] = Field(
        default=None,
        description="Retrieved source chunks associated with the AI response"
    )
    
    # --- NEW Fields for "file" messages ---
    file_id: Optional[str] = Field(None, description="The unique ID of an uploaded file")
    file_name: Optional[str] = Field(None, description="The original name of an uploaded file")
    sector: Optional[str]


class NewChatResponseData(BaseModel):
    chat_id: str = Field(..., description="The newly generated unique ID for the chat session.")

# This model represents a single detailed error
class ErrorDetail(BaseModel):
    message: str
    field: Optional[str] = None
    rejected_value: Optional[Any] = None

# This is our new, standardized response model that will be used for all API outputs
class StandardResponse(BaseModel, Generic[T]):
    status: str = "success"
    status_code: int = 200
    message: Optional[str] = None
    data: Optional[T] = None
    errors: Optional[List[ErrorDetail]] = None
    meta: Optional[Dict[str, Any]] = None
    process_time: Optional[str] = None

class QueryUnderstanding(BaseModel):
    authority: Optional[str] = Field(None, description="Regulatory Authority like RBI, SEBI, NPCI")
    regulation: Optional[str] = Field(None, description="Specific regulation like KYC, AML, GDPR")
    document_type: Optional[str] = Field(None, description="Type of document like circular, guideline, act")
    task_type: Optional[str] = Field(None, description="Intent: define, compare, explain, list")
    entities: List[str] = Field(default_factory=list, description="Mentioned entities like NBFC, bank")
    jurisdiction: Optional[str] = Field(None, description="Region: India, EU, Global")
    confidence: float = Field(0.0, description="Confidence score 0-1")


# --- DATA-SPECIFIC MODELS ---
# These define the structure of the 'data' field for different successful responses.

# --- MODIFIED FOR MULTI-SOURCE RETRIEVAL ---
class QueryRequest(BaseModel):
    """Single-sector query request for organization HR RAG."""
    query: str = Field(..., min_length=1, description="The user's search query")
    project_id: str = Field(..., description="The project ID")

    style: Optional[str] = Field(
        default="Detailed",
        description="Response style: 'Simple', 'Formal', 'Detailed', 'Experimental'"
    )

    excluded_files: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific filenames to exclude from search results"
    )

    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Total number of chunks to retrieve")
    dense_weight: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    sparse_weight: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    use_reranking: Optional[bool] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is our leave policy for probation employees?",
                "project_id": "Project_123",
                "top_k": 5
            }
        }

# --- NEW: Enhanced source document with retrieval metadata ---
class SourceDocument(BaseModel):
    """
    Represents a single retrieved document chunk with metadata.
    """
    page_content: str = Field(..., description="The text content of the chunk")
    
    metadata: Dict[str, Any] = Field(
        ..., 
        description="Metadata including source, file_name, chunk_id, page_number, etc."
    )
    
    # --- NEW: Retrieval scores ---
    relevance_score: Optional[float] = Field(
        None, 
        description="Final relevance score (after fusion/reranking)"
    )
    
    dense_score: Optional[float] = Field(
        None, 
        description="Dense embedding similarity score"
    )
    
    sparse_score: Optional[float] = Field(
        None, 
        description="BM25 sparse retrieval score"
    )
    
    rerank_score: Optional[float] = Field(
        None, 
        description="Cross-encoder reranking score (if reranking enabled)"
    )


# --- NEW: Detailed retrieval statistics ---
class RetrievalStats(BaseModel):
    """
    Statistics about the retrieval process.
    """
    total_chunks_searched: int = Field(..., description="Total number of chunks in the search space")
    chunks_retrieved: int = Field(..., description="Number of chunks actually retrieved")
    sources_queried: List[str] = Field(..., description="List of sources that were searched")
    excluded_files_count: int = Field(default=0, description="Number of files excluded via blacklist")
    retrieval_method: str = Field(..., description="Method used: 'hybrid', 'dense_only', 'sparse_only'")
    reranking_applied: bool = Field(..., description="Whether reranking was applied")
    processing_time_ms: Optional[float] = Field(None, description="Time taken for retrieval in milliseconds")


class SectorComparison(BaseModel):
    """Results from a single sector for comparative analysis."""
    sector: str = Field(..., description="Sector name (e.g., 'RBI', 'GDPR')")
    chunks_found: int = Field(..., description="Number of chunks found in this sector")
    top_chunks: List[SourceDocument] = Field(..., description="Top chunks from this sector")
    sector_summary: Optional[str] = Field(None, description="Brief summary of what this sector says")



# --- MODIFIED: Enhanced query response with stats ---
class QueryResponseData(BaseModel):
    """
    Enhanced query response with comparative analysis support.
    """
    result: str = Field(..., description="The LLM's generated answer")
    
    source_documents: List[SourceDocument] = Field(
        ..., 
        description="List of retrieved source chunks used to generate the answer"
    )
    
    # NEW: Comparative analysis data
    comparative_analysis: Optional[List[SectorComparison]] = Field(
        None,
        description="Breakdown by sector for comparative queries"
    )
    
    is_comparative: bool = Field(
        default=False,
        description="Whether this was a comparative query across multiple sectors"
    )
    
    retrieval_stats: Optional[RetrievalStats] = Field(
        None, 
        description="Detailed statistics about the retrieval process"
    )
    
    sources_used: Optional[Dict[str, int]] = Field(
        None,
        description="Breakdown of how many chunks came from each source"
    )


# --- MODIFIED FOR BULK UPLOAD ---
# This is the 'data' for a SINGLE successful file in the File Router's upload response
class FileUploadResponseData(BaseModel):
    file_id: str
    project_id: str
    filename: str
    file_url: str
    sector: str  # <-- ADDED as requested
    category: str = "General" 
    compliance_type: str = "General"


# --- CLEANUP: Fixed consistency ---
class DeleteRequest(BaseModel):
    filename: str
    project_id: str # Correct: The project the file is in

# --- CLEANUP: Fixed consistency ---
# We find the file by filename and project_id,
# then update its sector tag.
class UpdateRequest(BaseModel):
    filename: str
    project_id: str # Correct: The project the file is in
    new_sector: str   # Correct: The new sector tag to apply
    
# --- MODIFIED ---
# This is now the response schema for the chat's /upload endpoint.
# It includes the file_id for the frontend.
class RagUploadResponseData(BaseModel):
    file_id: str = Field(..., description="The unique ID of the stored file.")
    file_name: str = Field(..., description="The original name of the uploaded file.")
    sector: str = Field(..., description="The sector the file was tagged with.")
    message: str = Field(default="File processed and indexed successfully.")


# --- NEW SCHEMAS FOR BULK UPLOAD RESPONSE ---

class FileUploadErrorData(BaseModel):
    """Holds information about a single failed file upload."""
    filename: str = Field(..., description="The name of the file that failed.")
    error: str = Field(..., description="The reason for the failure.")

class BulkUploadResponseData(BaseModel):
    """The 'data' field for a bulk upload response. Separates successes and failures."""
    successful_uploads: List[FileUploadResponseData] = []
    failed_uploads: List[FileUploadErrorData] = []


# ==================== NEW: RETRIEVAL LAYER SCHEMAS ====================

# --- Layer 4: Retrieval Internal Schemas ---

class SearchFilter(BaseModel):
    """
    Unified search filter for Qdrant queries.
    Treats project_id and sectors uniformly as 'sources'.
    """
    sources: List[str] = Field(
        ..., 
        description="List of sources to search (project_id + sectors)"
    )
    
    excluded_files: Optional[List[str]] = Field(
        default=None,
        description="List of filenames to exclude from results"
    )
    query_understanding: Optional[QueryUnderstanding] = Field(default=None, description="Structured intent extracted from query")
    
    def to_qdrant_filter(self) -> Dict[str, Any]:
        """
        Convert to Qdrant filter format.
        
        Returns a filter dict like:
        {
            "must": [
                {"key": "source", "match": {"any": ["Project_123", "RBI", "HIPAA"]}}
            ],
            "must_not": [
                {"key": "file_name", "match": {"any": ["outdated.pdf", "deprecated.pdf"]}}
            ]
        }
        """
        qdrant_filter = {
            "must": [
                {
                    "key": "source",
                    "match": {"any": self.sources}
                }
            ]
        }
        
        if self.excluded_files:
            qdrant_filter["must_not"] = [
                {
                    "key": "file_name",
                    "match": {"any": self.excluded_files}
                }
            ]
        
        return qdrant_filter


class RetrievedChunk(BaseModel):
    """
    Internal representation of a retrieved chunk before final formatting.
    """
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_score: float


class RetrievalResult(BaseModel):
    """
    Internal result from the retrieval pipeline before LLM generation.
    """
    chunks: List[RetrievedChunk]
    stats: RetrievalStats
    filter_applied: SearchFilter

class GlobalFileUploadRequest(BaseModel):
    """Request for uploading global sector files."""
    sector: str = Field(..., description="Sector name (e.g., 'RBI', 'GDPR', 'HIPAA')")
    description: Optional[str] = Field(None, description="File description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sector": "RBI",
                "description": "RBI Master Circular on KYC 2024"
            }
        }

class GlobalFileUploadResponse(BaseModel):
    """Response for global file upload."""
    file_id: str
    sector: str
    filename: str
    document_type: str
    chunks_indexed: int
    message: str = "Global file uploaded successfully"

class GPUStat(BaseModel):
    id: int
    name: str  # e.g., "Tesla T4" or "RTX 4090"
    memory_used_mb: int
    memory_total_mb: int
    gpu_util_percent: int
    temp_c: int

class ModelStat(BaseModel):
    count: int
    last_latency_ms: float
    avg_latency_ms: float
    status: str  # "idle", "processing", "error"
    vram_mb: Optional[float] = 0.0

class ModelServerStats(BaseModel):
    server_status: str
    gpus: List[GPUStat]
    models: Dict[str, ModelStat]

class FullSystemMonitor(BaseModel):
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_connections: int
    model_server: Optional[ModelServerStats] = None


