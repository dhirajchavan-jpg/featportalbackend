
import os
from typing import Dict, List, ClassVar, Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    SERVER_BASE_URL: str = "http://localhost:8000"
    # MODEL_SERVER_URLS="http://localhost:8074,http://localhost:8075,http://localhost:8076"
    MODEL_SERVER_URLS="http://localhost:8074,http://localhost:8075"
    BASE_DIR: ClassVar[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROMPTS_DIR: ClassVar[str] = os.path.join(BASE_DIR, "prompts")

    DEBUG: bool = False

    # =========================================================
    # NEW: REDIS SETTINGS (For Async Queue)
    # =========================================================
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6382
    REDIS_DB: int = 0
    REDIS_QUEUE_NAME: str = "rag_query_queue"
    REDIS_RESULT_TTL: int = 3600  # Results expire after 1 hour

    # Qdrant Settings
    QDRANT_HOST: str = "http://localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: str = f"{QDRANT_HOST}:{QDRANT_PORT}"
    QDRANT_COLLECTION_NAME: str = "hr_rag_collection"
    
    # Qdrant Configuration for Hybrid Search
    QDRANT_DENSE_VECTOR_NAME: str = "dense"  # BGE-M3 embeddings
    QDRANT_SPARSE_VECTOR_NAME: str = "sparse"  # BM25 scores

    # Ollama LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    BACKEND_PUBLIC_URL: str = "http://localhost:8072"
    
    # Multi-Model Setup
    LLM_MODEL_SIMPLE: str = "qwen2.5:1.5b-instruct"
    LLM_MODEL_COMPLEX: str = "qwen2.5:14b"
    LLM_MODEL: str = LLM_MODEL_SIMPLE
    
    # Router Model
    ROUTER_MODEL: str = "qwen2.5:0.5b"
    
    # Multi-Embedding Setup
    EMBEDDING_MODEL: str = "nomic-embed-text"
    BGE_M3_MODEL: str = "BAAI/bge-m3"
    BGE_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    # MongoDB Cache Settings
    MONGO_URI: str = "mongodb://localhost:27017/"
    MONGO_DB_NAME: str = "feathr_hr_portal_hr"
    MONGO_COLLECTION_NAME: str = "query_cache"

    # File Settings
    UPLOAD_DIR: str = "uploads"
    PROCESSED_DIR: str = "data/processed"
    JSON_OUTPUT_DIR: str = "data/json_outputs"

    # JWT Authentication Settings
    # Using Optional so it doesn't crash if env var is missing during local dev
    JWT_SECRET_KEY: Optional[str] = os.environ.get("JWT_SECRET_KEY")
    JWT_ALGORITHM: str = "HS256"
    access_token_expire_minutes: int = 1440
    
    # Layer 1 - Document Processing Settings
    LAYOUT_MODEL: str = "cmarkea/detr-layout-detection"
    LAYOUT_CONFIDENCE_THRESHOLD: float = 0.5
    
    OCR_LANGUAGES: list = ["en"]
    OCR_USE_GPU: bool = False
    
    TABLE_PARSER: str = "pdfplumber"
    DETECT_FORMULAS: bool = True
    
    # Layer 2 - Chunking Settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 60
    ENABLE_SECTION_CHUNKING: bool = True
    ENABLE_SEMANTIC_CHUNKING: bool = True
    
    # Layer 3 - Indexing Settings
    USE_HYBRID_EMBEDDING: bool = True
    BGE_M3_BATCH_SIZE: int = 32
    
    # Layer 4 - Retrieval Settings
    ENABLE_QUERY_EXPANSION: bool = True
    ENABLE_METADATA_EXTRACTION: bool = True
    
    BM25_TOP_K: int = 50
    DENSE_TOP_K: int = 50
    RRF_TOP_K: int = 50
    RERANK_TOP_K: int = 10
    RRF_K: int = 60
    
    MAX_CONTEXT_TOKENS: int = 4000
    ENABLE_DEDUPLICATION: bool = True

    ALLOWED_MIME_TYPES: Dict[str, str] = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain"
    }
    # Assigned default value (10MB) to prevent errors
    MAX_FILE_SIZE: int = 10 * 1024 * 1024

    # Query Routing Thresholds
    SIMPLE_QUERY_MAX_TOKENS: int = 20
    COMPLEX_QUERY_KEYWORDS: list = [
        "compare", "analyze", "explain in detail", "comprehensive",
        "multiple", "complex", "intricate", "elaborate"
    ]

    # Rate Limiting Settings
    RATE_LIMIT_UPLOADS_PER_HOUR: int = 5
    RATE_LIMIT_CHAT_PER_MINUTE: int = 15

    # =========================================================
    # MODIFIED: MODEL SERVER URLS (For Multi-GPU Load Balancing)
    # =========================================================
    # Accepts a single URL or a comma-separated list
    # Example in .env: MODEL_SERVER_URLS="http://localhost:8074,http://localhost:8075"
    MODEL_SERVER_URLS: str = Field(
        default="http://localhost:8074,http://localhost:8075", 
        env="MODEL_SERVER_URLS"
    )

    # Change the fallback to 8072 to match your actual backend port
    BACKEND_PUBLIC_URL: str = "http://localhost:8072"

    @property
    def model_server_urls_list(self) -> List[str]:
        """Parses the comma-separated string into a list of URLs."""
        return [url.strip() for url in self.MODEL_SERVER_URLS.split(",") if url.strip()]

    # Other External Services
    phoenix_collector_endpoint: str = Field(..., env="PHOENIX_COLLECTOR_ENDPOINT")
    phoenix_project_name: str = "compliance-rag-api2.0"

    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    
    # Evaluation Settings
    EVALUATION_SAMPLE_RATE: float = Field(default=0.1, env="EVALUATION_SAMPLE_RATE")
    EVALUATION_INTERVAL_DAYS: int = Field(default=7, env="EVALUATION_INTERVAL_DAYS")

    class Config:
        env_file = ".env"
        extra = "ignore" 

settings = Settings()

# Create necessary directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
os.makedirs(settings.JSON_OUTPUT_DIR, exist_ok=True)


