import os
import logging
import httpx
import json
import time 
import asyncio 
import random 
import pynvml # <--- REQUIRED: To check GPU memory
from typing import List, Dict, Any, Optional
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.outputs import LLMResult
from qdrant_client import QdrantClient
from app.config import settings

# Import the dedicated OCR client
from app.services.document_processing.ocr_engine import get_ocr_engine

logger = logging.getLogger(__name__)

# --- CONFIGURATION: ENABLE PARALLELISM ---
# Allow up to 10 concurrent requests (Ollama handles internal queuing)
MAX_CONCURRENT_LLM_REQUESTS = 10 
_llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_REQUESTS)

# --- GLOBAL VARIABLES ---
_llm_simple = None      
_llm_complex = None     
_llm_router = None      
_llm = None             
_embeddings = None      
_qdrant_client = None
_dense_embedder = None  
_sparse_embedder_cache = {} 
_reranker = None        
_ocr_engine = None
_layout_processor = None 

# --- NEW: OLLAMA MULTI-GPU MAPPING ---
# Maps GPU Index to the Port where that specific Ollama instance is running.
OLLAMA_GPU_MAP = {
    0: "http://localhost:11434",  # GPU 0 (Light Models + Model Server)
    1: "http://localhost:11435",  # GPU 1 (Heavy 14B Model)
    2: "http://localhost:11436"   # GPU 2 (Light Models + Model Server)
}

# Estimated VRAM usage (MB) for your models
MODEL_VRAM_REQ = {
    "qwen2.5:7b": 5500,
    "qwen2.5:14b": 11000,
    "qwen2.5:0.5b": 800,
    "llama3": 5500,
}

# =========================================================================
# MONITORING STATE
# =========================================================================

llm_stats = {
    "llm": {
        "count": 0, "last_latency_ms": 0, "avg_latency_ms": 0, 
        "status": "idle", "vram_mb": 0, "queue_depth": 0 
    }
}

def get_llm_stats():
    return llm_stats

# --- SHARED ASYNC CLIENT ---
_global_async_client = None

def get_shared_async_client():
    global _global_async_client
    if _global_async_client is None or _global_async_client.is_closed:
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        _global_async_client = httpx.AsyncClient(limits=limits, timeout=60.0)
    return _global_async_client

def estimate_vram():
    """Estimates VRAM usage based on the configured simple model."""
    model_name = settings.LLM_MODEL_SIMPLE.lower()
    estimated_mb = 0
    for key, mb in MODEL_VRAM_REQ.items():
        if key in model_name:
            estimated_mb = mb
            break
    if estimated_mb == 0 and model_name:
        estimated_mb = 4000 
    llm_stats["llm"]["vram_mb"] = estimated_mb
    logger.info(f"[Monitor] Estimated LLM VRAM for '{model_name}': {estimated_mb} MB")

# =========================================================================
# NEW: DYNAMIC GPU SELECTION LOGIC (ROLE BASED)
# =========================================================================

def get_best_ollama_url(model_name: str) -> str:
    """
    Scans GPUs based on CONFIG ROLES (Robust):
    - If model == COMPLEX Setting -> GPU 1 Only
    - If model == SIMPLE or ROUTER Setting -> GPU 0 or GPU 2
    """
    
    # 1. Determine Allowed GPUs based on ROLES
    if model_name == settings.LLM_MODEL_COMPLEX:
        allowed_gpus = [1] # Dedicated Heavy GPU
        logger.info(f"[GPU Select] '{model_name}' matches COMPLEX role. Restricted to GPU 1.")
        
    elif model_name == settings.LLM_MODEL_SIMPLE or model_name == settings.ROUTER_MODEL:
        allowed_gpus = [0, 2] # Shared Light GPUs
        logger.info(f"[GPU Select] '{model_name}' matches LIGHT role. Scanning GPU 0 & 2.")
        
    else:
        allowed_gpus = [0, 1, 2] # Fallback
        logger.info(f"[GPU Select] '{model_name}' is unclassified. Scanning all GPUs.")

    # 2. Determine VRAM Requirement
    required_mb = 4000 # Default safe fallback
    for key, mb in MODEL_VRAM_REQ.items():
        if key in model_name.lower():
            required_mb = mb
            break
            
    try:
        # 3. Init NVML
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        best_gpu_id = -1
        max_free_memory = -1

        # 4. Scan ONLY Allowed GPUs
        for i in range(device_count):
            # Skip if this GPU is not in our allowed list for this model
            if i not in allowed_gpus:
                continue
                
            # Skip if we don't have a mapped port
            if i not in OLLAMA_GPU_MAP:
                continue

            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mb = mem_info.free / 1024**2
            
            logger.info(f"    > GPU {i} (Port {OLLAMA_GPU_MAP[i]}): {free_mb:.0f} MB Free")

            # Logic: Pick allowed GPU with MOST free memory
            if free_mb > required_mb and free_mb > max_free_memory:
                max_free_memory = free_mb
                best_gpu_id = i

        # 5. Return Winner
        if best_gpu_id != -1:
            target_url = OLLAMA_GPU_MAP[best_gpu_id]
            logger.info(f"[GPU Select]  Selected GPU {best_gpu_id} ({target_url}) for {model_name}")
            return target_url
        
        # 6. Fallback (If preferred GPUs are full/unavailable)
        fallback_id = allowed_gpus[0] if allowed_gpus else 0
        fallback_url = OLLAMA_GPU_MAP.get(fallback_id, settings.OLLAMA_BASE_URL)
        logger.warning(f"[GPU Select]  Preferred GPUs full! Defaulting to {fallback_url}")
        return fallback_url

    except Exception as e:
        logger.error(f"[GPU Select] NVML Error: {e}. Defaulting to standard URL.")
        return settings.OLLAMA_BASE_URL 

# =========================================================================
# OLLAMA WRAPPER (UNCHANGED)
# =========================================================================

class MonitoredOllama(Ollama):
    """Wraps OllamaLLM to catch ALL usage patterns."""
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> LLMResult:
        llm_stats["llm"]["status"] = "processing"
        start_time = time.time()
        try:
            result = super()._generate(prompts, stop, run_manager, **kwargs)
            self._update_stats(time.time() - start_time)
            return result
        finally:
            llm_stats["llm"]["status"] = "idle"

    async def _agenerate(self, prompts, stop, run_manager, **kwargs) -> LLMResult:
        llm_stats["llm"]["queue_depth"] += 1
        async with _llm_semaphore:
            llm_stats["llm"]["queue_depth"] -= 1
            llm_stats["llm"]["status"] = "processing"
            start_time = time.time()
            try:
                result = await asyncio.to_thread(super()._generate, prompts, stop, run_manager, **kwargs)
                return result
            finally:
                self._update_stats(time.time() - start_time)
                llm_stats["llm"]["status"] = "idle"

    def _update_stats(self, duration_sec):
        ms = round(duration_sec * 1000, 2)
        s = llm_stats["llm"]
        s["count"] += 1
        s["last_latency_ms"] = ms
        if s["avg_latency_ms"] == 0: s["avg_latency_ms"] = ms
        else: s["avg_latency_ms"] = round((s["avg_latency_ms"] * 0.9) + (ms * 0.1), 2)

# =========================================================================
# REMOTE CLASSES
# =========================================================================

class RemoteDenseEmbedder:
    """Proxy for /embed endpoint with Load Balancing"""
    def __init__(self, base_urls: List[str]):
        self.endpoints = [f"{url.rstrip('/')}/embed" for url in base_urls]
        logger.info(f"[RemoteEmbedder] Initialized with {len(self.endpoints)} nodes: {self.endpoints}")
    
    def _get_endpoint(self):
        return random.choice(self.endpoints)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        endpoint = self._get_endpoint()
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(endpoint, json={"texts": texts})
                response.raise_for_status()
                return response.json()["embeddings"]
        except Exception as e:
            logger.error(f"[RemoteEmbedder] Error calling {endpoint}: {e}")
            return [[0.0] * 1024 for _ in texts]

    def embed_query(self, query: str) -> List[float]:
        res = self.embed_documents([query])
        return res[0] if res else [0.0] * 1024
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        endpoint = self._get_endpoint()
        try:
            client = get_shared_async_client()
            response = await client.post(endpoint, json={"texts": texts}, timeout=60.0)
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            logger.error(f"[RemoteEmbedder Async] Error calling {endpoint}: {e}")
            return [[0.0] * 1024 for _ in texts]

    async def aembed_query(self, query: str) -> List[float]:
        res = await self.aembed_documents([query])
        return res[0] if res else [0.0] * 1024


class RemoteReranker:
    """Proxy for /rerank endpoint with Load Balancing"""
    def __init__(self, base_urls: List[str]):
        self.endpoints = [f"{url.rstrip('/')}/rerank" for url in base_urls]
        self.model = True # Fake attribute
        logger.info(f"[RemoteReranker] Initialized with {len(self.endpoints)} nodes.")
    
    def load_model(self): pass

    def _get_endpoint(self):
        return random.choice(self.endpoints)

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        endpoint = self._get_endpoint()
        try:
            payload = {"query": query, "documents": documents, "top_k": top_k}
            with httpx.Client(timeout=90.0) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()["results"]
        except Exception as e:
            logger.error(f"[RemoteReranker] Failed at {endpoint}: {e}")
            return documents[:top_k]
        
    async def arerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        endpoint = self._get_endpoint()
        try:
            payload = {"query": query, "documents": documents, "top_k": top_k}
            client = get_shared_async_client()
            response = await client.post(endpoint, json=payload, timeout=90.0)
            response.raise_for_status()
            return response.json()["results"]
        except Exception as e:
            logger.error(f"[RemoteReranker Async] Failed at {endpoint}: {e}")
            return documents[:top_k]


class RemoteLayoutProcessor:
    """Proxy for /layout endpoint with Load Balancing"""
    def __init__(self, base_urls: List[str]):
        self.endpoints = [f"{url.rstrip('/')}/layout" for url in base_urls]
        logger.info(f"[RemoteLayout] Initialized with {len(self.endpoints)} nodes.")
        
    def _get_endpoint(self):
        return random.choice(self.endpoints)

    def detect(self, image_bytes: bytes) -> List[Dict]:
        endpoint = self._get_endpoint()
        try:
            files = {'file': ('page.png', image_bytes, 'image/png')}
            with httpx.Client(timeout=60.0) as client:
                response = client.post(endpoint, files=files)
                response.raise_for_status()
                data = response.json()
                if "pages" in data and len(data["pages"]) > 0:
                    return data["pages"][0].get("elements", [])
                return []
        except Exception as e:
            logger.error(f"[RemoteLayout] Failed at {endpoint}: {e}")
            return []

# =========================================================================
# UPDATED LOAD MODELS FUNCTION
# =========================================================================

def load_models():
    """
    Initializes Clients to connect to the Remote Model Servers.
    """
    global _llm_simple, _llm_complex, _llm_router, _llm, _embeddings, _qdrant_client
    global _dense_embedder, _sparse_embedder_cache, _reranker, _ocr_engine, _layout_processor
    
    logger.info("="*70)
    logger.info(f"STARTING LLM PROVIDER (MULTI-GPU OLLAMA MODE)")
    
    server_urls = settings.model_server_urls_list
    logger.info(f"[LoadBalancer] Targeting Model Server Nodes: {server_urls}")
    
    # 1. Load Ollama Models with DYNAMIC GPU SELECTION
    try:
        common_args = {"temperature": 0, "keep_alive": "5m"}
        ephemeral_args = {"temperature": 0, "keep_alive": "0"}

        # A. ROUTER (Light) -> Scans GPU 0 & 2
        router_url = get_best_ollama_url(settings.ROUTER_MODEL)
        _llm_router = MonitoredOllama(
            base_url=router_url,
            model=settings.ROUTER_MODEL, 
            **ephemeral_args
        )
        logger.info(f"[LLM] Router ({settings.ROUTER_MODEL}) assigned to: {router_url}")
        
        # B. SIMPLE MODEL (Light) -> Scans GPU 0 & 2
        simple_url = get_best_ollama_url(settings.LLM_MODEL_SIMPLE)
        _llm_simple = MonitoredOllama(
            base_url=simple_url,
            model=settings.LLM_MODEL_SIMPLE, 
            **ephemeral_args
        )
        logger.info(f"[LLM] Simple ({settings.LLM_MODEL_SIMPLE}) assigned to: {simple_url}")
        
        # C. COMPLEX MODEL (Heavy) -> Restricted to GPU 1
        complex_url = get_best_ollama_url(settings.LLM_MODEL_COMPLEX)
        _llm_complex = MonitoredOllama(
            base_url=complex_url,
            model=settings.LLM_MODEL_COMPLEX, 
            **common_args
        )
        logger.info(f"[LLM] Complex ({settings.LLM_MODEL_COMPLEX}) assigned to: {complex_url}")
        
        _llm = _llm_simple
        
        estimate_vram()
        
    except Exception as e:
        logger.error(f"[LLM] Failed to init Ollama clients: {e}")

    # 2. Qdrant Client
    try:
        _qdrant_client = QdrantClient(url=settings.QDRANT_URL)
        logger.info("[Qdrant] Client initialized.")
    except Exception as e:
        logger.error(f"[Qdrant] Failed to initialize: {e}")

    # 3. Remote Services
    _dense_embedder = RemoteDenseEmbedder(server_urls)
    _reranker = RemoteReranker(server_urls)
    _layout_processor = RemoteLayoutProcessor(server_urls)
    
    try:
        _ocr_engine = get_ocr_engine()
        logger.info(f"[OCR] Remote OCR Engine initialized.")
    except Exception as e:
        logger.error(f"[OCR] Failed to initialize OCR Engine: {e}")

    logger.info("="*70)
    logger.info("CLIENT PROVIDER READY")
    logger.info("="*70)

# --- UPDATED GET_LLM (Dynamic Switching) ---

def get_llm(model_name: Optional[str] = None):
    """Get an LLM instance, creating one dynamically on the best GPU if needed."""
    global _llm, _llm_simple, _llm_complex, _llm_router

    if not model_name: return _llm

    if _llm_simple and getattr(_llm_simple, "model", "") == model_name: return _llm_simple
    if _llm_complex and getattr(_llm_complex, "model", "") == model_name: return _llm_complex
    if _llm_router and getattr(_llm_router, "model", "") == model_name: return _llm_router

    logger.info(f"[get_llm] Model '{model_name}' not loaded. finding best GPU...")
    
    try:
        # DYNAMICALLY FIND BEST GPU FOR THIS NEW MODEL (Respecting Routing Rules)
        best_url = get_best_ollama_url(model_name)
        
        new_llm = MonitoredOllama(
            base_url=best_url, 
            model=model_name, 
            temperature=0,
            keep_alive="5m"
        )
        return new_llm
    except Exception as e:
        logger.error(f"[get_llm] Failed to create dynamic LLM for {model_name}: {e}")
        return _llm

# --- GETTERS ---
def get_simple_llm(): return _llm_simple
def get_complex_llm(): return _llm_complex
def get_router_llm(): return _llm_router
def get_embeddings(): return _embeddings
def get_qdrant_client(): return _qdrant_client
def get_dense_embedder(): return _dense_embedder
def get_bge_reranker(): return _reranker
def get_ocr_engine_instance(): return _ocr_engine
def get_layout_engine(): return _layout_processor

def get_sparse_embedder(index_id: str):
    global _sparse_embedder_cache
    if index_id in _sparse_embedder_cache:
        return _sparse_embedder_cache[index_id]
    try:
        from app.services.embedding.sparse_embedder import SparseEmbedder
        embedder = SparseEmbedder() 
        filename = f"bm25_{index_id}.pkl"
        index_path = os.path.join(settings.PROCESSED_DIR, filename)
        if os.path.exists(index_path):
            embedder.load_index(index_path)
            if embedder.bm25_index:
                _sparse_embedder_cache[index_id] = embedder
                return embedder
        return None
    except Exception:
        return None

def cleanup_vram():
    urls = list(OLLAMA_GPU_MAP.values())
    models_to_unload = list(set([settings.LLM_MODEL_SIMPLE, settings.LLM_MODEL_COMPLEX, settings.ROUTER_MODEL]))
    
    for base_url in urls:
        api_url = f"{base_url}/api/generate"
        for model_name in models_to_unload:
            try:
                payload = {"model": model_name, "keep_alive": 0}
                with httpx.Client(timeout=2.0) as client:
                    client.post(api_url, json=payload)
            except Exception as e:
                pass 
    logger.info("[VRAM] Cleanup signal sent to all Ollama instances.")

def check_models_health():
    health = {"status": "ok", "mode": "client", "endpoints": {}}
    
    # Check all Ollama instances
    for gpu_id, url in OLLAMA_GPU_MAP.items():
        try:
            with httpx.Client(timeout=1.0) as client:
                res = client.get(f"{url}") 
                health["endpoints"][f"ollama_gpu_{gpu_id}"] = "reachable" if res.status_code == 200 else "error"
        except:
            health["endpoints"][f"ollama_gpu_{gpu_id}"] = "unreachable"
            
    # Check Model Servers
    for url in settings.model_server_urls_list:
        try:
             with httpx.Client(timeout=1.0) as client:
                res = client.get(f"{url}/health")
                health["endpoints"][url] = "reachable" if res.status_code == 200 else "error"
        except:
             health["endpoints"][url] = "unreachable"

    return health