# model_server.py
# Run: python model_server.py --port 8074

import os
import io
import sys
import asyncio
import argparse # <--- Required for dynamic port selection
import logging
import gc
from sentence_transformers import SentenceTransformer
import json
import time  # <--- ADDED FOR MONITORING (Latency calculation)
import torch
import pynvml # <--- Already imported, will be used now
import psutil # <--- NEW: For process name resolution if NVML fails
import numpy as np
import fitz  # PyMuPDF
import easyocr
import uvicorn
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any

# HuggingFace / Transformers
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from transformers import AutoImageProcessor, DetrForSegmentation
from PIL import Image

# --- LOGGER SETUP (IST Timezone) ---
def setup_logger():
    """Setup application logger with Indian Standard Time (IST)."""
    ist_timezone = ZoneInfo("Asia/Kolkata")

    def ist_converter(*args):
        return datetime.now(ist_timezone).timetuple()
    
    logging.Formatter.converter = ist_converter

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # CHANGED: Added PID to filename to prevent conflicts when running multiple servers
    pid = os.getpid()
    timestamp = datetime.now(ist_timezone).strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"app_{timestamp}_{pid}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    local_logger = logging.getLogger("ModelServer")
    local_logger.info(f" Logging initialized in IST. Log file: {log_file}")
    return local_logger

# Initialize Logger
logger = setup_logger()

# --- CONFIGURATION ---

# =========================================================================
# CHANGED: Commented out to allow dynamic GPU selection via terminal
# =========================================================================
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

logger.info(f" Starting Model Server on {DEVICE}...")

# Model Paths
EMBED_MODEL_NAME = "BAAI/bge-m3"

# KEPT ORIGINAL RERANKER AS REQUESTED
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

LAYOUT_MODEL_NAME = "cmarkea/detr-layout-detection"
EVAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # <--- ADD THIS

app = FastAPI(title="RAG Model Server (GPU)")

# --- GLOBAL MODEL STORAGE ---
models = {}

# --- ADDED: SPLIT LOCKS FOR BOTTLENECK FIX ---
# fast_lock: For Embeddings, Reranking, Eval (Low Latency)
# heavy_lock: For OCR, Layout (High Latency, High VRAM spikes)
fast_lock = asyncio.Lock()
heavy_lock = asyncio.Lock()

# =========================================================================
#  MONITORING STORAGE & HELPERS (UPDATED)
# =========================================================================
# Added "vram_mb" to track estimated memory usage per model
model_stats = {
    "embedder": {"count": 0, "last_latency_ms": 0, "avg_latency_ms": 0, "status": "idle", "vram_mb": 0},
    "reranker": {"count": 0, "last_latency_ms": 0, "avg_latency_ms": 0, "status": "idle", "vram_mb": 0},
    "ocr":      {"count": 0, "last_latency_ms": 0, "avg_latency_ms": 0, "status": "idle", "vram_mb": 0},
    "layout":   {"count": 0, "last_latency_ms": 0, "avg_latency_ms": 0, "status": "idle", "vram_mb": 0}
}

def update_stats(model_name, duration_sec):
    """Update global stats for a specific model."""
    s = model_stats[model_name]
    s["count"] += 1
    s["last_latency_ms"] = round(duration_sec * 1000, 2)
    # Simple moving average for smoothness
    if s["avg_latency_ms"] == 0:
        s["avg_latency_ms"] = s["last_latency_ms"]
    else:
        s["avg_latency_ms"] = round((s["avg_latency_ms"] * 0.9) + (s["last_latency_ms"] * 0.1), 2)
    s["status"] = "idle"

def estimate_model_size(model_obj) -> float:
    """Calculates approximate VRAM usage of a PyTorch model in MB."""
    try:
        if hasattr(model_obj, "parameters"):
            # Standard PyTorch model
            mem_params = sum([param.nelement() * param.element_size() for param in model_obj.parameters()])
            mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model_obj.buffers()])
            return round((mem_params + mem_bufs) / 1024**2, 2)
        else:
            return 0.0
    except Exception:
        return 0.0
# =========================================================================


@app.on_event("startup")
async def load_models():
    """Load all heavy models into GPU VRAM at startup."""
    logger.info(" Starting Model Loading Sequence... (This may take 1-2 minutes)")
    
    # --- ADDED FOR MONITORING: Initialize NVML ---
    try:
        pynvml.nvmlInit()
        logger.info(" [Monitor] NVIDIA Management Library Initialized.")
    except Exception as e:
        logger.warning(f" [Monitor] Failed to init NVML: {e}")
    # ---------------------------------------------

    try:
        # 1. Dense Embedder
        logger.info(f"   1/4 Loading Embedder ({EMBED_MODEL_NAME})...")
        models['embedder'] = BGEM3FlagModel(EMBED_MODEL_NAME, use_fp16=True, device=DEVICE)
        
        # Estimate Size
        if hasattr(models['embedder'], 'model'):
            model_stats['embedder']['vram_mb'] = estimate_model_size(models['embedder'].model)
            
        logger.info("    Embedder Loaded.")
        
        # 2. Reranker
        logger.info(f"   2/4 Loading Reranker ({RERANK_MODEL_NAME})...")
        reranker = FlagReranker(RERANK_MODEL_NAME, use_fp16=True, device=DEVICE)
        
        # --- QWEN RERANKER FIXES ---
        if reranker.tokenizer.pad_token is None:
            reranker.tokenizer.pad_token = reranker.tokenizer.eos_token
            reranker.tokenizer.pad_token_id = reranker.tokenizer.eos_token_id
        reranker.tokenizer.padding_side = "right"
        reranker.model.to(DEVICE)
        reranker.model.eval()
        models['reranker'] = reranker
        
        # Estimate Size
        model_stats['reranker']['vram_mb'] = estimate_model_size(reranker.model)
        
        logger.info("    Reranker Loaded.")
        # ---------------------------
        
        # 3. OCR Engine
        logger.info(f"   3/4 Loading EasyOCR (English)...")
        models['ocr'] = easyocr.Reader(['en'], gpu=(DEVICE == "cuda"))
        
        # Estimate Size (Detector + Recognizer)
        ocr_size = estimate_model_size(models['ocr'].detector) + estimate_model_size(models['ocr'].recognizer)
        model_stats['ocr']['vram_mb'] = round(ocr_size, 2)
        
        logger.info("    EasyOCR Loaded.")
        
        # 4. Layout Detector (UPDATED)
        logger.info(f"   4/4 Loading Layout Detector ({LAYOUT_MODEL_NAME})...")
        models['layout_processor'] = AutoImageProcessor.from_pretrained(LAYOUT_MODEL_NAME)
        models['layout_model'] = DetrForSegmentation.from_pretrained(LAYOUT_MODEL_NAME)
        models['layout_model'].to(DEVICE)
        models['layout_model'].eval()
        
        # Estimate Size
        model_stats['layout']['vram_mb'] = estimate_model_size(models['layout_model'])
        
        logger.info("    Layout Detector Loaded.")
        logger.info(f"   5/5 Loading Eval Model ({EVAL_MODEL_NAME})...")
        models['evaluator'] = SentenceTransformer(EVAL_MODEL_NAME, device=DEVICE)
        models['evaluator'].eval()
        
        model_stats['evaluator'] = {"count": 0, "vram_mb": 200, "status": "idle"}
        logger.info("    Eval Model Loaded.")
        logger.info(f" ALL MODELS LOADED. Memory Stats: {json.dumps({k: v['vram_mb'] for k,v in model_stats.items()})}")
    except Exception as e:
        logger.critical(f" Critical Error Loading Models: {e}", exc_info=True)
        raise e

# --- ADDED FOR MONITORING: Shutdown NVML ---
@app.on_event("shutdown")
def shutdown_event():
    try:
        pynvml.nvmlShutdown()
    except:
        pass
# -------------------------------------------


# --- Pydantic Models ---
class EmbedRequest(BaseModel):
    texts: List[str]

class RerankRequest(BaseModel):
    query: str
    documents: List[Dict[str, Any]]
    top_k: int = 5

# --- HELPER: SAFE RERANKING ---
def safe_compute_score(reranker_instance, pairs, batch_size=4):
    """Manual inference loop for Qwen3-Reranker compatibility."""
    tokenizer = reranker_instance.tokenizer
    model = reranker_instance.model
    all_scores = []
    
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            try:
                inputs = tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt', 
                    max_length=8192 
                ).to(DEVICE)

                scores = model(**inputs, return_dict=True).logits.view(-1,).float()
                all_scores.extend(scores.cpu().numpy().tolist())
            
            except Exception as e:
                logger.error(f" Rerank Batch Error at index {i}: {e}", exc_info=True)
                all_scores.extend([-10.0] * len(batch))
            
            finally:
                if 'inputs' in locals(): del inputs
                if 'scores' in locals(): del scores
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    return [1 / (1 + np.exp(-s)) for s in all_scores]

# --- ENDPOINTS ---

# =========================================================================
#  NEW MONITORING ENDPOINT (UPDATED FOR PROCESS BREAKDOWN)
# =========================================================================
@app.get("/monitor", tags=["Monitoring"])
def get_gpu_and_model_stats():
    """
    Returns Real-Time GPU Stats (via pynvml) + Process List + Model Usage Latency.
    """
    gpu_data = []
    
    # 1. Fetch GPU Stats
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes): 
                name = name.decode('utf-8')
            
            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # --- PROCESS BREAKDOWN ---
            active_processes = []
            project_vram_usage = 0.0
            
            try:
                # Get list of processes running on GPU
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                
                for p in procs:
                    try:
                        # Try NVML Name
                        p_name = pynvml.nvmlSystemGetProcessName(p.pid).decode('utf-8')
                    except:
                        # Fallback to psutil
                        try:
                            p_name = psutil.Process(p.pid).name()
                        except:
                            p_name = "unknown"

                    mem_used = round(p.usedGpuMemory / 1024**2, 2)
                    
                    active_processes.append({
                        "pid": p.pid,
                        "name": p_name,
                        "memory_used_mb": mem_used
                    })

                    # Calculate "My Project" usage (Adjust keywords if needed)
                    # "python" covers 'python.exe' and 'python3'
                    # "uvicorn" covers the server process
                    if "python" in p_name.lower() or "uvicorn" in p_name.lower():
                        project_vram_usage += mem_used

            except Exception as e:
                # Often happens if no permission to read other PIDs
                pass
            # -------------------------

            gpu_data.append({
                "id": i,
                "name": name,
                "memory_used_mb": int(mem_info.used / 1024**2),
                "memory_total_mb": int(mem_info.total / 1024**2),
                "gpu_util_percent": util.gpu,
                "temp_c": temp,
                "processes": active_processes,          # <--- Detailed List
                "project_memory_used_mb": round(project_vram_usage, 2) # <--- Your usage
            })
            
    except Exception as e:
        # If NVML fails (e.g. running on CPU), just log error and return empty list
        pass

    # 2. Return Combined Stats
    return {
        "server_status": "active",
        "gpus": gpu_data,
        "models": model_stats
    }
# =========================================================================


@app.post("/embed")
async def embed_texts(req: EmbedRequest):
    model_stats["embedder"]["status"] = "processing" 
    req_start_time = time.time() 

    logger.info(f" [Embed] Received request for {len(req.texts)} text chunks.")
    start_time = datetime.now()
    
    def _inference():
        try:
            res = models['embedder'].encode(
                req.texts, 
                batch_size=12, 
                max_length=512
            )['dense_vecs']
            return res
        except Exception as e:
            logger.error(f" [Embed] Inference failed: {e}", exc_info=True)
            raise e

    # CHANGED: Use fast_lock (Allows concurrency with OCR)
    try: 
        async with fast_lock:
            embeddings = await asyncio.to_thread(_inference)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f" [Embed] Success. Generated {len(embeddings)} vectors in {duration:.2f}s.")
        
        update_stats("embedder", time.time() - req_start_time) 
        return {"embeddings": embeddings.tolist()}
    
    except Exception as e:
        model_stats["embedder"]["status"] = "error" 
        raise e


@app.post("/rerank")
async def rerank_docs(req: RerankRequest):
    doc_count = len(req.documents)
    logger.info(f" [Rerank] Received query: '{req.query[:50]}...' with {doc_count} docs.")
    
    if not req.documents:
        logger.warning(" [Rerank] No documents provided.")
        return {"results": []}

    model_stats["reranker"]["status"] = "processing" 
    req_start_time = time.time() 
    
    def _inference():
        try:
            pairs = []
            for d in req.documents:
                content = d.get('content', '') or ""
                pairs.append([req.query, content[:12000]]) 

            logger.info(f"   [Rerank] Computing scores for {len(pairs)} pairs...")
            scores = safe_compute_score(models['reranker'], pairs)
            return scores
        except Exception as e:
            logger.error(f" [Rerank] Inference failed: {e}", exc_info=True)
            raise e

    # CHANGED: Use fast_lock (Allows concurrency with OCR)
    try: 
        async with fast_lock:
            scores = await asyncio.to_thread(_inference)
        
        scored_docs = []
        for doc, score in zip(req.documents, scores):
            doc['rerank_score'] = float(score)
            scored_docs.append(doc)
            
        ranked = sorted(scored_docs, key=lambda x: x['rerank_score'], reverse=True)[:req.top_k]
        logger.info(f" [Rerank] Success. Returned top {len(ranked)} docs.")
        
        update_stats("reranker", time.time() - req_start_time) 
        return {"results": ranked}

    except Exception as e:
        model_stats["reranker"]["status"] = "error" 
        raise e


@app.post("/ocr")
async def process_ocr(file: UploadFile = File(...)):
    """Legacy endpoint: Accepts PDF, returns text per page."""
    model_stats["ocr"]["status"] = "processing" 
    req_start_time = time.time() 

    logger.info(f" [OCR] Received file: {file.filename}")
    content = await file.read()
    
    def _inference():
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            logger.info(f"   [OCR] PDF opened. Total pages: {len(doc)}")
            results = []
            for page_num, page in enumerate(doc, start=1):
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                logger.info(f"   [OCR] Processing Page {page_num}...")
                text_list = models['ocr'].readtext(img_bytes, detail=0, paragraph=True)
                text = " ".join(text_list)
                if text.strip():
                    results.append({
                        "page": page_num, 
                        "text": text.strip(), 
                        "avg_confidence": 0.9
                    })
            return results
        except Exception as e:
            logger.error(f" [OCR] Processing failed: {e}", exc_info=True)
            raise e

    # CHANGED: Use heavy_lock (Blocks Layout but allows Embed/Rerank)
    try: 
        async with heavy_lock:
            results = await asyncio.to_thread(_inference)
        
        logger.info(f"[OCR] Completed. Extracted text from {len(results)} pages.")
        
        update_stats("ocr", time.time() - req_start_time) 
        return {"pages": results}

    except Exception as e:
        model_stats["ocr"]["status"] = "error" 
        raise e


@app.post("/layout")
async def detect_layout(file: UploadFile = File(...)):
    """
    UPDATED: Accepts an IMAGE (PNG/JPG), not a PDF.
    """
    model_stats["layout"]["status"] = "processing" 
    req_start_time = time.time() 

    logger.info(f"[Layout] Received image file: {file.filename}")
    content = await file.read()

    def _inference():
        layout_results = []
        processor = models['layout_processor']
        model = models['layout_model']
        
        try:
            # 1. Open Image
            img = Image.open(io.BytesIO(content)).convert("RGB")
            logger.info(f"   [Layout] Image opened. Size: {img.size}")
            
            # 2. Preprocess
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            
            # 3. Inference
            logger.info("   [Layout] Running inference...")
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 4. Post-process
            target_sizes = torch.tensor([img.size[::-1]]).to(DEVICE)
            results = processor.post_process_object_detection(
                outputs, 
                threshold=0.5,
                target_sizes=target_sizes
            )[0]
            
            # 5. Format Results
            page_elements = []
            id2label = model.config.id2label
            
            count = 0
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box_values = [round(i, 2) for i in box.tolist()]
                label_name = id2label[label.item()]
                count += 1
                
                page_elements.append({
                    "type": label_name, 
                    "score": round(score.item(), 4),
                    "bbox": box_values
                })
            
            logger.info(f"   [Layout] Detected {count} elements.")

            if page_elements:
                layout_results.append({
                    "page": 1, 
                    "elements": page_elements
                })
                
        except Exception as e:
            logger.error(f" [Layout] Error during inference: {e}", exc_info=True)
            
        finally:
            # CLEANUP
            if 'inputs' in locals(): del inputs
            if 'outputs' in locals(): del outputs
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()
                logger.info("   [Layout] VRAM Cache Cleared.")
                
        return layout_results

    # CHANGED: Use heavy_lock
    try: 
        async with heavy_lock:
            results = await asyncio.to_thread(_inference)
        
        logger.info(" [Layout] Request processing finished.")
        
        update_stats("layout", time.time() - req_start_time) 
        return {"pages": results}

    except Exception as e:
        model_stats["layout"]["status"] = "error" 
        raise e

class EvalRequest(BaseModel):
    texts: List[str]

@app.post("/vectorize_eval")
async def vectorize_for_eval(req: EvalRequest):
    """
    Specific endpoint for Evaluation metrics (using MiniLM).
    """
    model_stats["evaluator"]["status"] = "processing"
    
    def _inference():
        # Encode with normalization (ideal for cosine similarity)
        return models['evaluator'].encode(req.texts, normalize_embeddings=True).tolist()

    # CHANGED: Use fast_lock
    try:
        async with fast_lock:
            embeddings = await asyncio.to_thread(_inference)
        
        return {"embeddings": embeddings}
    except Exception as e:
        logger.error(f" [Eval] Error: {e}")
        raise e
    finally:
        model_stats["evaluator"]["status"] = "idle"

# =========================================================================
# CHANGED: Replaced hardcoded execution with argparse
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8074, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    
    # Run uvicorn with the parsed arguments
    uvicorn.run(app, host=args.host, port=args.port, workers=1)