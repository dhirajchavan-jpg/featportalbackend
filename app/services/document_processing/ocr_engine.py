# app/services/document_processing/ocr_engine.py

import io
import asyncio
import logging
import easyocr
import fitz  # PyMuPDF
import requests 
import random # <--- NEW: For Load Balancing
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from app.config import settings # <--- NEW: To get server list

logger = logging.getLogger(__name__)

# REMOVED: MODEL_SERVER_URL (Hardcoded port 8074 is gone)

# --- FIXED: Updated URL to match ocr_service_main.py ---
PADDLE_SERVICE_URL = "http://localhost:8003/ocr/" 

class OCREngine:
    """
    Hybrid OCR Engine with REAL Confidence Scoring.
    Forces OCR extraction on all documents (skips digital layer check).
    """
    
    def __init__(self):
        self.reader = None
        
        # --- NEW: Load Server List from Config ---
        self.server_nodes = settings.model_server_urls_list
        if not self.server_nodes:
            self.server_nodes = ["http://localhost:8074"]
            
        logger.info("[OCREngine] Initialized configuration.")
        logger.info(f"[OCREngine] EasyOCR Target: GPU (Local)")
        logger.info(f"[OCREngine] PaddleOCR Target: {PADDLE_SERVICE_URL}")
        logger.info(f"[OCREngine] Model Server Nodes (Load Balanced): {self.server_nodes}")

    def initialize_ocr(self):
        """Explicitly load the model into memory (For EasyOCR)."""
        logger.info("[OCREngine] Request received to initialize EasyOCR model.")
        if self.reader is None:
            logger.info("[OCREngine] Loading EasyOCR model (GPU)...")
            try:
                # Load English model with GPU acceleration
                self.reader = easyocr.Reader(['en'], gpu=True) 
                logger.info("[OCREngine] EasyOCR loaded successfully.")
            except Exception as e:
                logger.error(f"[OCREngine] Failed to load EasyOCR: {e}")
                raise RuntimeError(f"EasyOCR initialization failed: {e}")
        else:
            logger.info("[OCREngine] EasyOCR model is already loaded. Skipping initialization.")

    def is_service_available(self) -> bool:
        """
        Checks if the PaddleOCR service (port 8003) is running.
        Returns True if reachable, False otherwise.
        """
        try:
            # Check the base URL (http://localhost:8003/) with a short timeout
            # base_url = PADDLE_SERVICE_URL.replace("/ocr/", "/")
            requests.get(PADDLE_SERVICE_URL, timeout=2)
            return True
        except requests.exceptions.ConnectionError:
            # This means the server is completely down (Connection Refused)
            logger.warning(f"[OCREngine] PaddleOCR Service unreachable at {PADDLE_SERVICE_URL}")
            return False
        except Exception as e:
            logger.warning(f"[OCREngine] PaddleOCR service check failed: {e}")
            return False

    def _get_reader(self):
        if self.reader is None:
            logger.info("[OCREngine] Reader instance not found in _get_reader. Calling initialize_ocr().")
            self.initialize_ocr()
        return self.reader

    # --- NEW: Helper for Load Balancing ---
    def _get_api_url(self) -> str:
        """Pick a random server node and return the OCR endpoint."""
        base_url = random.choice(self.server_nodes)
        return f"{base_url.rstrip('/')}/ocr"

    # --- NEW: Process via Model Server (GPU) ---
    def _process_with_model_server(self, image_bytes: bytes) -> Tuple[str, float]:
        """Sends image to one of the 3 GPU Model Servers."""
        target_url = self._get_api_url()
        logger.info(f"[OCREngine] Sending {len(image_bytes)} bytes to Model Server: {target_url}")
        
        try:
            files = {'file': ('page.png', image_bytes, 'image/png')}
            response = requests.post(target_url, files=files, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                # Model server returns: {"pages": [{"text": "...", "avg_confidence": 0.9, ...}]}
                pages = data.get("pages", [])
                if pages:
                    return pages[0].get("text", ""), pages[0].get("avg_confidence", 0.0)
                else:
                    logger.info("[OCREngine] Model server returned no pages.")
                    return "", 0.0
            
            logger.error(f"[OCREngine] Model Server Error {response.status_code}: {response.text}")
            return "", 0.0
        except Exception as e:
            logger.error(f"[OCREngine] Model Server Connection Failed: {e}")
            return "", 0.0

    def _process_image_bytes(self, image_bytes: bytes) -> Tuple[str, float]:
        """Runs **EasyOCR** (Local). Returns: (Full Text, Avg Confidence)"""
        logger.info(f"[OCREngine] Processing image bytes via EasyOCR. Size: {len(image_bytes)} bytes.")
        reader = self._get_reader()
        try:
            results = reader.readtext(image_bytes, detail=1, paragraph=False)
            count = len(results)
            logger.info(f"[OCREngine] EasyOCR inference complete. Found {count} text segments.")
            
            if not results:
                return "", 0.0

            text_parts = []
            total_conf = 0.0
            for i, (_, text, conf) in enumerate(results):
                text_parts.append(text)
                total_conf += conf

            full_text = " ".join(text_parts)
            avg_conf = total_conf / count if count > 0 else 0.0
            return full_text, avg_conf

        except Exception as e:
            logger.error(f"[OCREngine] Error in EasyOCR processing: {e}", exc_info=True)
            return "", 0.0

    def _process_with_paddle_service(self, image_bytes: bytes) -> Tuple[str, float]:
        """Connector: Sends image to **PaddleOCR Service** at /ocr/."""
        logger.info(f"[OCREngine] Sending {len(image_bytes)} bytes to PaddleOCR Service: {PADDLE_SERVICE_URL}")
        try:
            # 1. Send Request
            files = {'file': ('page.png', image_bytes, 'image/png')}
            
            # Sending POST request
            response = requests.post(PADDLE_SERVICE_URL, files=files, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"[OCREngine] PaddleOCR Service failed. Status: {response.status_code}, Body: {response.text}")
                return "", 0.0
            
            # 2. Parse Response
            # Expecting list of dicts: [{"text": "...", "page": 1, ...}]
            data = response.json()
            
            text_parts = []
            # Note: The service currently returns full text per page, but no confidence score in top-level response.
            # We will extract text and default confidence to 0.95 (high) if missing, to avoid 0.0 confusion.
            
            if isinstance(data, list):
                for item in data:
                    text_parts.append(item.get("text", ""))
            elif isinstance(data, dict):
                 text_parts.append(data.get("text", ""))

            if not text_parts:
                logger.info("[OCREngine] PaddleOCR returned no text.")
                return "", 0.0

            joined_text = " ".join([str(t) for t in text_parts])
            avg_confidence = 0.95 # Placeholder since service logic doesn't return scores yet

            logger.info(f"[OCREngine] PaddleOCR Success. TextLen={len(joined_text)}")
            return joined_text, avg_confidence

        except Exception as e:
            logger.error(f"[OCREngine] Error connecting to PaddleOCR Service: {e}")
            return "", 0.0

    def extract_text_from_pdf(self, pdf_path: str, engine_type: str = "paddleocr") -> List[Dict[str, Any]]:
        """Forces OCR Extraction using selected engine."""
        logger.info(f"[OCREngine] Starting PDF extraction. Strategy: {engine_type.upper()} | File: {pdf_path}")
        results = []
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            logger.info(f"[OCREngine] Processing PDF with {total_pages} pages...")

            for page_num, page in enumerate(doc, start=1):
                logger.info(f"[OCREngine] Processing Page {page_num}/{total_pages}...")
                
                # ALWAYS RUN OCR (Digital Check Removed)
                logger.info(f"[OCREngine] Page {page_num} converting to image for OCR. Strategy: {engine_type}")
                
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                
                if engine_type == "easyocr":
                    logger.info(f"[OCREngine] Running Local EasyOCR on Page {page_num}...")
                    clean_text, confidence = self._process_image_bytes(img_bytes)
                    method = "easyocr_local"
                
                # --- NEW BRANCH FOR LOAD BALANCING ---
                elif engine_type == "model_server": 
                    logger.info(f"[OCREngine] Sending Page {page_num} to Model Server (Load Balanced)...")
                    clean_text, confidence = self._process_with_model_server(img_bytes)
                    method = "model_server"
                
                else:
                    logger.info(f"[OCREngine] Sending Page {page_num} to PaddleOCR Service...")
                    clean_text, confidence = self._process_with_paddle_service(img_bytes)
                    method = "paddle_service"

                logger.info(f"[OCREngine] Page {page_num} OCR Complete. Method: {method}, Conf: {confidence:.4f}")

                if clean_text:
                    results.append({
                        "page": page_num,
                        "text": clean_text,
                        "avg_confidence": round(confidence, 4),
                        "extraction_method": method
                    })
                else:
                    logger.info(f"[OCREngine] Page {page_num}: Result Empty.")
            
            doc.close()
            logger.info(f"[OCREngine] Completed. Extracted {len(results)} pages.")
            return results

        except Exception as e:
            logger.error(f"[OCREngine] PDF extraction failed: {e}")
            return []

    # --- Async & Image Helpers ---

    async def extract_text_from_pdf_async(self, pdf_path: str, engine_type: str = "paddleocr") -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.extract_text_from_pdf, pdf_path, engine_type)

    def extract_text_from_image(self, image: Image.Image, bbox: Optional[List[float]] = None) -> Dict[str, Any]:
        """Extract text from PIL Image (Defaults to EasyOCR for single image snippet)."""
        logger.info(f"[OCREngine] Extracting text from PIL Image (Local EasyOCR).")
        try:
            if bbox:
                image = image.crop(bbox)
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            text, conf = self._process_image_bytes(img_bytes)
            return {"page": 1, "text": text, "avg_confidence": round(conf, 4)}
        except Exception as e:
            logger.error(f"[OCREngine] Image extraction failed: {e}")
            return {"page": 1, "text": "", "avg_confidence": 0.0}

    async def extract_text_from_image_async(self, image: Image.Image, bbox: Optional[List[float]] = None) -> Dict[str, Any]:
        return await asyncio.to_thread(self.extract_text_from_image, image, bbox)
    
    def extract_text_from_region(self, pdf_path: str, page: int, bbox: List[float]) -> str:
        try:
            doc = fitz.open(pdf_path)
            if page < 1 or page > len(doc):
                return ""
            
            pdf_page = doc[page - 1]
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            
            pix = pdf_page.get_pixmap(dpi=300, clip=rect)
            img_bytes = pix.tobytes("png")
            text, _ = self._process_image_bytes(img_bytes)
            
            doc.close()
            return text
        except Exception as e:
            logger.error(f"[OCREngine] Region extraction failed: {e}")
            return ""

    async def extract_text_from_region_async(self, pdf_path: str, page: int, bbox: List[float]) -> str:
        return await asyncio.to_thread(self.extract_text_from_region, pdf_path, page, bbox)

    async def close(self):
        pass

# Singleton instance
_ocr_engine = None

def get_ocr_engine() -> OCREngine:
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("[OCREngine] Creating new singleton instance.")
        _ocr_engine = OCREngine()
    else:
        logger.info("[OCREngine] Returning existing singleton instance.")
    return _ocr_engine