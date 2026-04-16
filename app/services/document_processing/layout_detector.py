# app/services/document_processing/layout_detector.py
"""
Layer 1: Layout Detection (Client)
Sends images to Model Servers (Load Balanced) for DETR layout detection.
"""

import io
import requests # REQUIRED: To talk to model_server
import logging
import random # <--- NEW: For Load Balancing
from typing import List, Dict, Any, Tuple
from PIL import Image
from pdf2image import convert_from_path
from app.config import settings

logger = logging.getLogger(__name__)

class LayoutDetector:
    """
    Client-Side Layout Detector.
    Offloads heavy DETR processing to the Model Servers (Multi-GPU).
    """
    
    def __init__(self):
        # 1. Load Server List from Config
        self.server_nodes = settings.model_server_urls_list
        
        # Fallback if list is empty
        if not self.server_nodes:
            logger.warning("[LayoutDetector] No model servers found in config! Defaulting to localhost:8074")
            self.server_nodes = ["http://localhost:8074"]

        logger.info(f"[LayoutDetector] Initializing Layout Client with {len(self.server_nodes)} nodes.")
        logger.info(f"[LayoutDetector] Target Nodes: {self.server_nodes}")
    
    def _get_api_url(self) -> str:
        """
        Load Balancer: Picks a random server node and returns the full layout endpoint.
        Example: "http://localhost:8075/layout"
        """
        selected_node = random.choice(self.server_nodes)
        return f"{selected_node.rstrip('/')}/layout"

    def load_model(self):
        """
        No-op in Client Mode. The server handles initialization.
        """
        logger.info("[LayoutDetector] Client mode active. No local model loading required.")
    
    def detect_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Detect layout from PDF file.
        1. Converts PDF to images locally.
        2. Sends each image to a random server node for detection.
        """
        logger.info(f"[LayoutDetector] Processing PDF file: {pdf_path}")

        try:
            logger.info("[LayoutDetector] Converting PDF pages to images (local processing)...")
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=200)
            total_pages = len(images)
            logger.info(f"[LayoutDetector] PDF conversion complete. Total pages: {total_pages}")
            
            all_detections = []
            
            for page_num, image in enumerate(images, start=1):
                logger.info(f"[LayoutDetector] Processing page {page_num}/{total_pages} via Server...")
                
                # Send single image to server
                page_detections = self.detect_from_image(image)
                
                logger.info(f"[LayoutDetector] Page {page_num} finished. Found {len(page_detections)} elements.")
                
                # Assign page numbers to results
                for detection in page_detections:
                    detection['page'] = page_num
                
                all_detections.extend(page_detections)
                
                # Cleanup image memory
                del image
            
            logger.info("[LayoutDetector] Cleaning up raw image list")
            del images
            
            logger.info(f"[LayoutDetector] PDF processing finished. Total elements detected: {len(all_detections)}")
            return all_detections
            
        except Exception as e:
            logger.error(f"[LayoutDetector] Error processing PDF: {e}", exc_info=True)
            return []
    
    def detect_from_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Sends PIL Image to Model Server for layout detection.
        """
        logger.info(f"[LayoutDetector] Preparing image for transmission (Size: {image.size})")
        
        try:
            # 1. Convert PIL Image to Bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # 2. Prepare Request
            files = {'file': ('page.png', img_bytes, 'image/png')}
            
            # 3. Get Load Balanced URL
            target_url = self._get_api_url() # <--- NEW: Pick random server
            
            # 4. Send to Server
            logger.info(f"[LayoutDetector] Sending HTTP POST request to {target_url}...")
            response = requests.post(target_url, files=files, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                # Server returns: {"pages": [{"page": 1, "elements": [...]}]}
                pages = data.get("pages", [])
                
                if pages and len(pages) > 0:
                    elements = pages[0].get("elements", [])
                    logger.info(f"[LayoutDetector] Server returned {len(elements)} elements.")
                    
                    # Ensure bbox area is calculated if missing
                    for el in elements:
                        if 'area' not in el and 'bbox' in el:
                            el['area'] = self._calculate_area(el['bbox'])
                            
                    return elements
                else:
                    logger.info("[LayoutDetector] Server returned no elements for this page.")
                    return []
            else:
                logger.error(f"[LayoutDetector] Server Error ({target_url}): {response.status_code} - {response.text}")
                return []
                
        except requests.exceptions.ConnectionError:
            logger.error(f"[LayoutDetector] FAILED TO CONNECT to {target_url}. Is it running?")
            return []
        except Exception as e:
            logger.error(f"[LayoutDetector] Client-side error during request: {e}", exc_info=True)
            return []
        finally:
            # Explicit cleanup of bytes
            if 'img_byte_arr' in locals():
                img_byte_arr.close()
    
    def _calculate_area(self, bbox: List[float]) -> float:
        """Calculate area of bounding box."""
        x_min, y_min, x_max, y_max = bbox
        return (x_max - x_min) * (y_max - y_min)
    
    def group_by_section(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group detected elements into logical sections.
        """
        logger.info(f"[LayoutDetector] Grouping {len(detections)} elements into sections")
        sections = []
        current_section = {"elements": [], "page": 1}
        
        for detection in detections:
            # Looks for 'title', 'section-header', 'heading'
            if detection['type'] in ['title', 'section-header', 'heading']:
                if current_section['elements']:
                    sections.append(current_section)
                current_section = {
                    "title": detection.get('text', 'Untitled Section'),
                    "elements": [],
                    "page": detection.get('page', 1)
                }
            
            current_section['elements'].append(detection)
        
        if current_section['elements']:
            sections.append(current_section)
        
        logger.info(f"[LayoutDetector] Grouping complete. Created {len(sections)} sections")
        return sections
    
    def extract_tables(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract only table detections."""
        logger.info(f"[LayoutDetector] Extracting tables from {len(detections)} detections")
        tables = [d for d in detections if d['type'] == 'table']
        logger.info(f"[LayoutDetector] Found {len(tables)} tables")
        return tables
    
    def extract_text_blocks(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract only text block detections."""
        logger.info(f"[LayoutDetector] Extracting text blocks from {len(detections)} detections")
        blocks = [d for d in detections if d['type'] in ['text', 'paragraph', 'text-block', 'list']]
        logger.info(f"[LayoutDetector] Found {len(blocks)} text blocks")
        return blocks
    
    def extract_figures(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract only figure/image detections."""
        logger.info(f"[LayoutDetector] Extracting figures from {len(detections)} detections")
        figures = [d for d in detections if d['type'] in ['figure', 'image', 'chart']]
        logger.info(f"[LayoutDetector] Found {len(figures)} figures")
        return figures
    
    def clear_memory(self):
        """No-op in client mode."""
        logger.info("[LayoutDetector] Memory clear requested (No-op in Client Mode).")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """No-op in client mode."""
        return {"message": "Client Mode - Memory managed by Server(s)"}


# Singleton instance
_layout_detector = None

def get_layout_detector() -> LayoutDetector:
    """Get or create singleton LayoutDetector instance."""
    global _layout_detector
    if _layout_detector is None:
        logger.info("[LayoutDetector] Creating new singleton instance")
        _layout_detector = LayoutDetector()
    else:
        logger.info("[LayoutDetector] Returning existing singleton instance")
    return _layout_detector