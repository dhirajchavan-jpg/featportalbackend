# app/services/document_processing/json_builder.py
"""
Layer 1: JSON Output Builder
Combines layout detection, OCR, table parsing, and formula conversion
into structured JSON output.
"""

import json
import os
from typing import Dict, Any, List
from datetime import datetime
from app.config import settings
from .docx_extractor import get_docx_extractor
from .layout_detector import get_layout_detector
from .ocr_engine import get_ocr_engine
from .table_parser import get_table_parser
from .formula_converter import get_formula_converter

import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Orchestrates document processing pipeline."""
    
    def __init__(self):
        logger.info("[DocumentProcessor] Initializing Layer 1 components...")
        self.docx_extractor = get_docx_extractor()
        self.layout_detector = get_layout_detector()
        self.ocr_engine = get_ocr_engine()
        self.table_parser = get_table_parser()
        self.formula_converter = get_formula_converter()
        logger.info("[DocumentProcessor] Initialized all Layer 1 components")
    
    def process_document(
        self, 
        file_path: str, 
        file_id: str,
        project_id: str,
        sector: str,
        ocr_engine_name: str = "paddleocr", # <--- NEW ARGUMENT (Default: paddleocr)
        page_source_path: str = None,
        require_exact_pages: bool = False,
    ) -> Dict[str, Any]:
        """
        Process document through complete Layer 1 pipeline.
        """
        logger.info(f"[DocumentProcessor] Processing {file_path} (File ID: {file_id})")
        logger.info(f"[DocumentProcessor] OCR Strategy Selected: {ocr_engine_name}") # <--- Log the strategy
        logger.info("[DocumentProcessor] Exact page mode: %s", require_exact_pages)
        
        start_time = datetime.utcnow()

        file_ext = os.path.splitext(file_path)[1].lower()
        logger.info("[DocumentProcessor] File extension detected: %s", file_ext or "unknown")

        if file_ext == ".docx":
            logger.info("[DocumentProcessor] DOCX detected. Using native paginated extraction path.")
            extracted_doc = self.docx_extractor.extract_docx(file_path)
            logger.info(
                "[DocumentProcessor] DOCX native extraction returned pages=%s, tables=%s",
                len(extracted_doc.get("pages", [])),
                len(extracted_doc.get("tables", [])),
            )
            pages = extracted_doc.get("pages", [])
            tables = extracted_doc.get("tables", [])
            document_structure = extracted_doc.get("document_structure", self._extract_document_structure(pages))
            extraction_metadata = extracted_doc.get("metadata", {})
            logger.info(
                "[DocumentProcessor] DOCX metadata resolved. extraction_method=%s, ocr_engine=%s",
                extraction_metadata.get("extraction_method"),
                extraction_metadata.get("ocr_engine"),
            )
        else:
            logger.info("[DocumentProcessor] Non-DOCX file detected. Falling back to PDF/OCR processing path.")
            pages, tables, document_structure, extraction_metadata = self._process_pdf_document(
                file_path,
                ocr_engine_name,
            )

        if settings.DETECT_FORMULAS:
            logger.info("[DocumentProcessor] Step 5: Processing formulas...")
            pages = self._process_formulas_in_pages(pages)
            logger.info("[DocumentProcessor] Step 5 Complete.")

        logger.info("[DocumentProcessor] Assembling final JSON structure...")
        document_json = {
            "metadata": {
                "file_id": file_id,
                "project_id": project_id,
                "sector": sector,
                "file_path": file_path,
                "ocr_engine": extraction_metadata.get("ocr_engine", ocr_engine_name),
                "extraction_method": extraction_metadata.get("extraction_method", "ocr_pdf"),
                "processed_at": start_time.isoformat(),
                "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                "total_pages": extraction_metadata.get("total_pages", len(pages)),
                "total_tables": extraction_metadata.get("total_tables", len(tables)),
                "has_formulas": any(p.get('formulas', []) for p in pages),
                "page_source_path": extraction_metadata.get("page_source_path", file_path),
                "logical_source_path": extraction_metadata.get("logical_source_path", file_path),
                "page_numbering": extraction_metadata.get("page_numbering", "physical"),
                "source_format": extraction_metadata.get("source_format", file_ext.lstrip(".")),
            },
            "pages": pages,
            "tables": tables,
            "document_structure": document_structure
        }
        logger.info(
            "[DocumentProcessor] Final document JSON ready. total_pages=%s, total_tables=%s, has_formulas=%s",
            document_json["metadata"]["total_pages"],
            document_json["metadata"]["total_tables"],
            document_json["metadata"]["has_formulas"],
        )
        
        # Save JSON to disk
        logger.info("[DocumentProcessor] Saving JSON to disk...")
        json_path = self._save_json(document_json, file_id)
        document_json['metadata']['json_output_path'] = json_path
        
        logger.info(f"[DocumentProcessor] Processing complete: {json_path}")
        return document_json

    def _process_pdf_document(
        self,
        file_path: str,
        ocr_engine_name: str,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Run the PDF/OCR extraction path and return pages, tables, structure, and metadata."""
        logger.info("[DocumentProcessor] Step 1: Detecting layout...")
        layout_elements = self.layout_detector.detect_from_pdf(file_path)
        logger.info(f"[DocumentProcessor] Step 1 Complete. Detected {len(layout_elements)} layout elements.")

        logger.info("[DocumentProcessor] Step 2: Extracting tables...")
        tables = self.table_parser.extract_tables_from_pdf(file_path)
        logger.info(f"[DocumentProcessor] Step 2 Complete. Found {len(tables)} tables.")

        logger.info(f"[DocumentProcessor] Step 3: Running OCR (Engine: {ocr_engine_name})...")
        try:
            ocr_results = self.ocr_engine.extract_text_from_pdf(
                file_path,
                engine_type=ocr_engine_name
            )
            logger.info(f"[DocumentProcessor] OCR completed: {len(ocr_results)} pages extracted")
        except Exception as e:
            logger.info(f"[DocumentProcessor] OCR Error: {e}")
            import traceback
            traceback.print_exc()
            ocr_results = []

        logger.info("[DocumentProcessor] Step 4: Building structured output...")
        pages = self._build_pages(layout_elements, ocr_results, tables)
        logger.info(f"[DocumentProcessor] Step 4 Complete. Built {len(pages)} pages.")
        document_structure = self._extract_document_structure(pages)
        extraction_metadata = {
            "ocr_engine": ocr_engine_name,
            "extraction_method": "ocr_pdf",
            "page_source_path": file_path,
            "logical_source_path": file_path,
            "page_numbering": "physical",
            "source_format": "pdf",
        }
        logger.info(
            "[DocumentProcessor] PDF/OCR metadata resolved. extraction_method=%s, ocr_engine=%s",
            extraction_metadata["extraction_method"],
            extraction_metadata["ocr_engine"],
        )
        return pages, tables, document_structure, extraction_metadata
    
    def _build_pages(
        self, 
        layout_elements: List[Dict], 
        ocr_results: List[Dict],
        tables: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Build page-by-page structure."""
        logger.info("[DocumentProcessor] Building pages from OCR, Layout, and Tables...")
        pages = {}
        
        # --- FIXED: Robust Page Number Handling ---
        # Use enumerate(idx + 1) as fallback if 'page' key is missing
        for idx, ocr_result in enumerate(ocr_results):
            # Try to get explicit page number, otherwise use index (1-based)
            page_num = ocr_result.get('page') or (idx + 1)
            logger.info("[DocumentProcessor] Registering OCR result for page %s", page_num)
            
            pages[page_num] = {
                "page_number": page_num,
                "layout_elements": [],
                "text_content": ocr_result.get('text', ''),
                "ocr_confidence": ocr_result.get('avg_confidence', 0.0),
                "tables": [],
                "sections": []
            }
        
        # Group layout elements by page
        logger.info(f"[DocumentProcessor] Mapping {len(layout_elements)} layout elements to pages...")
        for element in layout_elements:
            page_num = element.get('page', 1)
            logger.info("[DocumentProcessor] Mapping layout element type=%s to page %s", element.get('type'), page_num)
            
            # Create page entry if Layout found a page that OCR missed
            if page_num not in pages:
                pages[page_num] = {
                    "page_number": page_num,
                    "layout_elements": [],
                    "text_content": "",
                    "tables": [],
                    "sections": []
                }
            pages[page_num]['layout_elements'].append(element)
        
        # Add tables to correct pages
        for table in tables:
            page_num = table.get('page', 1)
            logger.info("[DocumentProcessor] Mapping table index=%s to page %s", table.get('table_index'), page_num)
            
            # Create page entry if Table found a page that OCR/Layout missed
            if page_num not in pages:
                pages[page_num] = {
                    "page_number": page_num,
                    "layout_elements": [],
                    "text_content": "",
                    "tables": [],
                    "sections": []
                }
            pages[page_num]['tables'].append(table)
        
        # Extract sections from layout logic
        logger.info("[DocumentProcessor] Extracting logical sections...")
        for page_num in pages:
            # Only extract sections if we have layout elements
            if pages[page_num]['layout_elements']:
                logger.info("[DocumentProcessor] Extracting sections for page %s", page_num)
                pages[page_num]['sections'] = self._extract_sections(
                    pages[page_num]['layout_elements'],
                    pages[page_num]['text_content']
                )
        
        # Convert to list and sort by page number to ensure order 1, 2, 3...
        sorted_pages = [pages[i] for i in sorted(pages.keys())]
        logger.info("[DocumentProcessor] Page build complete. Total pages assembled=%s", len(sorted_pages))
        return sorted_pages
    
    def _extract_sections(
        self, 
        layout_elements: List[Dict], 
        text: str
    ) -> List[Dict[str, Any]]:
        """Extract logical sections from page."""
        sections = []
        current_section = None
        
        for element in layout_elements:
            element_type = element.get('type', '')
            
            # Start new section on headers
            if element_type in ['title', 'heading', 'section-header']:
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    "type": element_type,
                    "bbox": element.get('bbox'),
                    "elements": [element],
                    "text": "" # Text could be populated if bbox matching is implemented
                }
            elif current_section:
                current_section['elements'].append(element)
        
        # Add last section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _process_formulas_in_pages(self, pages: List[Dict]) -> List[Dict]:
        """Detect and convert formulas in all pages."""
        logger.info(f"[DocumentProcessor] Converting formulas across {len(pages)} pages.")
        for page in pages:
            text = page.get('text_content', '')
            if text:
                formula_result = self.formula_converter.process_text_with_formulas(text)
                page['text_content'] = formula_result['text']
                page['formulas'] = formula_result['formulas']
                page['formula_count'] = formula_result['formula_count']

                if formula_result['formula_count'] > 0:
                    logger.info(f"[DocumentProcessor] Page {page.get('page_number')} processed: {formula_result['formula_count']} formulas found.")
        
        return pages
    
    def _extract_document_structure(self, pages: List[Dict]) -> Dict[str, Any]:
        """Extract high-level document structure."""
        logger.info("[DocumentProcessor] Generating high-level document structure...")
        structure = {
            "title": "",
            "sections": [],
            "has_toc": False,
            "section_hierarchy": []
        }
        
        # Try to identify title from first page
        if pages:
            first_page = pages[0]
            for element in first_page.get('layout_elements', []):
                if element.get('type') == 'title':
                    structure['title'] = element.get('text', 'Untitled Document')
                    logger.info(f"[DocumentProcessor] Document title detected: {structure['title']}")
                    break
        
        # Extract section hierarchy
        for page in pages:
            page_ref = page.get('page_number')
            if page_ref is None:
                page_ref = page.get('logical_page_number')
            for section in page.get('sections', []):
                if section.get('type') in ['heading', 'section-header']:
                    section_entry = {
                        "type": section['type'],
                        "text": section.get('text', '')
                    }
                    if page_ref is not None:
                        section_entry["page"] = page_ref
                    structure['sections'].append(section_entry)


        logger.info(f"[DocumentProcessor] Document structure generated with {len(structure['sections'])} sections.")
        return structure
    
    def _save_json(self, document_json: Dict, file_id: str) -> str:
        """Save JSON output to disk."""
        try:
            json_filename = f"{file_id}_processed.json"
            json_path = os.path.join(settings.JSON_OUTPUT_DIR, json_filename)
            logger.info(f"[DocumentProcessor] Preparing to save JSON to: {json_path}")
            
            os.makedirs(settings.JSON_OUTPUT_DIR, exist_ok=True)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(document_json, f, indent=2, ensure_ascii=False)
            
            logger.info("[DocumentProcessor] JSON file written successfully.")
            return json_path
            
        except Exception as e:
            logger.info(f"[DocumentProcessor] Error saving JSON: {e}")
            return ""
    
    def load_processed_json(self, file_id: str) -> Dict[str, Any]:
        """Load previously processed JSON."""
        try:
            json_filename = f"{file_id}_processed.json"
            json_path = os.path.join(settings.JSON_OUTPUT_DIR, json_filename)
            
            logger.info(f"[DocumentProcessor] Attempting to load processed JSON from: {json_path}")
            if not os.path.exists(json_path):
                logger.info(f"[DocumentProcessor] JSON file not found: {json_path}")
                return None
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info("[DocumentProcessor] JSON loaded successfully.")
                return data

        except Exception as e:
            logger.info(f"[DocumentProcessor] Error loading JSON: {e}")
            return None


# Singleton instance
_document_processor = None

def get_document_processor() -> DocumentProcessor:
    """Get or create singleton DocumentProcessor instance."""
    global _document_processor
    if _document_processor is None:
        logger.info("[DocumentProcessor] Creating new singleton instance.")
        _document_processor = DocumentProcessor()
    else:
        logger.info("[DocumentProcessor] Returning existing singleton instance.")
    return _document_processor
