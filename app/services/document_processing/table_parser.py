# app/services/document_processing/table_parser.py
"""
Layer 1: Table Parser
Extracts and structures tables from documents using pdfplumber, camelot, or tabula.
"""

import pdfplumber
import pandas as pd
from typing import List, Dict, Any, Optional
from app.config import settings

import logging

logger = logging.getLogger(__name__)


class TableParser:
    """Parse and structure tables from PDF documents."""
    
    def __init__(self, parser: str = None):
        self.parser = parser or settings.TABLE_PARSER
        logger.info(f"[TableParser] Initializing TableParser. Selected strategy: {self.parser}")
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract all tables from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of parsed tables with metadata
        """
        logger.info(f"[TableParser] Starting table extraction for file: {pdf_path}")
        
        if self.parser == "pdfplumber":
            logger.info("[TableParser] Routing to pdfplumber extractor.")
            return self._extract_with_pdfplumber(pdf_path)
        elif self.parser == "camelot":
            logger.info("[TableParser] Routing to Camelot extractor.")
            return self._extract_with_camelot(pdf_path)
        elif self.parser == "tabula":
            logger.info("[TableParser] Routing to Tabula extractor.")
            return self._extract_with_tabula(pdf_path)
        else:
            logger.info(f"[TableParser] Unknown parser configuration '{self.parser}'. Falling back to pdfplumber.")
            return self._extract_with_pdfplumber(pdf_path)
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber."""
        logger.info("[TableParser] Opening PDF with pdfplumber...")
        try:
            tables_data = []
            
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"[TableParser] PDF opened. Total pages: {total_pages}")
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    logger.info(f"[TableParser] Scanning page {page_num} for tables...")
                    tables = page.extract_tables()
                    
                    if not tables:
                        logger.info(f"[TableParser] No tables found on page {page_num}.")
                        continue
                        
                    logger.info(f"[TableParser] Found {len(tables)} tables on page {page_num}. Processing...")
                    
                    for table_idx, table in enumerate(tables):
                        # Skip empty or very small tables
                        if not table or len(table) < 2:
                            logger.info(f"[TableParser] Skipping table {table_idx} on page {page_num} (too small or empty).")
                            continue
                        
                        logger.info(f"[TableParser] Converting table {table_idx} to DataFrame.")
                        df = pd.DataFrame(table[1:], columns=table[0])
                        
                        logger.info(f"[TableParser] Cleaning DataFrame for table {table_idx}...")
                        df = self._clean_dataframe(df)
                        
                        tables_data.append({
                            "page": page_num,
                            "table_index": table_idx,
                            "data": df.to_dict('records'),
                            "headers": df.columns.tolist(),
                            "rows": len(df),
                            "columns": len(df.columns),
                            "raw_data": table
                        })
            
            logger.info(f"[TableParser] pdfplumber extraction complete. Total tables extracted: {len(tables_data)}")
            return tables_data
            
        except Exception as e:
            logger.error(f"[TableParser] Critical error during pdfplumber extraction: {e}", exc_info=True)
            return []
    
    def _extract_with_camelot(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables using Camelot."""
        logger.info("[TableParser] Starting Camelot extraction...")
        try:
            import camelot
            
            tables_data = []
            logger.info(f"[TableParser] Running camelot.read_pdf on {pdf_path} (flavor='lattice')...")
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            
            logger.info(f"[TableParser] Camelot detected {len(tables)} tables.")
            
            for idx, table in enumerate(tables):
                logger.info(f"[TableParser] Processing Camelot table {idx} (Page {table.page})...")
                df = table.df
                
                # Try to promote the first row to header if headers are generic ints
                if not df.empty and all(isinstance(c, int) for c in df.columns):
                    logger.info(f"[TableParser] Promoting first row to header for table {idx}.")
                    new_header = df.iloc[0] # Grab the first row
                    df = df[1:] # Take the data less the header
                    df.columns = new_header # Set the header
                    df = df.reset_index(drop=True)
                
                logger.info(f"[TableParser] Cleaning DataFrame for table {idx}...")
                df = self._clean_dataframe(df)
                
                tables_data.append({
                    "page": table.page,
                    "table_index": idx,
                    "data": df.to_dict('records'),
                    "headers": df.columns.tolist(),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "accuracy": table.accuracy,
                    "whitespace": table.whitespace
                })
            
            logger.info(f"[TableParser] Camelot extraction complete. Total tables: {len(tables_data)}")
            return tables_data
            
        except Exception as e:
            logger.error(f"[TableParser] Error with Camelot: {e}", exc_info=True)
            return []
    
    def _extract_with_tabula(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables using Tabula."""
        logger.info("[TableParser] Starting Tabula extraction...")
        try:
            import tabula
            
            tables_data = []
            logger.info(f"[TableParser] Running tabula.read_pdf on {pdf_path}...")
            tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            logger.info(f"[TableParser] Tabula returned {len(tables)} objects.")
            
            for idx, df in enumerate(tables):
                if df.empty:
                    logger.info(f"[TableParser] Skipping empty table at index {idx}.")
                    continue
                
                logger.info(f"[TableParser] Cleaning DataFrame for table {idx}...")
                df = self._clean_dataframe(df)
                
                tables_data.append({
                    "page": None,  # Tabula doesn't provide page info easily
                    "table_index": idx,
                    "data": df.to_dict('records'),
                    "headers": df.columns.tolist(),
                    "rows": len(df),
                    "columns": len(df.columns)
                })
            
            logger.info(f"[TableParser] Tabula extraction complete. Total tables: {len(tables_data)}")
            return tables_data
            
        except Exception as e:
            logger.error(f"[TableParser] Error with Tabula: {e}", exc_info=True)
            return []
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize DataFrame."""
        try:
            logger.info(f"[TableParser] _clean_dataframe started. Input shape: {df.shape}")
            
            # --- FIX for UserWarning: DataFrame columns are not unique ---
            cols = df.columns
            if cols.has_duplicates:
                logger.info("[TableParser] Duplicate columns detected. Renaming...")
                # Create a counter for each name
                counts = {col: 0 for col in cols}
                new_cols = []
                for col in cols:
                    count = counts[col]
                    counts[col] += 1
                    if count > 0:
                        # Rename duplicate: 'Value' -> 'Value.1', 'Value.2'
                        new_cols.append(f"{col}.{count}") 
                    else:
                        new_cols.append(col) # Keep original
                df.columns = new_cols
            # --- END FIX ---

            # Remove completely empty rows and columns
            original_shape = df.shape
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            if df.shape != original_shape:
                logger.info(f"[TableParser] Dropped empty rows/cols. New shape: {df.shape}")
            
            # Iterate by column *index* to bypass duplicate-name issues.
            logger.info("[TableParser] Normalizing column types to string and stripping whitespace.")
            for col_idx, col_type in enumerate(df.dtypes):
                if col_type == 'object':
                    # Use .iloc to access the column by its integer index
                    df.iloc[:, col_idx] = df.iloc[:, col_idx].astype(str).str.strip()
            
            # Replace NaN with empty string
            df = df.fillna('')
            
            # Reset index
            df = df.reset_index(drop=True)
            
            logger.info("[TableParser] DataFrame cleaning complete.")
            return df
            
        except Exception as e:
            logger.error(f"[TableParser] Error cleaning dataframe: {e}", exc_info=True)
            return df
    
    def extract_table_from_region(
        self, 
        pdf_path: str, 
        page: int, 
        bbox: List[float]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract table from specific region.
        
        Args:
            pdf_path: Path to PDF file
            page: Page number (1-indexed)
            bbox: Bounding box [x_min, y_min, x_max, y_max]
            
        Returns:
            Parsed table data or None
        """
        logger.info(f"[TableParser] Region extraction requested. Page: {page}, Bbox: {bbox}")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page > len(pdf.pages):
                    logger.warning(f"[TableParser] Page {page} out of range (Total: {len(pdf.pages)}).")
                    return None
                
                page_obj = pdf.pages[page - 1]
                
                logger.info(f"[TableParser] Cropping page {page} to bounding box...")
                cropped = page_obj.crop(bbox)
                
                logger.info("[TableParser] Extracting tables from cropped region...")
                tables = cropped.extract_tables()
                
                if not tables or len(tables[0]) < 2:
                    logger.info("[TableParser] No valid tables found in specified region.")
                    return None
                
                logger.info("[TableParser] Table found. Processing...")
                table = tables[0]
                df = pd.DataFrame(table[1:], columns=table[0])
                df = self._clean_dataframe(df)
                
                logger.info("[TableParser] Region table extraction successful.")
                return {
                    "page": page,
                    "data": df.to_dict('records'),
                    "headers": df.columns.tolist(),
                    "rows": len(df),
                    "columns": len(df.columns)
                }
                
        except Exception as e:
            logger.error(f"[TableParser] Error extracting table from region: {e}", exc_info=True)
            return None
    
    def table_to_markdown(self, table_data: Dict[str, Any]) -> str:
        """Convert table data to markdown format."""
        logger.info(f"[TableParser] Converting table to markdown (Rows: {len(table_data.get('data', []))}).")
        try:
            df = pd.DataFrame(table_data['data'])
            return df.to_markdown(index=False)
        except Exception as e:
            logger.error(f"[TableParser] Error converting to markdown: {e}", exc_info=True)
            return ""
    
    def table_to_json(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert table to JSON format."""
        logger.info("[TableParser] Converting table to JSON format.")
        return {
            "headers": table_data['headers'],
            "rows": table_data['data']
        }


# Singleton instance
_table_parser = None

def get_table_parser() -> TableParser:
    """Get or create singleton TableParser instance."""
    global _table_parser
    if _table_parser is None:
        logger.info("[TableParser] Creating new singleton instance.")
        _table_parser = TableParser()
    else:
        logger.info("[TableParser] Returning existing singleton instance.")
    return _table_parser