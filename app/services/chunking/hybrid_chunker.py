"""
Layer 2: Hybrid Compliance Chunker (Patched)
Strategy:
1. Global State Tracking: Remembers Section Headers across page boundaries.
2. Legal Regex Chunking: "Eats" text from one clause number to the next.
3. **Context Injection (NEW):** Merges definition headers with their list items.
4. Markdown Tables: Preserves table structure.
5. Deterministic Fallback: Semantic chunking is used ONLY if structure is completely missing.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings
import logging
import re

logger = logging.getLogger(__name__)

class HybridChunker:
    def __init__(self, embeddings):
        logger.info("[HybridChunker] Initializing HybridChunker...")
        self.embeddings = embeddings
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        
        # Initialize semantic chunker
        logger.info(f"[HybridChunker] Setting up SemanticChunker with threshold type 'percentile'.")
        # Fallback 1: Semantic (Only for purely unstructured text blocks)
        self.semantic_chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile"
        )
        
        # Initialize recursive chunker as fallback
        logger.info(f"[HybridChunker] Setting up RecursiveCharacterTextSplitter (Size: {self.chunk_size}, Overlap: {self.chunk_overlap}).")
        # Fallback 2: Recursive (Safety net)
        self.recursive_chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        logger.info(f"[HybridChunker] Initialization complete.")
    
        logger.info(f"[HybridChunker] Initialized. Primary Strategy: Legal Regex + Sticky Context.")

    def _clean_text(self, text: str) -> str:
        """
        Aggressive cleanup for OCR artifacts common in Indian compliance docs.
        """
        if not text: return ""
        
        # 1. Fix Rupee symbol often OCR'd as backtick or '?'
        text = text.replace("` ", "₹").replace("Rs .", "Rs.")
        
        # 2. Remove common page footers/headers (e.g., "Page | 10", "Confidential")
        text = re.sub(r'(?i)Page\s*\|?\s*\d+', '', text)
        
        # 3. Fix "Floating Integers" (OCR noise: isolated numbers on lines)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # 4. Merge broken sentences (hyphenated at line end)
        text = re.sub(r'([a-z])-\n\s*([a-z])', r'\1\2', text)
        
        # 5. Collapse multiple newlines to max 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    def chunk_document(self, document_json: Dict[str, Any]) -> List[Document]:
        """
        Master function with Global Context Tracking.
        """
        all_chunks = []
        pages = document_json.get('pages', [])
        metadata_base = document_json.get('metadata', {})
        file_name = metadata_base.get('file_name', 'Unknown File')
        
        # GLOBAL CONTEXT TRACKER
        # Persists across pages to keep track of "Chapter IV", etc.
        section_tracker = {
            "current_header": "Preamble / Introduction",
            "last_clause_index": 0
        }

        # Base context string
        file_context_str = f"File: {file_name}"

        for page in pages:
            page_num = page.get('page_number', 1)
            raw_text = page.get('text_content', '')
            page_text = self._clean_text(raw_text)
            
            # --- PROCESS TABLES FIRST ---
            if page.get('tables'):
                table_chunks = self._chunk_tables(
                    page['tables'], page_num, metadata_base, file_context_str
                )
                all_chunks.extend(table_chunks)

            # --- PROCESS TEXT ---
            if page_text and len(page_text) >= 50:
                
                # Primary Strategy: Legal Clause Regex
                if settings.ENABLE_SECTION_CHUNKING:
                    clause_chunks = self._chunk_by_legal_clauses(
                        page_text, 
                        page_num, 
                        metadata_base, 
                        file_context_str, 
                        section_tracker  # Pass reference to update definition headers
                    )
                    all_chunks.extend(clause_chunks)
                    
                # Fallback: Semantic (Only if regex found 0 structure)
                elif settings.ENABLE_SEMANTIC_CHUNKING:
                    logger.warning(f"Page {page_num}: No legal structure found, using Semantic.")
                    semantic_chunks = self._chunk_semantically(
                        page_text, page_num, metadata_base, 
                        context_prefix=f"{file_context_str} > Content: "
                    )
                    logger.info(f"[HybridChunker] Page {page_num}: Generated {len(semantic_chunks)} semantic chunks.")
                    all_chunks.extend(semantic_chunks)
                
                else:
                    logger.info(f"[HybridChunker] Page {page_num}: Strategy 3 - Recursive Chunking (Default/Fallback).")
                    recursive_chunks = self._chunk_recursively(
                        page_text, page_num, metadata_base,
                        context_prefix=f"{file_context_str} > Content: "
                    )
                    logger.info(f"[HybridChunker] Page {page_num}: Generated {len(recursive_chunks)} recursive chunks.")
                    all_chunks.extend(recursive_chunks)

        # Generate Embeddings (Batched)
        # Note: If your VectorStore handles embedding on upsert, comment this out.
        self._generate_embeddings_batched(all_chunks)

        return all_chunks

    def _chunk_by_legal_clauses(
        self,
        text: str,
        page_num: int,
        base_metadata: Dict,
        file_context_str: str,
        section_tracker: Dict
    ) -> List[Document]:
        """
        Regex-based splitting with 'Sticky Header' logic.
        Injects definition headers into subsequent list items.
        """
        # logger.info(f"[HybridChunker] Splitting text by sections (Page {page_num})...") 
        # (Detailed loop logging commented out to prevent spam)
        chunks = []
        
        # --- ROBUST COMPLIANCE REGEX (Updated) ---
        header_pattern = (
            r'(?:^|\n)\s*'  # Anchor to start of line
            r'(?:'
            r'(?:CHAPTER|SECTION|PART|APPENDIX|SCHEDULE|PARAGRAPH|PARA)\s+[IVXLCDM\dA-Z\-\.]+|' 
            r'Article\s+[IVXLCDM\d]+|'       # Matches "Article 5"
            r'(?:\d+\.)+\d{1,3}(?!\d)|'      # Matches "1.2.3"
            r'\d+\.(?!\d)|'                  # Matches "1."
            r'\([a-z0-9]+\)|'                # Matches "(a)" or "(ii)"
            r'[a-z]\)|'                      # Matches "a)"
            r'Explanation\s*[-\:]?|'         # Matches "Explanation -"
            r'Note\s*[-\:]?'                 # Matches "Note :"
            r')'
            r'\s+' 
        )
        
        # Split text but keep the delimiters
        parts = re.split(f'({header_pattern})', text, flags=re.MULTILINE)
        
        # --- THE FIX: Lead-in Context Handling ---
        # parts[0] is the text BEFORE the first list item.
        preamble_text = parts[0].strip()
        local_definition_context = ""

        # Logic: If preamble is short and ends with markers like ':', '-', or 'means',
        # it is a header for the list items that follow.
        if preamble_text and (
            preamble_text.endswith(':') or 
            preamble_text.endswith('-') or 
            "means" in preamble_text[-150:] # Look for "means" in the last few words
        ):
            local_definition_context = preamble_text
            # We do NOT save it as a separate chunk yet. We save it to inject later.
        elif preamble_text:
            # It's just normal text from the previous page, chunk it normally
            context_str = f"{file_context_str} > Section: {section_tracker['current_header']}"
            chunks.extend(self._create_chunks(preamble_text, context_str, page_num, base_metadata, "continuation"))

        # Iterate through pairs: (Header Marker, Content)
        for i in range(1, len(parts), 2):
            header_marker = parts[i].strip()
            content = parts[i+1] if i+1 < len(parts) else ""
            
            # --- INTELLIGENT CONTEXT UPDATE ---
            major_keywords = ["CHAPTER", "SECTION", "PART", "ARTICLE", "APPENDIX", "SCHEDULE", "PARAGRAPH"]
            is_major_section = any(header_marker.upper().startswith(k) for k in major_keywords)
            
            if is_major_section:
                first_line_title = content.split('\n')[0].strip()[:100]
                # Reset local definition context if we hit a major new section
                local_definition_context = "" 
                section_tracker['current_header'] = f"{header_marker} {first_line_title}".strip()
            
            # Combine Marker + Content
            full_clause = header_marker + " " + content
            
            # --- INJECT LOCAL DEFINITION (Sticky Header) ---
            # If we have a saved definition header, prepend it to this clause.
            final_text_content = full_clause
            if local_definition_context:
                final_text_content = f"{local_definition_context}\n   -> {full_clause}"

            # Construct Metadata Context
            current_context = (
                f"{file_context_str} > "
                f"Section: {section_tracker['current_header']} > "
                f"Clause: {header_marker}" 
            )
            
            # Create Document object(s)
            chunks.extend(self._create_chunks(final_text_content, current_context, page_num, base_metadata, "legal_clause"))

        return chunks

    def _create_chunks(self, text: str, context: str, page: int, meta: Dict, method: str) -> List[Document]:
        """
        Helper to safely wrap text into Documents, handling size limits.
        """
        text = text.strip()
        if not text: return []
        
        # If the clause is massive (larger than chunk_size), split it recursively
        if len(text) > self.chunk_size:
            sub_docs = self.recursive_chunker.create_documents([text])
            docs = []
            for i, d in enumerate(sub_docs):
                docs.append(Document(
                    page_content=f"{context} (Part {i+1})\n{d.page_content}",
                    metadata={**meta, 'page': page, 'chunk_method': f"{method}_split"}
                ))
            return docs
        else:
            return [Document(
                page_content=f"{context}\n{text}",
                metadata={**meta, 'page': page, 'chunk_method': method}
            )]

    def _chunk_tables(self, tables: List[Dict], page_num: int, base_metadata: Dict, file_context_str: str) -> List[Document]:
        """Convert JSON tables to Markdown chunks."""
        chunks = []
        for table_idx, table in enumerate(tables):
            md_text = self._table_to_markdown(table)
            if len(md_text) < 10: continue
            
            final_content = f"{file_context_str} > Table Data:\n{md_text}"
            
            chunks.append(Document(
                page_content=final_content,
                metadata={
                    **base_metadata,
                    'page': page_num,
                    'table_index': table_idx,
                    'content_type': 'table',
                    'chunk_method': 'table_markdown'
                }
            ))
        return chunks

    def _table_to_markdown(self, table: Dict) -> str:
        try:
            headers = table.get('headers', [])
            data = table.get('data', [])
            if not headers and not data: return ""
            lines = []
            if headers:
                lines.append("| " + " | ".join(str(h).replace('\n', ' ') for h in headers) + " |")
                lines.append("| " + " | ".join(['---'] * len(headers)) + " |")
            for row in data:
                if isinstance(row, dict):
                    vals = [str(row.get(h, '')).replace('\n', ' ') for h in headers]
                elif isinstance(row, list):
                    vals = [str(v).replace('\n', ' ') for v in row]
                else: continue
                lines.append("| " + " | ".join(vals) + " |")
            return "\n".join(lines)
        except Exception:
            return ""

    def _chunk_semantically(self, text: str, page_num: int, base_metadata: Dict, context_prefix: str) -> List[Document]:
        """Fallback for unstructured text."""
        try:
            docs = self.semantic_chunker.create_documents([text])
            for d in docs:
                d.page_content = context_prefix + d.page_content
                d.metadata.update({**base_metadata, 'page': page_num, 'chunk_method': 'semantic'})
            return docs
        except Exception:
            return self._chunk_recursively(text, page_num, base_metadata, context_prefix)

    def _chunk_recursively(self, text: str, page_num: int, base_metadata: Dict, context_prefix: str) -> List[Document]:
        """Ultimate fallback."""
        docs = self.recursive_chunker.create_documents([text])
        for d in docs:
            d.page_content = context_prefix + d.page_content
            d.metadata.update({**base_metadata, 'page': page_num, 'chunk_method': 'recursive'})
        return docs

    def _generate_embeddings_batched(self, chunks: List[Document], batch_size=50):
        """
        Batches embedding generation to avoid timeouts/payload errors.
        """
        if not chunks: return
        
        total = len(chunks)
        logger.info(f"[HybridChunker] Embedding {total} chunks in batches of {batch_size}...")
        
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.page_content for c in batch]
            try:
                embeddings = self.embeddings.embed_documents(texts)
                for j, emb in enumerate(embeddings):
                    batch[j].metadata['dense_embedding'] = emb
            except Exception as e:
                logger.error(f"Error embedding batch {i}: {e}")
    def get_chunk_statistics(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Returns basic statistics about the chunks generated.
        Used by the pipeline logger.
        """
        if not chunks:
            return {"total_chunks": 0, "avg_length": 0}
        
        lengths = [len(c.page_content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_length": sum(lengths) / len(chunks),
            "min_length": min(lengths),
            "max_length": max(lengths)
        }            

def get_hybrid_chunker(embeddings):
    return HybridChunker(embeddings)