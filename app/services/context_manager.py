# app/services/context_manager.py
"""
Layer 4: Context Manager
Assembles final context for the LLM from retrieved documents and chat history.
Enforces strict citation formatting: [File: Name, Page: N].
"""

import os
from typing import List, Dict, Any, Optional
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Manages context assembly for LLM generation.
    Loads system prompts from the 'prompts/' directory defined in settings.
    Returns structured message arrays for modern chat models.
    """
    
    def __init__(self):
        logger.info("[ContextManager] Initializing ContextManager...")
        self.max_tokens = settings.MAX_CONTEXT_TOKENS
        self.enable_deduplication = settings.ENABLE_DEDUPLICATION
        self.default_history_turns = 5  # Default fallback if not provided in config
        
        # --- Load Prompts from Files ---
        self.prompts_dir = settings.PROMPTS_DIR
        logger.info(f"[ContextManager] Loading prompts from {self.prompts_dir}...")
        
        # Load Complexity-Based Prompts
        self.prompt_simple = self._load_prompt_file("system_simple.txt", default="You are a helpful assistant.")
        self.prompt_complex = self._load_prompt_file("system_complex.txt", default="You are a senior analyst.")

        # --- UPDATED STYLE OVERLAYS (Anti-Hallucination for Detailed Mode) ---
        self.style_overlays = {
    "Simple": (
        "\n\n### RESPONSE MODE: SIMPLE\n"
        "STYLE OVERRIDE RULE:\n"
        "If any formatting, length, or structure instructions conflict with these STYLE INSTRUCTIONS, "
        "the STYLE INSTRUCTIONS MUST BE FOLLOWED.\n\n"
        "You must answer in SIMPLE mode.\n\n"
        "STRICT RULES:\n"
        "- Use plain HR language suitable for employees, managers, and HR coordinators.\n"
        "- Use ONE short paragraph OR up to TWO short sentences only.\n"
        "- Focus only on the direct HR answer such as entitlement, rule, timeline, approval, or next step.\n- Avoid policy jargon unless it appears in the source text.\n"
        "- Do NOT explain background, reasoning, or implications.\n"
        "- Do NOT add examples, commentary, or extra advice.\n"
        "- If the answer is not explicitly stated in the context, respond with: "
        "'The provided documents do not contain this information.'\n\n"
        "OUTPUT FORMAT:\n"
        "- A single concise answer.\n"
        "- No bullet points.\n"
        "- No headings.\n"
    ),
    "Formal": (
        "\n\n### RESPONSE MODE: FORMAL\n"
        "STYLE OVERRIDE RULE:\n"
        "If any formatting, length, or structure instructions conflict with these STYLE INSTRUCTIONS, "
        "the STYLE INSTRUCTIONS MUST BE FOLLOWED.\n\n"
        "You must answer in FORMAL HR policy documentation style.\n\n"
        "STRICT RULES:\n"
        "- Use professional HR policy and employee governance language.\n"
        "- Maintain a neutral, objective, and authoritative tone.\n"
        "- Base the response strictly on the retrieved context.\n"
        "- Emphasize policy rule, applicability, condition, timeline, approval, and exception where available.\n- Do NOT include assumptions, interpretations, or recommendations.\n"
        "- Do NOT restate the question.\n"
        "- Do NOT exceed FIVE bullet points.\n\n"
        "OUTPUT FORMAT:\n"
        "- Use bullet points only.\n"
        "- Each bullet must be a complete, precise statement.\n"
        "- Avoid explanatory or narrative paragraphs.\n"
    ),
    "Detailed": (
        "\n\n### RESPONSE MODE: DETAILED\n"
        "STYLE OVERRIDE RULE:\n"
        "If any formatting, length, or structure instructions conflict with these STYLE INSTRUCTIONS, "
        "the STYLE INSTRUCTIONS MUST BE FOLLOWED.\n\n"
        "You must answer in DETAILED HR analytical mode.\n\n"
        "STRICT RULES:\n"
        "- Use only the information explicitly present in the provided documents.\n"
        "- EVERY bullet point MUST include at least one citation in the format: [File: filename.pdf, Page: X].\n"
        "- Do NOT infer, interpret, or introduce external knowledge.\n"
        "- Cover the HR rule thoroughly by including relevant eligibility, timelines, approvals, limits, exceptions, or process steps only when they are supported by the cited text.\n- If the HR excerpt is partial or truncated, summarize the clearly visible policy points and state that the excerpt appears incomplete when needed.\n- Detailed responses must NOT include interpretive sentences beyond the cited HR text.\n"
        "- If a detail cannot be cited, it MUST NOT be included.\n"
        "- Expand each point ONLY by restating or clarifying cited text.\n"
        "- Do NOT exceed FIVE bullet points.\n\n"
        "OUTPUT FORMAT:\n"
        "- Use structured bullet points.\n"
        "- Each bullet may include a short explanatory paragraph WITH citations.\n"
        "- Avoid redundancy and irrelevant detail.\n"
    )
}
        logger.info(f"[ContextManager] Initialization complete.")
    
    def _load_prompt_file(self, filename: str, default: str) -> str:
        """Helper to read prompt text files safely."""
        try:
            file_path = os.path.join(self.prompts_dir, filename)
            logger.info(f"[ContextManager] Loading prompt file: {filename}")
            if not os.path.exists(file_path):
                logger.warning(f"[ContextManager] Prompt file not found: {filename}. Using default.")
                return default
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                logger.info(f"[ContextManager] Successfully loaded {filename} ({len(content)} chars).")
                return content
        except Exception as e:
            logger.error(f"[ContextManager] Failed to load prompt {filename}: {e}")
            return default

    def build_context(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        chat_history: List[Dict[str, Any]] = None,
        current_sector: Optional[str] = None,
        include_history: bool = True,
        chat_history_limit: int = 5  # <--- NEW: Dynamic Limit (Default 5)
    ) -> str:
        """
        Build final context for Standard (non-comparative) LLM generation.
        """
        logger.info(f"[ContextManager] Building standard context for query: '{query[:50]}...' (History Limit: {chat_history_limit})")
        context_parts = []
        
        # 1. Add Chat History (Sector-Filtered)
        if include_history:
            logger.info(f"[ContextManager] Processing chat history (total turns: {len(chat_history) if chat_history else 0})...")
            if chat_history and len(chat_history) > 0:
                filtered_history = self._filter_history_by_sector(chat_history, current_sector)
                logger.info(f"[ContextManager] History filtered by sector '{current_sector}': {len(filtered_history)} turns remaining.")
                
                if filtered_history:
                    # Pass the dynamic limit here
                    history_context = self._format_chat_history(filtered_history, limit=chat_history_limit)
                    if history_context:
                        context_parts.append("## Previous Conversation\n")
                        context_parts.append("(Use this to understand follow-up questions and maintain context)\n\n")
                        context_parts.append(history_context)
                        context_parts.append("\n---\n\n")
            else:
                logger.info("[ContextManager] No chat history to add.")
        else:
            logger.info("[ContextManager] Chat history inclusion disabled.")
        
        # 2. Add Retrieved Documents with CITATIONS
        logger.info(f"[ContextManager] Formatting {len(retrieved_docs)} retrieved documents...")
        if retrieved_docs:
            docs_context = self._format_retrieved_docs(retrieved_docs)
            if docs_context:
                context_parts.append("## Relevant Information\n")
                context_parts.append(docs_context)
        else:
            logger.warning("[ContextManager] No retrieved documents provided.")
        
        # 3. Combine & Deduplicate
        full_context = "".join(context_parts)
        initial_len = len(full_context)
        logger.info(f"[ContextManager] Initial context length: {initial_len} chars.")
        
        if self.enable_deduplication:
            logger.info("[ContextManager] Running deduplication...")
            full_context = self._deduplicate_content(full_context)
            logger.info(f"[ContextManager] Deduplication complete. Size reduced from {initial_len} to {len(full_context)} chars.")
            
        # 4. Truncate
        logger.info("[ContextManager] Checking for truncation...")
        full_context = self._smart_truncate_context(full_context)
            
        return full_context
    
    def build_comparative_context(
        self,
        query: str,
        sector_results: List[Dict[str, Any]],
        chat_history: List[Dict[str, Any]] = None,
        include_history: bool = True,
        chat_history_limit: int = 5  # <--- NEW: Dynamic Limit (Default 5)
    ) -> str:
        """
        Build context for Comparative Analysis queries.
        Groups documents by Sector. SKIPS sectors with no data.
        """
        logger.info(f"[ContextManager] Building comparative context for query: '{query[:50]}...' (History Limit: {chat_history_limit})")
        context_parts = []
        
        # FIXED: Always include history if available for follow-up context
        if include_history:
            logger.info(f"[ContextManager] Processing history for comparative context ({len(chat_history) if chat_history else 0} turns)...")
            if chat_history and len(chat_history) > 0:
                # Use dynamic limit
                history_context = self._format_chat_history(chat_history, limit=chat_history_limit)
                
                if history_context:
                    context_parts.append("## Recent Conversation History\n")
                    context_parts.append("(Use this context to understand follow-up questions like 'tell me more', 'explain this', etc.)\n\n")
                    context_parts.append(history_context)
                    context_parts.append("\n\n---\n\n")
                    
                    logger.info(f"[ContextManager] Included history context (max {chat_history_limit} turns)")
        else:
            logger.info("[ContextManager] History disabled for comparative context.")
        
        context_parts.append("# Cross-Sector Regulatory Context\n\n")
        
        relevant_sectors_count = 0
        logger.info(f"[ContextManager] Processing results for {len(sector_results)} sectors...")
        
        for sector_data in sector_results:
            chunks = sector_data.get('chunks', [])
            sector = sector_data['sector']
            
            # Skip empty sectors
            if not chunks:
                logger.info(f"[ContextManager] Skipping sector '{sector}' (no chunks found).")
                continue
            
            relevant_sectors_count += 1
            logger.info(f"[ContextManager] Adding {len(chunks)} chunks for sector '{sector}'.")
            
            context_parts.append(f"## {sector} Sector\n\n")
            
            for chunk in chunks:
                formatted_chunk = self._format_doc(chunk)
                context_parts.append(formatted_chunk + "\n\n")
        
        # Handle case where ALL sectors came back empty
        if relevant_sectors_count == 0:
            logger.warning("[ContextManager] No relevant info found in any sector.")
            # FIXED: Even with no new docs, if there's history, still return it
            if chat_history and len(chat_history) > 0:
                full_context = "".join(context_parts)
                logger.info("[ContextManager] No new documents found, but returning with conversation history")
                return full_context
            else:
                return "No relevant information found in any sector regarding this query."
            
        full_context = "".join(context_parts)
        
        # Allow more tokens for comparative context
        logger.info("[ContextManager] Applying smart truncation (multiplier=5 for comparative)...")
        full_context = self._smart_truncate_context(full_context, multiplier=5)
        
        logger.info(f"[ContextManager] Built comparative context: {len(full_context)} chars, {relevant_sectors_count} relevant sectors")
        
        return full_context
    
    def prepare_prompt(
        self,
        query: str,
        context: str,
        complexity: str = "complex",
        system_message: str = None,
        style: str = "Detailed",
        is_comparative: bool = False # Deprecated but kept for signature compatibility
    ) -> List[Dict[str, str]]:
        """
        Prepare final prompt as a LIST OF MESSAGES (not a single string).
        Returns a messages array compatible with ChatOpenAI/ChatAnthropic.
        """
        logger.info(f"[ContextManager] Preparing final prompt messages (complexity={complexity})...")
        
        # 1. Determine System Instruction
        if system_message:
            logger.info("[ContextManager] Using custom system message.")
            system_instruction = system_message
        elif complexity == "simple":
            logger.info("[ContextManager] Using SIMPLE system prompt.")
            system_instruction = self.prompt_simple
        else:
            logger.info("[ContextManager] Using COMPLEX system prompt.")
            system_instruction = self.prompt_complex

        # Apply Style Overlay
        if style and hasattr(self, 'style_overlays') and style in self.style_overlays:
            # --- LOG 1: WHICH STYLE IS BEING APPLIED ---
            logger.info(f"[ContextManager] Applying Style Overlay: '{style}'")
            # -------------------------------------------
            system_instruction += self.style_overlays[style]
        else:
            logger.info(f"[ContextManager] No valid style found for '{style}'. Using base prompt only.")

        # 2. Build Messages Array
        messages = [
            {
                "role": "system",
                "content": system_instruction
            },
            {
                "role": "user",
                "content": f"=== CONTEXT DOCUMENTS ===\n{context}"
            },
            {
                "role": "user", 
                "content": f"=== USER QUESTION ===\n{query}"
            }
        ]
        
        # --- LOG 2: WHAT IS BEING SENT TO LLM ---
        logger.info(f"[ContextManager] SENDING TO LLM:")
        logger.info(f"   > System Prompt (First 300 chars): {system_instruction[:300]}...")
        logger.info(f"   > System Prompt (Last 100 chars): ...{system_instruction[-100:]}") # Shows the style instructions
        logger.info(f"   > User Query: {query}")
        # ----------------------------------------

        logger.info(f"[ContextManager] Constructed {len(messages)} messages for LLM.")
        return messages

    def _format_doc(self, doc: Dict[str, Any], index: int = None) -> str:
        """
        Helper to format a single document chunk with explicit metadata citations.
        Output: SOURCE: [File: X, Page: Y]
        """
        content = doc.get('page_content') or doc.get('content') or ""
        content = content.strip()
        
        meta = doc.get('metadata', {})
        
        # --- FIXED: Robust Fallback for Metadata ---
        file_name = meta.get('file_name') or 'Unknown File'
        page_num = meta.get('page_number') or meta.get('page') or '?'
        
        header = f"SOURCE: [File: {file_name}, Page: {page_num}]"
        
        if index:
            header = f"[{index}] {header}"
            
        return f"{header}\n{content}"

    def _format_retrieved_docs(self, docs: List[Dict[str, Any]]) -> str:
        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(self._format_doc(doc, index=i))
        return "\n\n".join(formatted)

    def _filter_history_by_sector(self, history, current_sector):
        if not history or not current_sector: return history
        norm_sector = current_sector.lower().strip()
        filtered = []
        for h in history:
            h_sector = h.get('sector')
            if not h_sector or h_sector.lower().strip() == norm_sector:
                filtered.append(h)
        return filtered

    def _format_chat_history(self, history: List[Dict[str, Any]], limit: int = 5) -> str:
        """
        Format chat history for inclusion in context.
        Uses the provided limit to slice history.
        """
        if not history:
            return ""
        
        formatted = []
        # --- UPDATED: Use dynamic limit instead of self.max_history_turns ---
        # Ensure limit is at least 1 if history exists, defaulting to 5 if None passed
        safe_limit = limit if limit is not None else self.default_history_turns
        
        recent = history[-safe_limit:]
        
        for i, h in enumerate(recent, 1):
            q = h.get('user_query', '').strip()
            a = h.get('llm_answer', '').strip()
            
            if q and a:
                # Truncate very long answers to save tokens
                if len(a) > 500:
                    a = a[:500] + "... [truncated]"
                
                formatted.append(f"**Turn {i}:**\nUser: {q}\nAssistant: {a}")
        
        if not formatted:
            return ""
        
        return "\n\n".join(formatted)

    def _score_doc(self, doc: Dict[str, Any]) -> float:
        if not isinstance(doc, dict):
            return 0.0
        for key in ("rerank_score", "relevance_score", "rrf_score", "dense_score", "sparse_score"):
            value = doc.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    def _deduplicate_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not docs:
            return []

        unique_docs: Dict[str, Dict[str, Any]] = {}
        duplicate_count = 0

        for doc in docs:
            content = (doc.get("page_content") or doc.get("content") or "").strip()
            meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            dedup_key = "|".join([
                content,
                str(meta.get("file_name") or ""),
                str(meta.get("page_number") or meta.get("page") or ""),
            ])

            if dedup_key not in unique_docs:
                unique_docs[dedup_key] = doc
                continue

            duplicate_count += 1
            existing = unique_docs[dedup_key]
            if self._score_doc(doc) > self._score_doc(existing):
                unique_docs[dedup_key] = doc

        if duplicate_count:
            logger.info(
                f"[ContextManager] Removed {duplicate_count} duplicate retrieved chunks before prompt assembly."
            )

        return list(unique_docs.values())

    def _deduplicate_content(self, context: str) -> str:
        if not context: return ""
        paragraphs = context.split('\n\n')
        seen = set()
        unique_paragraphs = []
        for para in paragraphs:
            stripped = para.strip()
            if not stripped: continue
            # Always keep structural elements
            if stripped.startswith(("SOURCE:", "#", "Turn", "**Turn", "==")):
                unique_paragraphs.append(para)
                continue
            para_hash = hash(stripped)
            if para_hash not in seen:
                seen.add(para_hash)
                unique_paragraphs.append(para)
        return "\n\n".join(unique_paragraphs)

    def _smart_truncate_context(self, context: str, multiplier: int = 4) -> str:
        limit = self.max_tokens * multiplier
        if len(context) <= limit: return context
        
        logger.info(f"[ContextManager] Truncating context (Current: {len(context)}, Limit: {limit})")
        truncated = context[:limit]
        last_newline = truncated.rfind('\n')
        if last_newline > limit * 0.8:
            truncated = truncated[:last_newline]
        return truncated + "\n\n[...Context truncated...]"

    def estimate_tokens(self, text: str) -> int:
        return len(text.split())

    def get_context_stats(self, context: str) -> Dict[str, Any]:
        stats = {
            "length": len(context),
            "estimated_tokens": self.estimate_tokens(context),
            "limit": self.max_tokens
        }
        logger.info(f"[ContextManager] Context Stats: {stats}")
        return stats

# Singleton instance
_context_manager = None

def get_context_manager() -> ContextManager:
    global _context_manager
    if _context_manager is None:
        logger.info("[ContextManager] Creating new singleton instance.")
        _context_manager = ContextManager()
    else:
        logger.info("[ContextManager] Returning existing singleton instance.")
    return _context_manager


