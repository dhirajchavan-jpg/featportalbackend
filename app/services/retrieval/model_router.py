"""
Layer 4: Model Router
Routes queries to appropriate LLM based on complexity.
"""

from typing import Dict, Any, Optional
from langchain_ollama import OllamaLLM as Ollama # <--- Added Import
from app.config import settings
from app.core import llm_provider
import logging

logger = logging.getLogger(__name__)

class ModelRouter:
    """
    Routes queries to appropriate model based on complexity.
    Optimized for "Always Comparative" architecture.
    """
    
    def __init__(self):
        logger.info("[ModelRouter] Initializing ModelRouter...")
        
        # Log System Default
        logger.info(f" [ROUTER CONFIG] Default System Router: {settings.ROUTER_MODEL}")

        try:
            logger.info("[ModelRouter] Connecting to Default Router LLM...")
            # This is the FALLBACK/DEFAULT client (e.g., qwen2.5:0.5b)
            self.default_client = llm_provider.get_router_llm()
            
            logger.info("[ModelRouter] Connecting to Simple LLM...")
            self.simple_model = llm_provider.get_simple_llm()
            
            logger.info("[ModelRouter] Connecting to Complex LLM...")
            self.complex_model = llm_provider.get_complex_llm()
            
            logger.info("[ModelRouter] Successfully connected to all models.")
        except Exception as e:
            logger.error(f"[ModelRouter] Connection error during initialization: {e}")
            self.default_client = None
            self.simple_model = None
            self.complex_model = None
    
    async def route_query(
        self, 
        query: str, 
        query_metadata: Dict[str, Any],
        router_model_name: Optional[str] = None # <--- NEW ARGUMENT
    ) -> str:
        """
        Determine complexity: 'simple' or 'complex'.
        Supports dynamic model switching per project.
        """
        # 1. SETUP THE CLIENT (Custom vs Default)
        client = self.default_client
        current_model_name = settings.ROUTER_MODEL # For logging

        # Check if Project overrides the Router Model
        if router_model_name and router_model_name != settings.ROUTER_MODEL:
            try:
                logger.info(f"[ModelRouter] PROJECT OVERRIDE: Swapping {settings.ROUTER_MODEL} -> {router_model_name}")
                
                # Create a temporary lightweight client for this specific request
                client = Ollama(
                    base_url=settings.OLLAMA_BASE_URL, 
                    model=router_model_name, 
                    temperature=0
                )
                current_model_name = router_model_name
            except Exception as e:
                logger.error(f"[ModelRouter] Failed to load custom model {router_model_name}: {e}")
                # Fallback to default
                client = self.default_client

        if not client:
            logger.warning("[ModelRouter] No router client available. Defaulting to 'complex'.")
            return 'complex' 
            
        # 2. Check Hard Rules for COMPLEXITY (High Priority)
        # Note: If rules match, we DO NOT call the LLM to save time/cost.
        logger.info("[ModelRouter] Step 1: Checking Hard Rules for COMPLEXITY...")
        if self._is_complex_by_rules(query, query_metadata):
            logger.info(f"[ModelRouter] Decision: COMPLEX (Rule Match). Model {current_model_name} was NOT called.")
            return 'complex'
            
        # 3. Check Hard Rules for SIMPLICITY (Medium Priority)
        logger.info("[ModelRouter] Step 2: Checking Hard Rules for SIMPLICITY...")
        if self._is_simple_by_rules(query, query_metadata):
            logger.info(f"[ModelRouter] Decision: SIMPLE (Rule Match). Model {current_model_name} was NOT called.")
            return 'simple'
            
        # 4. AI Judge (The Router Model)
        # Only runs if no hard rules were matched.
        logger.info("[ModelRouter] Step 3: No hard rules matched. Delegating to AI Judge.")
        
        # --- Banner Log for Execution ---
        print(f"\n{'='*40}")
        print(f" STARTING ROUTER LLM: {current_model_name}")
        print(f"{'='*40}\n")
        # --------------------------------

        # Pass the specific client we selected (Default or Custom) AND the model name for logging
        decision = await self._route_with_model(query, client, model_name=current_model_name)
        
        logger.info(f"[ModelRouter] AI Judge Decision: {decision}")
        return decision
    
    async def _route_with_model(self, query: str, client: Any, model_name: str = "Unknown") -> str:
        """
        Ask the Router Model to judge difficulty.
        Accepts 'client' argument to support dynamic models.
        """
        logger.info("[ModelRouter] Preparing prompt for AI routing...")
        try:
            prompt = f"""Analyze the difficulty of this query.

SIMPLE: Factual questions, definitions, simple comparisons (e.g. "Compare rate A and B"), listing items.
COMPLEX: Multi-step reasoning, deep analysis, explaining cause-and-effect, synthesizing multiple abstract concepts.

Query: {query}

Answer with ONLY one word - "simple" or "complex":"""
            
            # --- NEW: Explicit Terminal Log ---
            print(f"\n>>> [ModelRouter] SENDING PROMPT TO: {model_name} ...")
            # ----------------------------------

            logger.info("[ModelRouter] Invoking Router Model...")
            
            # USE THE PASSED CLIENT, NOT SELF.ROUTER_MODEL
            response = await client.ainvoke(prompt)
            
            if hasattr(response, 'content'):
                decision = response.content.strip().lower()
            else:
                decision = str(response).strip().lower()
            
            # --- NEW: Log Response ---
            print(f">>> [ModelRouter] RESPONSE FROM {model_name}: {decision}\n")
            # -------------------------

            logger.info(f"[ModelRouter] Router Model raw response: '{decision}'")
            
            if 'complex' in decision:
                return 'complex'
            elif 'simple' in decision:
                return 'simple'
            else:
                logger.warning(f"[ModelRouter] Ambiguous AI response '{decision}'. Defaulting to 'complex'.")
                return 'complex'
                
        except Exception as e:
            logger.error(f"[ModelRouter] AI routing failed: {e}")
            return 'complex'

    def _is_complex_by_rules(self, query: str, metadata: Dict[str, Any]) -> bool:
        """Detect fundamentally hard queries."""
        query_lower = query.lower()
        token_count = metadata.get('token_count', len(query.split()))
        
        # Rule A: Long Queries
        if token_count > 25:
            logger.info(f"[ModelRouter] Complex Rule Matched: Long query ({token_count} words)")
            return True
            
        # Rule B: Multi-Part
        if query.count('?') > 1:
            logger.info("[ModelRouter] Complex Rule Matched: Multiple questions")
            return True
            
        # Rule C: Keywords
        deep_analysis_keywords = [
            'implications', 'impact analysis', 'relationship between', 
            'synthesize', 'comprehensive', 'detailed breakdown', 'evaluate'
        ]
        for keyword in deep_analysis_keywords:
            if keyword in query_lower:
                logger.info(f"[ModelRouter] Complex Rule Matched: Keyword '{keyword}'")
                return True
        return False
    
    def _is_simple_by_rules(self, query: str, metadata: Dict[str, Any]) -> bool:
        """Detect queries that are definitely easy."""
        query_lower = query.lower()
        token_count = metadata.get('token_count', len(query.split()))

        # Rule A: Short Queries
        if token_count < 8:
            logger.info(f"[ModelRouter] Simple Rule Matched: Short query ({token_count} words)")
            return True
            
        # Rule B: Definitions
        start_phrases = ('what is', 'define', 'who is', 'when was', 'list')
        if query_lower.startswith(start_phrases):
            logger.info(f"[ModelRouter] Simple Rule Matched: Definition/Fact phrase")
            return True
        return False

    def get_model(self, complexity: str):
        if complexity == 'complex':
            return self.complex_model
        return self.simple_model
    
    def get_model_info(self, complexity: str) -> Dict[str, str]:
        if complexity == 'complex':
            return {'model_name': settings.LLM_MODEL_COMPLEX}
        return {'model_name': settings.LLM_MODEL_SIMPLE}

# Singleton instance
_model_router = None

def get_model_router() -> ModelRouter:
    global _model_router
    if _model_router is None:
        logger.info("[ModelRouter] Creating new ModelRouter singleton instance.")
        _model_router = ModelRouter()
    return _model_router