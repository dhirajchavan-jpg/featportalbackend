"""
Layer 4: Model Router
Routes queries to appropriate LLM based on complexity.
"""

from typing import Dict, Any, Optional
from langchain_ollama import OllamaLLM as Ollama
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
        logger.info(f" [ROUTER CONFIG] Default System Router: {settings.ROUTER_MODEL}")

        try:
            logger.info("[ModelRouter] Connecting to Default Router LLM...")
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
        router_model_name: Optional[str] = None,
    ) -> str:
        """
        Determine complexity: 'simple' or 'complex'.
        Supports dynamic model switching per project.
        """
        client = self.default_client
        current_model_name = settings.ROUTER_MODEL

        if router_model_name and router_model_name != settings.ROUTER_MODEL:
            try:
                logger.info(f"[ModelRouter] PROJECT OVERRIDE: Swapping {settings.ROUTER_MODEL} -> {router_model_name}")
                client = Ollama(
                    base_url=settings.OLLAMA_BASE_URL,
                    model=router_model_name,
                    temperature=0,
                )
                current_model_name = router_model_name
            except Exception as e:
                logger.error(f"[ModelRouter] Failed to load custom model {router_model_name}: {e}")
                client = self.default_client

        if not client:
            logger.warning("[ModelRouter] No router client available. Defaulting to 'complex'.")
            return "complex"

        logger.info("[ModelRouter] Step 1: Checking Hard Rules for COMPLEXITY...")
        complex_reason = self._get_complex_rule_reason(query, query_metadata)
        if complex_reason:
            logger.info(f"[ModelRouter] Decision: COMPLEX (Rule Match: {complex_reason}). Model {current_model_name} was NOT called.")
            return "complex"

        logger.info("[ModelRouter] Step 2: Checking Hard Rules for SIMPLICITY...")
        simple_reason = self._get_simple_rule_reason(query, query_metadata)
        if simple_reason:
            logger.info(f"[ModelRouter] Decision: SIMPLE (Rule Match: {simple_reason}). Model {current_model_name} was NOT called.")
            return "simple"

        logger.info("[ModelRouter] Step 3: No hard rules matched. Delegating to AI Judge.")

        print(f"\n{'=' * 40}")
        print(f" STARTING ROUTER LLM: {current_model_name}")
        print(f"{'=' * 40}\n")

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

SIMPLE: Factual questions, definitions, simple comparisons (e.g. \"Compare rate A and B\"), listing items.
COMPLEX: Multi-step reasoning, deep analysis, explaining cause-and-effect, synthesizing multiple abstract concepts.

Query: {query}

Answer with ONLY one word - \"simple\" or \"complex\":"""

            print(f"\n>>> [ModelRouter] SENDING PROMPT TO: {model_name} ...")
            logger.info("[ModelRouter] Invoking Router Model...")
            response = await client.ainvoke(prompt)

            if hasattr(response, "content"):
                decision = response.content.strip().lower()
            else:
                decision = str(response).strip().lower()

            print(f">>> [ModelRouter] RESPONSE FROM {model_name}: {decision}\n")
            logger.info(f"[ModelRouter] Router Model raw response: '{decision}'")

            if "complex" in decision:
                return "complex"
            if "simple" in decision:
                return "simple"

            logger.warning(f"[ModelRouter] Ambiguous AI response '{decision}'. Defaulting to 'complex'.")
            return "complex"
        except Exception as e:
            logger.error(f"[ModelRouter] AI routing failed: {e}")
            return "complex"

    def _get_complex_rule_reason(self, query: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Detect fundamentally hard queries and HR policy synthesis requests."""
        query_lower = query.lower().strip()
        token_count = metadata.get("token_count", len(query.split()))

        if token_count > 25:
            return f"long query ({token_count} words)"

        if query.count("?") > 1:
            return "multiple questions"

        deep_analysis_keywords = [
            "implications",
            "impact analysis",
            "relationship between",
            "synthesize",
            "comprehensive",
            "detailed breakdown",
            "evaluate",
        ]
        for keyword in deep_analysis_keywords:
            if keyword in query_lower:
                return f"keyword '{keyword}'"

        hr_policy_topics = [
            "leave policy",
            "attendance policy",
            "probation",
            "benefits",
            "notice period",
            "final settlement",
            "grievance",
            "disciplinary",
            "misconduct",
            "travel policy",
            "reimbursement",
            "working hours",
            "code of conduct",
            "remote work",
        ]
        synthesis_phrases = [
            "key points",
            "key point",
            "what are the rules",
            "summarize",
            "summary",
            "tell me about",
            "what does",
            "policy on",
            "how does",
        ]

        if any(topic in query_lower for topic in hr_policy_topics):
            if any(phrase in query_lower for phrase in synthesis_phrases):
                return "HR policy synthesis query"
            if query_lower.startswith(("what are", "what is", "explain", "summarize", "list")):
                return "HR policy extraction query"

        return None

    def _get_simple_rule_reason(self, query: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Detect queries that are definitely easy."""
        query_lower = query.lower().strip()
        token_count = metadata.get("token_count", len(query.split()))

        if token_count < 8:
            return f"short query ({token_count} words)"

        start_phrases = ("what is", "define", "who is", "when was", "list")
        if query_lower.startswith(start_phrases):
            return "definition/fact phrase"

        return None

    def _is_complex_by_rules(self, query: str, metadata: Dict[str, Any]) -> bool:
        return self._get_complex_rule_reason(query, metadata) is not None

    def _is_simple_by_rules(self, query: str, metadata: Dict[str, Any]) -> bool:
        return self._get_simple_rule_reason(query, metadata) is not None

    def get_model(self, complexity: str):
        if complexity == "complex":
            return self.complex_model
        return self.simple_model

    def get_model_info(self, complexity: str) -> Dict[str, str]:
        if complexity == "complex":
            return {"model_name": settings.LLM_MODEL_COMPLEX}
        return {"model_name": settings.LLM_MODEL_SIMPLE}


_model_router = None


def get_model_router() -> ModelRouter:
    global _model_router
    if _model_router is None:
        logger.info("[ModelRouter] Creating new ModelRouter singleton instance.")
        _model_router = ModelRouter()
    return _model_router
