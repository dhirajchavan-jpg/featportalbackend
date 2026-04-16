import logging
import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import google.generativeai as genai
from app.config import settings

logger = logging.getLogger(__name__)


class GeminiJudgeEvaluator:
    """Use Gemini as a judge to evaluate RAG responses against retrieved chunks."""
    
    def __init__(self, api_key: str = None):
        """Initialize Gemini judge."""
        self.api_key = api_key or settings.GEMINI_API_KEY
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        
    async def evaluate_response(
        self,
        query: str,
        response: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        logger.info(f"[GEMINI_JUDGE] Evaluating response for query: {query[:50]}...")
        
        # Build context from retrieved chunks
        context = self._format_retrieved_chunks(retrieved_chunks)
        
        # ---------------------------------------------------------
        # CHANGED: Combined Prompt with YOUR EXACT TOXICITY DEFINITIONS
        # ---------------------------------------------------------
        #         
        prompt = f"""You are an expert evaluator for a RAG system. 
Your task is to evaluate the interaction based on the CONTEXT, USER QUESTION, and ANSWER.

CONTEXT (Retrieved Chunks):
{context}

USER QUESTION (Evaluate this for Query Toxicity):
{query}

ANSWER TO EVALUATE (Evaluate this for Quality & Toxicity):
{response}

EVALUATION METRICS:

1. Faithfulness (0-100): Is the answer grounded in the context? (100=Fully grounded)
2. Relevance (0-100): Does the answer address the user's question? (100=Direct answer)
3. Completeness (0-100): Does the answer address all parts of the question? (100=Complete)
4. Hallucination (0-100): Does the answer contain unsupported info? (0=No hallucinations)

5. Query Toxicity (0-100): 
   - **Target:** The USER QUESTION.
   - **Definition:** Harmful, abusive, hateful, profanity, threats.
   - 0 = Safe, 100 = Toxic.

6. Response Toxicity (0-100):
   - **Target:** The AI ANSWER.
   - **Definition:** Harmful, abusive, hateful, profanity, threats.
   - 0 = Safe/Professional, 100 = Toxic.

Respond in strict JSON format:
{{
  "faithfulness": {{ "score": <0-100>, "explanation": "<string>" }},
  "relevance": {{ "score": <0-100>, "explanation": "<string>" }},
  "completeness": {{ "score": <0-100>, "explanation": "<string>" }},
  "hallucination": {{ "score": <0-100>, "explanation": "<string>" }},
  "query_toxicity": {{ "score": <0-100>, "explanation": "<string>" }},
  "response_toxicity": {{ "score": <0-100>, "explanation": "<string>" }}
}}
"""

        # Initialize evaluation dict
        evaluation = {
            "query": query,
            "response_length": len(response),
            "num_chunks_used": len(retrieved_chunks),
            "evaluated_at": datetime.utcnow().isoformat()
        }
        
        try:
            # Single API Call
            result_text = await self._call_gemini(prompt)
            results = self._parse_json_response(result_text)
            
            # 1. Parse Standard Quality Metrics
            score_names = ["faithfulness", "relevance", "completeness", "hallucination"]
            for name in score_names:
                metric_data = results.get(name, {})
                evaluation[f"{name}_score"] = metric_data.get("score")
                evaluation[f"{name}_explanation"] = metric_data.get("explanation")

            # 2. Parse Query Toxicity (New Field)
            q_tox = results.get("query_toxicity", {})
            evaluation["query_toxicity_score"] = q_tox.get("score")
            evaluation["query_toxicity_explanation"] = q_tox.get("explanation")
            evaluation["query_toxicity_categories"] = q_tox.get("categories", [])

            # 3. Parse Response Toxicity (New Field)
            r_tox = results.get("response_toxicity", {})
            evaluation["response_toxicity_score"] = r_tox.get("score")
            evaluation["response_toxicity_explanation"] = r_tox.get("explanation")
                
             

        except Exception as e:
            logger.error(f"[GEMINI_JUDGE] Evaluation failed: {e}")
            score_names = ["faithfulness", "relevance", "completeness", "hallucination", "toxicity"]
            for name in score_names:
                evaluation[f"{name}_score"] = None
                evaluation[f"{name}_explanation"] = f"Error: {str(e)}"
        
        # ... (rest of the calculation logic remains the same) ...
        
        # Calculate overall score
        valid_scores = []
        
        # Positive metrics (Higher is better)
        for name in ["faithfulness", "relevance", "completeness"]:
            val = evaluation.get(f"{name}_score")
            if val is not None:
                valid_scores.append(val)
        
        # Negative metrics (Lower is better) -> Invert for average
        # We include Hallucination AND Response Toxicity in the AI quality score.
        # (We usually do NOT include Query Toxicity, as that is user behavior, not AI performance)
        
        if evaluation.get("hallucination_score") is not None:
            valid_scores.append(100 - evaluation["hallucination_score"])
            
        if evaluation.get("response_toxicity_score") is not None:
            valid_scores.append(100 - evaluation["response_toxicity_score"])

        evaluation["overall_score"] = sum(valid_scores) / len(valid_scores) if valid_scores else None
        
        return evaluation
    
    def _format_retrieved_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string."""
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get('page_content') or chunk.get('content', '')
            metadata = chunk.get('metadata', {})
            file_name = metadata.get('file_name', 'Unknown')
            page = metadata.get('page_number', '?')
            
            formatted.append(f"[Chunk {i}] Source: {file_name}, Page: {page}\n{content}")
        
        return "\n\n".join(formatted)
    
    async def _call_gemini(self, prompt: str, max_retries: int = 3) -> str:
        """Call Gemini API with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"[GEMINI_JUDGE] Failed after {max_retries} attempts: {e}")
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"[GEMINI_JUDGE] Retry {attempt + 1} after {wait_time}s...")
                await asyncio.sleep(wait_time)
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from Gemini."""
        
        try:
            # Clean response
            cleaned = response_text.strip()
            
            # Extract JSON from markdown blocks
            if "```json" in cleaned:
                json_start = cleaned.find("```json") + 7
                json_end = cleaned.find("```", json_start)
                if json_end > json_start:
                    cleaned = cleaned[json_start:json_end].strip()
            elif "```" in cleaned:
                json_start = cleaned.find("```") + 3
                json_end = cleaned.find("```", json_start)
                if json_end > json_start:
                    cleaned = cleaned[json_start:json_end].strip()
            
            # Find JSON object
            if '{' in cleaned and '}' in cleaned:
                start = cleaned.find('{')
                end = cleaned.rfind('}') + 1
                cleaned = cleaned[start:end]
            
            result = json.loads(cleaned)
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"[GEMINI_JUDGE] JSON parse error: {e}")
            logger.debug(f"[GEMINI_JUDGE] Failed text: {response_text[:200]}")
            # Return empty dict so keys are missed rather than crashing
            return {} 
        except Exception as e:
            logger.error(f"[GEMINI_JUDGE] Unexpected error: {e}")
            return {}


# ============================================================================
# RETRIEVAL QUALITY EVALUATOR (Unchanged)
# ============================================================================

class RetrievalQualityEvaluator:
    """Evaluate retrieval quality using Gemini as judge."""
    
    def __init__(self, api_key: str = None):
        """Initialize evaluator."""
        self.api_key = api_key or settings.GEMINI_API_KEY
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
    
    async def evaluate_retrieved_chunks(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate if the retrieved chunks are relevant to the query."""
        logger.info(f"[RETRIEVAL_JUDGE] Evaluating {len(retrieved_chunks)} retrieved chunks...")
        
        # Format chunks
        chunks_text = "\n\n".join([
            f"[Chunk {i+1}]\n{chunk.get('page_content') or chunk.get('content', '')}"
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        prompt = f"""You are an expert evaluator. Assess the quality of retrieved chunks for answering the question.

USER QUESTION:
{query}

RETRIEVED CHUNKS:
{chunks_text}

TASK:
Evaluate each chunk's relevance to the question on a scale of 0-100:
- 100: Directly answers the question
- 75: Highly relevant context
- 50: Somewhat relevant
- 25: Marginally relevant
- 0: Irrelevant

Then calculate overall retrieval quality.

Respond in JSON format:
{{
  "overall_score": <0-100>,
  "chunk_scores": [<score1>, <score2>, ...],
  "num_relevant_chunks": <count of chunks with score >= 50>,
  "explanation": "<brief assessment>"
}}
"""
        
        try:
            result = await self._call_gemini(prompt)
            parsed = self._parse_json_response(result)
            
            parsed["query"] = query
            parsed["num_chunks_evaluated"] = len(retrieved_chunks)
            parsed["evaluated_at"] = datetime.utcnow().isoformat()
            
            return parsed
            
        except Exception as e:
            logger.error(f"[RETRIEVAL_JUDGE] Evaluation failed: {e}")
            return {
                "overall_score": None,
                "error": str(e),
                "query": query,
                "num_chunks_evaluated": len(retrieved_chunks)
            }
    
    async def _call_gemini(self, prompt: str, max_retries: int = 3) -> str:
        """Call Gemini API with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from Gemini."""
        try:
            cleaned = response_text.strip()
            
            if "```json" in cleaned:
                json_start = cleaned.find("```json") + 7
                json_end = cleaned.find("```", json_start)
                if json_end > json_start:
                    cleaned = cleaned[json_start:json_end].strip()
            elif "```" in cleaned:
                json_start = cleaned.find("```") + 3
                json_end = cleaned.find("```", json_start)
                if json_end > json_start:
                    cleaned = cleaned[json_start:json_end].strip()
            
            if '{' in cleaned and '}' in cleaned:
                start = cleaned.find('{')
                end = cleaned.rfind('}') + 1
                cleaned = cleaned[start:end]
            
            return json.loads(cleaned)
            
        except json.JSONDecodeError as e:
            logger.error(f"[RETRIEVAL_JUDGE] JSON parse error: {e}")
            return {"overall_score": None, "error": f"Parse error: {str(e)}"}


# ============================================================================
# INTEGRATION WITH EXISTING EVALUATION SYSTEM
# ============================================================================

def get_gemini_judge_evaluator() -> GeminiJudgeEvaluator:
    """Get singleton instance of Gemini judge evaluator."""
    if not hasattr(get_gemini_judge_evaluator, "_instance"):
        get_gemini_judge_evaluator._instance = GeminiJudgeEvaluator()
    return get_gemini_judge_evaluator._instance


def get_retrieval_judge_evaluator() -> RetrievalQualityEvaluator:
    """Get singleton instance of retrieval judge evaluator."""
    if not hasattr(get_retrieval_judge_evaluator, "_instance"):
        get_retrieval_judge_evaluator._instance = RetrievalQualityEvaluator()
    return get_retrieval_judge_evaluator._instance