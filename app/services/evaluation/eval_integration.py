"""
RAG Evaluation Integration - Connect Evaluation to Production Pipeline

This module integrates the evaluation system into your existing RAG service,
allowing for continuous monitoring using Gemini as a judge.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Import database collections directly
from app.database import db
from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# LIVE EVALUATION WRAPPER
# ============================================================================

class LiveEvaluationWrapper:
    """Wrapper to evaluate production queries in real-time using Gemini as judge."""
    
    def __init__(self, sample_rate: float = 1.0):
        """
        Initialize live evaluator with Gemini judge.
        
        Args:
            sample_rate: Fraction of queries to evaluate (0.0 to 1.0)
                        Default: 1.0 = 100% (evaluate every query)
        """
        self.sample_rate = sample_rate
        self.eval_collection = db["live_evaluations"]
        
        # Use Gemini as judge instead of Phoenix evaluators
        from app.services.evaluation.gemini_judge_evaluator import (
            get_gemini_judge_evaluator,
            get_retrieval_judge_evaluator
        )
        self.response_judge = get_gemini_judge_evaluator()
        self.retrieval_judge = get_retrieval_judge_evaluator()
    
    async def evaluate_query_result(
        self,
        query: str,
        rag_result: Dict[str, Any],
        user_id: str,
        project_id: str,
        span_context: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a production query result using Gemini as judge.
        Logs evaluation metrics to Phoenix UI.
        
        Args:
            query: User query
            rag_result: Result from query_rag_pipeline
            user_id: User ID
            project_id: Project ID
            span_context: OpenTelemetry span to attach metrics to (optional)
            
        Returns:
            Evaluation scores or None
        """
        import random
        from opentelemetry import trace
        
        # Sampling: only evaluate a fraction of queries
        if random.random() > self.sample_rate:
            return None
        
        logger.info(f"[LIVE_EVAL] Evaluating query with Gemini judge: {query[:50]}...")
        
        try:
            # Extract data from RAG result
            response = rag_result.get("result", "")
            retrieved_docs = rag_result.get("source_documents", [])
            
            if not retrieved_docs:
                logger.warning(f"[LIVE_EVAL] No retrieved docs for query, skipping evaluation")
                return None
            
            # Evaluate response quality using Gemini judge
            response_eval = await self.response_judge.evaluate_response(
                query=query,
                response=response,
                retrieved_chunks=retrieved_docs
            )
            
            # Optionally evaluate retrieval quality
            retrieval_eval = await self.retrieval_judge.evaluate_retrieved_chunks(
                query=query,
                retrieved_chunks=retrieved_docs
            )
            
            # Combine results
            eval_scores = {
                "query": query,
                "user_id": user_id,
                "project_id": project_id,
                "model_used": rag_result.get("model_used"),
                "processing_time": rag_result.get("processing_time"),
                "is_comparative": rag_result.get("is_comparative", False),
                
                # Response quality scores
                "faithfulness_score": response_eval.get("faithfulness_score"),
                "relevance_score": response_eval.get("relevance_score"),
                "completeness_score": response_eval.get("completeness_score"),
                "hallucination_score": response_eval.get("hallucination_score"),
                "toxicity_score": response_eval.get("toxicity_score"),
                "overall_response_score": response_eval.get("overall_score"),
                
                # Retrieval quality scores
                "retrieval_quality_score": retrieval_eval.get("overall_score"),
                "num_relevant_chunks": retrieval_eval.get("num_relevant_chunks"),
                
                # Metadata
                "response_length": len(response),
                "num_chunks_used": len(retrieved_docs),
                "evaluated_at": datetime.utcnow().isoformat(),
                "evaluation_method": "gemini_judge"
            }
            
            # ============================================================
            # LOG EVALUATION METRICS TO PHOENIX (if span provided)
            # ============================================================
            if span_context:
                try:
                    # Create a new child span for evaluation metrics
                    # This keeps the evaluation separate from the main flow
                    tracer = trace.get_tracer(__name__)
                    with tracer.start_as_current_span(
                        "evaluation_results",
                        context=span_context
                    ) as eval_span:
                        # Add evaluation scores as span attributes
                        eval_span.set_attribute("eval.faithfulness_score", float(eval_scores.get("faithfulness_score") or 0))
                        eval_span.set_attribute("eval.relevance_score", float(eval_scores.get("relevance_score") or 0))
                        eval_span.set_attribute("eval.completeness_score", float(eval_scores.get("completeness_score") or 0))
                        eval_span.set_attribute("eval.hallucination_score", float(eval_scores.get("hallucination_score") or 0))
                        eval_span.set_attribute("eval.overall_score", float(eval_scores.get("overall_response_score") or 0))
                        eval_span.set_attribute("eval.retrieval_quality", float(eval_scores.get("retrieval_quality_score") or 0))
                        eval_span.set_attribute("eval.num_relevant_chunks", int(eval_scores.get("num_relevant_chunks") or 0))
                        
                        # Add explanations as attributes (for debugging)
                        if response_eval.get("faithfulness_explanation"):
                            eval_span.set_attribute("eval.faithfulness_explanation", str(response_eval["faithfulness_explanation"])[:500])
                        if response_eval.get("relevance_explanation"):
                            eval_span.set_attribute("eval.relevance_explanation", str(response_eval["relevance_explanation"])[:500])
                        if response_eval.get("completeness_explanation"):
                            eval_span.set_attribute("eval.completeness_explanation", str(response_eval["completeness_explanation"])[:500])
                        if response_eval.get("hallucination_explanation"):
                            eval_span.set_attribute("eval.hallucination_explanation", str(response_eval["hallucination_explanation"])[:500])
                        
                        logger.debug(f"[LIVE_EVAL] Logged metrics to Phoenix span")
                except Exception as span_error:
                    logger.warning(f"[LIVE_EVAL] Failed to log to Phoenix span: {span_error}")
            
            # Save to database (async, non-blocking)
            await self.eval_collection.insert_one(eval_scores)
            
            logger.info(f"[LIVE_EVAL] Evaluation complete. Overall score: {eval_scores['overall_response_score']}")
            
            return eval_scores
            
        except Exception as e:
            logger.error(f"[LIVE_EVAL] Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def get_evaluation_stats(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get aggregated evaluation statistics."""
        from datetime import timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        cutoff_iso = cutoff_time.isoformat()
        
        pipeline = [
            {
                "$match": {
                    "evaluated_at": {"$gte": cutoff_iso}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_evaluated": {"$sum": 1},
                    "avg_faithfulness_score": {"$avg": "$faithfulness_score"},
                    "avg_relevance_score": {"$avg": "$relevance_score"},
                    "avg_completeness_score": {"$avg": "$completeness_score"},
                    "avg_hallucination_score": {"$avg": "$hallucination_score"},
                    "avg_toxicity_score": {"$avg": "$toxicity_score"},
                    "avg_overall_score": {"$avg": "$overall_response_score"},
                    "avg_retrieval_quality": {"$avg": "$retrieval_quality_score"},
                    "avg_response_length": {"$avg": "$response_length"}
                }
            }
        ]
        
        result = await self.eval_collection.aggregate(pipeline).to_list(length=1)
        
        if result:
            stats = result[0]
            stats.pop("_id", None)
            return stats
        
        return {
            "total_evaluated": 0,
            "message": "No evaluations in time window"
        }


# ============================================================================
# INTEGRATION WITH RAG SERVICE
# ============================================================================

async def evaluate_rag_query_live(
    query: str,
    rag_result: Dict[str, Any],
    user_id: str,
    project_id: str,
    sample_rate: float = 1.0,
    span_context: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a RAG query in production (to be called after query_rag_pipeline).
    
    Usage in rag_service.py:
    ```python
    # After getting rag_result from query_rag_pipeline
    eval_scores = await evaluate_rag_query_live(
        query=query,
        rag_result=rag_result,
        user_id=current_user.user_id,
        project_id=project_id,
        sample_rate=1.0,  # Evaluate 100% of queries (every query)
        span_context=trace.get_current_span().get_span_context()  # Optional
    )
    ```
    
    Args:
        query: User query
        rag_result: Result from query_rag_pipeline
        user_id: User ID
        project_id: Project ID
        sample_rate: Fraction of queries to evaluate (default: 1.0 = 100%)
        span_context: OpenTelemetry span context for Phoenix logging
        
    Returns:
        Evaluation scores or None
    """
    evaluator = LiveEvaluationWrapper(sample_rate=sample_rate)
    return await evaluator.evaluate_query_result(
        query=query,
        rag_result=rag_result,
        user_id=user_id,
        project_id=project_id,
        span_context=span_context
    )


# ============================================================================
# EVALUATION API ENDPOINTS (Simplified - Live Evaluation Only)
# ============================================================================

async def get_live_eval_stats(hours: int = 24) -> Dict[str, Any]:
    """
    Get live evaluation statistics.
    
    Args:
        hours: Time window in hours
        
    Returns:
        Aggregated statistics
    """
    evaluator = LiveEvaluationWrapper()
    return await evaluator.get_evaluation_stats(time_window_hours=hours)


async def get_evaluation_history(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent live evaluation results.
    
    Args:
        limit: Number of results to return
        
    Returns:
        List of evaluation results
    """
    eval_collection = db["live_evaluations"]
    
    results = await eval_collection.find(
        sort=[("evaluated_at", -1)]
    ).limit(limit).to_list(length=limit)
    
    return results


# ============================================================================
# SINGLETON INSTANCES
# ============================================================================

def get_live_evaluator(sample_rate: float = 1.0) -> LiveEvaluationWrapper:
    """Get singleton live evaluator instance."""
    if not hasattr(get_live_evaluator, "_instance"):
        get_live_evaluator._instance = LiveEvaluationWrapper(sample_rate)
    return get_live_evaluator._instance