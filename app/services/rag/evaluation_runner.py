import os
from app.services.rag.history_and_cache import _cache_result
os.environ["HF_HUB_OFFLINE"] = "1"
from opentelemetry import trace, context as otel_context
import time
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# ============================================================================
# EVALUATION IMPORTS - Comprehensive 6-Stage System
# ============================================================================
from app.services.evaluation.pipeline_evaluator import ComprehensivePipelineEvaluator

_comprehensive_evaluator = None

def get_comprehensive_evaluator():
    """Get or create comprehensive evaluator (singleton)."""
    global _comprehensive_evaluator
    if _comprehensive_evaluator is None:
        _comprehensive_evaluator = ComprehensivePipelineEvaluator(
            evaluate_query=True,        # Query expansion quality
            evaluate_embedding=False,   # Skip for now (expensive, optional)
            evaluate_retrieval=True,    # Hit rate, MRR, NDCG
            evaluate_reranking=True,    # Reranking effectiveness
            evaluate_context=True,      # Context quality
            evaluate_generation=True,   # Gemini judge
            embedding_model_name="all-MiniLM-L6-v2"
        )
        logger.info("[EVAL] Comprehensive evaluator initialized")
    return _comprehensive_evaluator


def _run_evaluation_sync(eval_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous wrapper for evaluation - runs in thread pool.
    Creates its own event loop to run async evaluation code.
    """
    import asyncio
    
    # Extract trace context if present (but don't pass to evaluator)
    trace_context = eval_data.pop('_trace_context', None)
    
    # Get evaluator
    evaluator = get_comprehensive_evaluator()
    
    # Create new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run evaluation in this thread's event loop
        metrics = loop.run_until_complete(
            evaluator.evaluate_complete_pipeline(**eval_data)
        )
        
        # Restore trace context to metrics (for Phoenix span creation)
        if trace_context:
            metrics['_trace_context'] = trace_context
        
        return metrics
        
    finally:
        loop.close()


async def _run_comprehensive_evaluation_background(eval_data: Dict[str, Any]):
    """
    Run comprehensive 6-stage evaluation in THREAD POOL.
    
     CRITICAL: Uses thread pool to avoid blocking the event loop.
    FastAPI's BackgroundTasks will run this AFTER response is sent.
    
    Evaluates:
    1. Query processing
    2. Embedding quality (optional)
    3. Retrieval metrics
    4. Reranking effectiveness
    5. Context quality
    6. Generation quality
    """
    try:
        eval_start = time.time()
        
        logger.info(f"[EVAL] Starting comprehensive evaluation in background...")
        logger.info(f"[EVAL] Query: {eval_data.get('original_query', 'N/A')[:50]}...")
        
        #  Run in thread pool to avoid blocking event loop
        import asyncio
        loop = asyncio.get_event_loop()
        
        # Run evaluation in thread pool
        metrics = await loop.run_in_executor(
            None,  # Use default ThreadPoolExecutor
            _run_evaluation_sync,
            eval_data
        )
        
        # Save to MongoDB
        from app.database import db
        eval_collection = db['comprehensive_evaluations']
        
        # Remove non-serializable fields before saving
        if '_trace_context' in metrics:
            del metrics['_trace_context']
        if '_span_id' in metrics:
            del metrics['_span_id']

            # --- NEW: Add Metadata fields to the document ---
        metrics['user_query'] = eval_data.get('original_query')       
        metrics['llm_answer'] = eval_data.get('generated_response')   
        metrics['user_id'] = eval_data.get('user_id')                 
        metrics['project_id'] = eval_data.get('project_id')           
        metrics['timestamp'] = eval_start                             
        # -----------------------------------------------
        
        await eval_collection.insert_one(metrics)
        
        eval_time = (time.time() - eval_start) * 1000
        
        # Log results
        logger.info(f"[EVAL]  Evaluation complete in {eval_time:.0f}ms!")
        logger.info(f"[EVAL]  Pipeline health: {metrics.get('pipeline_health_score', 0):.2f}")
        logger.info(f"[EVAL] Stage scores: {metrics.get('stage_scores_summary', {})}")
        
        # Log individual stage highlights
        if 'query_processing' in metrics:
            score = metrics['query_processing'].get('expansion_quality_score', 0)
            logger.info(f"[EVAL]    Query expansion quality: {score:.2f}")
        
        if 'retrieval' in metrics:
            hit_rate = metrics['retrieval'].get('estimated_hit_rate_at_3', 
                       metrics['retrieval'].get('hit_rate_at_3', 0))
            logger.info(f"[EVAL] Retrieval hit rate@3: {hit_rate:.2f}")
        
        if 'reranking' in metrics:
            score = metrics['reranking'].get('reranking_effectiveness_score', 0)
            logger.info(f"[EVAL] Reranking effectiveness: {score:.2f}")
        
        if 'context' in metrics:
            score = metrics['context'].get('context_quality_score', 0)
            logger.info(f"[EVAL] Context quality: {score:.2f}")
        
        if 'generation' in metrics:
            score = metrics['generation'].get('overall_score', 0)
            logger.info(f"[EVAL] Generation score: {score:.0f}/100")

            # =================================================================
            # CONDITIONAL CACHING LOGIC (Cleaned)
            # =================================================================
            
            gen_metrics = metrics.get('generation', {})
            raw_hallucination_score = gen_metrics.get('hallucination_score')
            
            # Normalize Score (Default to 100.0 if missing/error)
            final_hallucination_score = 100.0 
            if raw_hallucination_score is not None:
                try:
                    score_val = float(raw_hallucination_score)
                    if score_val <= 1.0 and score_val > 0:
                        final_hallucination_score = score_val * 100.0
                    else:
                        final_hallucination_score = score_val
                except (ValueError, TypeError):
                    logger.error(f"[EVAL] Invalid hallucination score format: {raw_hallucination_score}")

            threshold = float(eval_data.get('hallucination_threshold', 100.0))
            cache_key = eval_data.get('cache_key')
            
            if cache_key:
                if final_hallucination_score < threshold:
                    # Log the approval with the score/threshold context
                    logger.info(f"[EVAL] Caching APPROVED (Score: {final_hallucination_score:.1f}% < {threshold:.1f}%). Saving to database.")
                    try:
                        await _cache_result(
                            cache_key=cache_key,
                            user_id=eval_data.get('user_id'),
                            query=eval_data.get('original_query'),
                            project_id=eval_data.get('project_id'),
                            sectors=eval_data.get('sectors'),
                            answer=eval_data.get('generated_response'),
                            style=eval_data.get('style') # <--- UPDATED: Pass style here
                        )
                    except Exception as e:
                        logger.error(f"[EVAL] Cache save failed: {e}")
                else:
                    logger.warning(f"[EVAL] Caching BLOCKED. Hallucination ({final_hallucination_score:.1f}%) >= Threshold ({threshold:.1f}%).")
            else:
                logger.info("[EVAL] Cache skipped (Refusal message or no key provided).")
            
        
        logger.info(f"[EVAL] Results saved to MongoDB")
        
    except Exception as e:
        logger.error(f"[EVAL] Background evaluation failed: {e}")
        import traceback
        traceback.print_exc()