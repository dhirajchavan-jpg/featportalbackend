# app/services/evaluation/pipeline_evaluator.py
"""
RAG Pipeline Evaluator - Complete Orchestrator

Orchestrates all 6 evaluation stages:
1. Query Processing
2. Embedding Quality  
3. Retrieval Quality
4. Reranking Effectiveness
5. Context Quality
6. Generation Quality

Runs evaluations, aggregates metrics, logs to Phoenix, saves to MongoDB.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


# Import all evaluators
from app.services.evaluation.query_evaluator import QueryProcessingEvaluator
from app.services.evaluation.embedding_evaluator import EmbeddingQualityEvaluator
from app.services.evaluation.retrieval_evaluator import RetrievalMetricsEvaluator
from app.services.evaluation.reranking_evaluator import RerankingEffectivenessEvaluator
from app.services.evaluation.context_evaluator import ContextQualityEvaluator
from app.services.evaluation.gemini_judge_evaluator import GeminiJudgeEvaluator, RetrievalQualityEvaluator as GeminiRetrievalEvaluator

# Phoenix integration for telemetry
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class ComprehensivePipelineEvaluator:
    """
    Orchestrates all evaluation stages for complete pipeline assessment.
    """
    
    def __init__(
        self,
        evaluate_query: bool = True,
        evaluate_embedding: bool = True,
        evaluate_retrieval: bool = True,
        evaluate_reranking: bool = True,
        evaluate_context: bool = True,
        evaluate_generation: bool = True,
        embedding_model_name: Optional[str] = None
    ):
        """
        Initialize comprehensive evaluator.
        """
        self.name = "comprehensive_pipeline"
        
        # Initialize evaluators based on configuration
        self.query_evaluator = QueryProcessingEvaluator() if evaluate_query else None
        self.embedding_evaluator = EmbeddingQualityEvaluator(embedding_model_name) if evaluate_embedding else None
        self.retrieval_evaluator = RetrievalMetricsEvaluator() if evaluate_retrieval else None
        self.reranking_evaluator = RerankingEffectivenessEvaluator() if evaluate_reranking else None
        self.context_evaluator = ContextQualityEvaluator() if evaluate_context else None
        self.generation_evaluator = GeminiJudgeEvaluator() if evaluate_generation else None
        self.gemini_retrieval_evaluator = GeminiRetrievalEvaluator() if evaluate_generation else None
        
        logger.info("[PIPELINE_EVAL] Comprehensive evaluator initialized")
        logger.info(f"[PIPELINE_EVAL] Active stages: {self._get_active_stages()}")
    
    async def evaluate_complete_pipeline(
        self,
        # Query stage
        original_query: str,
        expanded_query: Optional[str] = None,
        query_processing_time_ms: Optional[float] = None,
        
        # Embedding stage
        query_embedding: Optional[Any] = None,
        doc_embeddings: Optional[List[Any]] = None,
        embedding_time_ms: Optional[float] = None,
        
        # Retrieval stage
        docs_before_rerank: Optional[List[Dict[str, Any]]] = None,
        retrieval_time_ms: Optional[float] = None,
        
        # Reranking stage
        docs_after_rerank: Optional[List[Dict[str, Any]]] = None,
        reranking_time_ms: Optional[float] = None,
        
        # Context stage
        assembled_context: Optional[str] = None,
        context_tokens: Optional[int] = None,
        
        # Generation stage
        generated_response: Optional[str] = None,
        generation_time_ms: Optional[float] = None,
        
        # Metadata
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        
        # --- NEW: Accept Model Config & Kwargs to prevent crashes ---
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run complete pipeline evaluation.
        
        Returns comprehensive metrics from all stages.
        """
        eval_start = time.time()
        
        all_metrics = {
            'query': original_query,
            'user_id': user_id,
            'project_id': project_id,
            'model_config': model_config, # Save config to metrics
            'evaluated_at': datetime.now(timezone.utc).isoformat(),
            'evaluation_method': 'comprehensive_pipeline'
        }
        
        # Stage 1: Query Processing
        if self.query_evaluator and expanded_query:
            logger.info("[PIPELINE_EVAL] Stage 1: Evaluating query processing...")
            query_metrics = self.query_evaluator.evaluate_query_processing(
                original_query=original_query,
                expanded_query=expanded_query,
                processing_time_ms=query_processing_time_ms
            )
            all_metrics['query_processing'] = query_metrics
        
        # Stage 2: Embedding Quality
        if self.embedding_evaluator and query_embedding is not None:
            logger.info("[PIPELINE_EVAL] Stage 2: Evaluating embedding quality...")
            doc_texts = None
            if docs_after_rerank or docs_before_rerank:
                docs = docs_after_rerank or docs_before_rerank
                doc_texts = [doc.get('page_content', '') for doc in docs[:5]]
            
            embedding_metrics = self.embedding_evaluator.evaluate_embeddings(
                query=original_query,
                query_embedding=query_embedding,
                doc_embeddings=doc_embeddings,
                doc_texts=doc_texts,
                embedding_time_ms=embedding_time_ms
            )
            all_metrics['embedding'] = embedding_metrics
        
        # Stage 3: Retrieval Quality
        if self.retrieval_evaluator and (docs_before_rerank or docs_after_rerank):
            logger.info("[PIPELINE_EVAL] Stage 3: Evaluating retrieval quality...")
            # Use pre-rerank docs if available, otherwise post-rerank
            docs_to_eval = docs_before_rerank or docs_after_rerank
            retrieval_metrics = self.retrieval_evaluator.evaluate_retrieval(
                query=original_query,
                retrieved_docs=docs_to_eval,
                k_values=[1, 3, 5, 10]
            )
            retrieval_metrics['retrieval_time_ms'] = retrieval_time_ms
            all_metrics['retrieval'] = retrieval_metrics
        
        # Stage 4: Reranking Effectiveness
        if self.reranking_evaluator and docs_before_rerank and docs_after_rerank:
            logger.info("[PIPELINE_EVAL] Stage 4: Evaluating reranking effectiveness...")
            reranking_metrics = self.reranking_evaluator.evaluate_reranking(
                docs_before_rerank=docs_before_rerank,
                docs_after_rerank=docs_after_rerank,
                reranking_time_ms=reranking_time_ms
            )
            all_metrics['reranking'] = reranking_metrics
        
        # Stage 5: Context Quality
        if self.context_evaluator and assembled_context:
            logger.info("[PIPELINE_EVAL] Stage 5: Evaluating context quality...")
            source_docs = docs_after_rerank or docs_before_rerank
            context_metrics = self.context_evaluator.evaluate_context(
                query=original_query,
                context=assembled_context,
                source_documents=source_docs,
                context_tokens=context_tokens
            )
            all_metrics['context'] = context_metrics
        
        # Stage 6: Generation Quality (Gemini Judge)
        if self.generation_evaluator and generated_response:
            logger.info("[PIPELINE_EVAL] Stage 6: Evaluating generation quality...")
            retrieved_chunks = docs_after_rerank or docs_before_rerank or []
            
            # Response evaluation
            generation_metrics = await self.generation_evaluator.evaluate_response(
                query=original_query,
                response=generated_response,
                retrieved_chunks=retrieved_chunks
            )
            generation_metrics['generation_time_ms'] = generation_time_ms
            all_metrics['generation'] = generation_metrics
        
        # Calculate overall metrics
        eval_time = (time.time() - eval_start) * 1000  # ms
        all_metrics['evaluation_time_ms'] = eval_time
        
        # Pipeline health score
        health_score = self._calculate_pipeline_health(all_metrics)
        all_metrics['pipeline_health_score'] = health_score
        
        # Stage-wise scores summary
        stage_scores = self._extract_stage_scores(all_metrics)
        all_metrics['stage_scores_summary'] = stage_scores
        
        logger.info(f"[PIPELINE_EVAL] Complete. Pipeline health: {health_score:.2f}")
        
        # ========== ADD COMPREHENSIVE METRICS TO PHOENIX ==========
        # Extract trace context if provided
        trace_context = kwargs.get('_trace_context') or all_metrics.pop('_trace_context', None)
        
        try:
            # Import context management
            from opentelemetry import context as otel_context
            
            # Restore the trace context if provided
            token = None
            if trace_context:
                token = otel_context.attach(trace_context)
            
            # Create evaluation span in the same trace
            with tracer.start_as_current_span(
                "comprehensive_evaluation",
                attributes={
                    SpanAttributes.INPUT_VALUE: original_query[:100] if original_query else "",
                    "evaluation.pipeline_health_score": float(health_score),
                    "evaluation.time_ms": float(eval_time),
                    "evaluation.model_config.strategy": model_config.get('search_strategy', 'unknown') if model_config else 'default'
                }
            ) as eval_span:
                
                # Stage 1: Query Processing Metrics
                if 'query_processing' in all_metrics:
                    qp = all_metrics['query_processing']
                    eval_span.set_attribute("evaluation.query.expansion_quality", float(qp.get('expansion_quality_score', 0)))
                    eval_span.set_attribute("evaluation.query.semantic_similarity", float(qp.get('semantic_similarity', 0)))
                    eval_span.set_attribute("evaluation.query.clarity", float(qp.get('clarity_score', 0)))
                
                # Stage 3: Retrieval Metrics
                if 'retrieval' in all_metrics:
                    ret = all_metrics['retrieval']
                    eval_span.set_attribute("evaluation.retrieval.hit_rate_3", float(ret.get('estimated_hit_rate_at_3', ret.get('hit_rate_at_3', 0))))
                    eval_span.set_attribute("evaluation.retrieval.precision_3", float(ret.get('estimated_precision_at_3', ret.get('precision_at_3', 0))))
                    eval_span.set_attribute("evaluation.retrieval.mrr", float(ret.get('mrr', ret.get('estimated_mrr', 0))))
                    eval_span.set_attribute("evaluation.retrieval.ndcg_3", float(ret.get('ndcg_at_3', ret.get('estimated_ndcg_at_3', 0))))
                    eval_span.set_attribute("evaluation.retrieval.max_score", float(ret.get('max_score', 0)))
                
                # Stage 4: Reranking Metrics
                if 'reranking' in all_metrics:
                    rr = all_metrics['reranking']
                    eval_span.set_attribute("evaluation.reranking.effectiveness", float(rr.get('reranking_effectiveness_score', 0)))
                    eval_span.set_attribute("evaluation.reranking.score_improvement", float(rr.get('score_improvement_mean', 0)))
                    eval_span.set_attribute("evaluation.reranking.top3_changed", bool(rr.get('top_3_changed', False)))
                
                # Stage 5: Context Quality Metrics
                if 'context' in all_metrics:
                    ctx = all_metrics['context']
                    eval_span.set_attribute("evaluation.context.quality", float(ctx.get('context_quality_score', 0)))
                    eval_span.set_attribute("evaluation.context.relevance", float(ctx.get('context_relevance_mean', 0)))
                    eval_span.set_attribute("evaluation.context.coverage", float(ctx.get('coverage_score', 0)))
                    eval_span.set_attribute("evaluation.context.coherence", float(ctx.get('coherence_score', 0)))
                
                # Stage 6: Generation Quality Metrics (Gemini Judge)
                if 'generation' in all_metrics:
                    gen = all_metrics['generation']
                    eval_span.set_attribute("evaluation.generation.overall", float(gen.get('overall_score', 0)))
                    eval_span.set_attribute("evaluation.generation.faithfulness", float(gen.get('faithfulness_score', 0)))
                    eval_span.set_attribute("evaluation.generation.relevance", float(gen.get('relevance_score', 0)))
                    eval_span.set_attribute("evaluation.generation.completeness", float(gen.get('completeness_score', 0)))
                    eval_span.set_attribute("evaluation.generation.hallucination", float(gen.get('hallucination_score', 0)))
                    eval_span.set_attribute("evaluation.generation.toxicity", float(gen.get('toxicity_score', 0)))
                
                # Stage Summary
                if stage_scores:
                    for stage, score in stage_scores.items():
                        eval_span.set_attribute(f"evaluation.stage.{stage}", float(score))
                
                logger.info("[PIPELINE_EVAL]  Comprehensive metrics added to Phoenix")
            
            # Detach context if we attached it
            if trace_context and token:
                otel_context.detach(token)
                
        except Exception as phoenix_error:
            logger.error(f"[PIPELINE_EVAL]  Failed to add Phoenix metrics: {phoenix_error}")
            # import traceback
            # traceback.print_exc()
        # ================================================================
        
        return all_metrics
    
    def _calculate_pipeline_health(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate overall pipeline health score (0-1).
        Uses weighted scoring: Generation & Retrieval are critical.
        """
        scores = {}
        
        # 1. Extract and Normalize Scores to 0.0 - 1.0
        
        # Query (Stage 1)
        if 'query_processing' in metrics:
            scores['query'] = metrics['query_processing'].get('expansion_quality_score', 0.5)
            
        # Retrieval (Stage 3) - CRITICAL
        if 'retrieval' in metrics:
            # Prefer NDCG, fallback to Hit Rate
            val = metrics['retrieval'].get('ndcg_at_3') or metrics['retrieval'].get('estimated_hit_rate_at_3') or 0.0
            scores['retrieval'] = val
            
        # Reranking (Stage 4)
        if 'reranking' in metrics:
            scores['reranking'] = metrics['reranking'].get('reranking_effectiveness_score', 0.5)
            
        # Context (Stage 5)
        if 'context' in metrics:
            scores['context'] = metrics['context'].get('context_quality_score', 0.5)
            
        # Generation (Stage 6) - CRITICAL
        if 'generation' in metrics:
            raw_score = metrics['generation'].get('overall_score', 0)
            scores['generation'] = raw_score / 100.0  # Convert 0-100 to 0-1

        if not scores:
            return 0.0

        # 2. Apply Weights
        # Weights: Generation(40%) + Retrieval(30%) + Context(15%) + Query(10%) + Reranking(5%)
        weighted_sum = 0.0
        total_weight = 0.0
        
        weights = {
            'generation': 0.40,
            'retrieval': 0.30,
            'context': 0.15,
            'query': 0.10,
            'reranking': 0.05
        }
        
        for stage, score in scores.items():
            weight = weights.get(stage, 0.1)
            weighted_sum += score * weight
            total_weight += weight
            
        # 3. Penalize Critical Failures
        # If Retrieval, Generation, OR Toxicity is bad
        retrieval_ok = scores.get('retrieval', 1.0) >= 0.2
        generation_ok = scores.get('generation', 1.0) >= 0.2
        
        # Check Toxicity (if available). > 0.5 (50/100) is considered toxic
        toxicity_score = 0
        if 'generation' in metrics:
             toxicity_score = metrics['generation'].get('toxicity_score', 0) or 0
        
        is_toxic = toxicity_score > 50
        
        critical_failure = (not retrieval_ok) or (not generation_ok) or is_toxic
        
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        if critical_failure:
            return min(final_score, 0.4) # Cap health at 40% if toxic
            
        return final_score
    
    def _extract_stage_scores(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract key score from each stage for easy summary."""
        summary = {}
        
        if 'query_processing' in metrics:
            summary['query'] = metrics['query_processing'].get('expansion_quality_score', 0.0)
        
        if 'embedding' in metrics:
            summary['embedding'] = metrics['embedding'].get('semantic_coherence_score', 0.0)
        
        if 'retrieval' in metrics:
            summary['retrieval'] = metrics['retrieval'].get('estimated_hit_rate_at_3', 
                                                           metrics['retrieval'].get('hit_rate_at_3', 0.0))
        
        if 'reranking' in metrics:
            summary['reranking'] = metrics['reranking'].get('reranking_effectiveness_score', 0.0)
        
        if 'context' in metrics:
            summary['context'] = metrics['context'].get('context_quality_score', 0.0)
        
        if 'generation' in metrics:
            summary['generation'] = metrics['generation'].get('overall_score', 0.0) / 100.0
        
        return summary
    
    def _get_active_stages(self) -> List[str]:
        """Get list of active evaluation stages."""
        stages = []
        if self.query_evaluator:
            stages.append("query_processing")
        if self.embedding_evaluator:
            stages.append("embedding")
        if self.retrieval_evaluator:
            stages.append("retrieval")
        if self.reranking_evaluator:
            stages.append("reranking")
        if self.context_evaluator:
            stages.append("context")
        if self.generation_evaluator:
            stages.append("generation")
        return stages


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

async def evaluate_complete_rag_pipeline(
    original_query: str,
    expanded_query: Optional[str] = None,
    docs_after_rerank: Optional[List[Dict[str, Any]]] = None,
    assembled_context: Optional[str] = None,
    generated_response: Optional[str] = None,
    model_config: Optional[Dict[str, Any]] = None, # Added argument here too
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for complete pipeline evaluation.
    """
    evaluator = ComprehensivePipelineEvaluator()
    return await evaluator.evaluate_complete_pipeline(
        original_query=original_query,
        expanded_query=expanded_query,
        docs_after_rerank=docs_after_rerank,
        assembled_context=assembled_context,
        generated_response=generated_response,
        model_config=model_config, # Pass it down
        **kwargs
    )