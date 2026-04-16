"""
Comprehensive Evaluation API Endpoints

Provides endpoints for viewing 6-stage pipeline evaluation metrics:
- Stage 1: Query Processing
- Stage 2: Embedding Quality (optional)
- Stage 3: Retrieval Metrics
- Stage 4: Reranking Effectiveness
- Stage 5: Context Quality
- Stage 6: Generation Quality

Plus overall pipeline health scoring.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from bson import ObjectId
from datetime import datetime, timedelta,timezone
from collections import defaultdict


from app.dependencies import get_current_user, UserPayload
from app.database import db
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


# ============================================================================
# MONGODB OBJECTID SERIALIZATION HELPER
# ============================================================================

def serialize_mongo_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert MongoDB document to JSON-serializable format, handling NaNs."""
    if doc is None:
        return None
    
    serialized = {}
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            serialized[key] = str(value)
        elif isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, float):
            # Check for NaN or Infinity and convert to None
            if math.isnan(value) or math.isinf(value):
                serialized[key] = None
            else:
                serialized[key] = value
        elif isinstance(value, dict):
            serialized[key] = serialize_mongo_doc(value)
        elif isinstance(value, list):
            serialized[key] = [
                serialize_mongo_doc(item) if isinstance(item, dict) else 
                str(item) if isinstance(item, ObjectId) else
                item.isoformat() if isinstance(item, datetime) else 
                (None if isinstance(item, float) and (math.isnan(item) or math.isinf(item)) else item)
                for item in value
            ]
        else:
            serialized[key] = value
    return serialized


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class EvaluationStatsResponse(BaseModel):
    """Response model for evaluation statistics."""
    total_evaluated: int
    avg_pipeline_health: Optional[float] = None
    avg_query_quality: Optional[float] = None
    avg_retrieval_hit_rate: Optional[float] = None
    avg_reranking_effectiveness: Optional[float] = None
    avg_context_quality: Optional[float] = None
    avg_generation_score: Optional[float] = None
    avg_query_toxicity: Optional[float] = None
    avg_response_toxicity: Optional[float] = None
    time_window_hours: int
    message: Optional[str] = None


class PipelineHealthResponse(BaseModel):
    """Response model for pipeline health check."""
    overall_health: float
    stage_health: Dict[str, float]
    issues: List[str]
    recommendations: List[str]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def get_comprehensive_stats(hours: int = 24) -> Dict[str, Any]:
    """
    Get aggregated statistics from comprehensive_evaluations collection.
    
    Args:
        hours: Time window in hours
        
    Returns:
        Dictionary of aggregated stats
    """
    try:
        collection = db['comprehensive_evaluations']
        
        # Calculate time threshold
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        
        # Query evaluations in time window
        cursor = collection.find({
            'evaluated_at': {'$gte': time_threshold.isoformat()}
        })
        
        evaluations = await cursor.to_list(length=None)
        
        if not evaluations:
            # --- FIX: Added 'time_window_hours' to satisfy Pydantic model ---
            return {
                'total_evaluated': 0,
                'time_window_hours': hours, 
                'message': f'No evaluations found in last {hours} hours'
            }
        
        # Aggregate metrics
        stats = {
            'total_evaluated': len(evaluations),
            'time_window_hours': hours
        }
        
        # Pipeline health scores
        pipeline_healths = [e.get('pipeline_health_score', 0) for e in evaluations if e.get('pipeline_health_score')]
        if pipeline_healths:
            stats['avg_pipeline_health'] = sum(pipeline_healths) / len(pipeline_healths)
            stats['min_pipeline_health'] = min(pipeline_healths)
            stats['max_pipeline_health'] = max(pipeline_healths)
        
        # Query processing
        query_scores = [
            e.get('query_processing', {}).get('expansion_quality_score', 0) 
            for e in evaluations 
            if e.get('query_processing')
        ]
        if query_scores:
            stats['avg_query_quality'] = sum(query_scores) / len(query_scores)
        
        # Retrieval
        retrieval_hits = [
            e.get('retrieval', {}).get('estimated_hit_rate_at_3', e.get('retrieval', {}).get('hit_rate_at_3', 0))
            for e in evaluations 
            if e.get('retrieval')
        ]
        if retrieval_hits:
            stats['avg_retrieval_hit_rate'] = sum(retrieval_hits) / len(retrieval_hits)
        
        # Reranking
        reranking_scores = [
            e.get('reranking', {}).get('reranking_effectiveness_score', 0)
            for e in evaluations 
            if e.get('reranking')
        ]
        if reranking_scores:
            stats['avg_reranking_effectiveness'] = sum(reranking_scores) / len(reranking_scores)
        
        # Context
        context_scores = [
            e.get('context', {}).get('context_quality_score', 0)
            for e in evaluations 
            if e.get('context')
        ]
        if context_scores:
            stats['avg_context_quality'] = sum(context_scores) / len(context_scores)
        
        # Generation
        generation_scores = [
            e.get('generation', {}).get('overall_score', 0)
            for e in evaluations 
            if e.get('generation')
        ]
        if generation_scores:
            stats['avg_generation_score'] = sum(generation_scores) / len(generation_scores)

        q_tox_scores = [
            e.get('generation', {}).get('query_toxicity_score', 0)
            for e in evaluations 
            if e.get('generation') and e.get('generation', {}).get('query_toxicity_score') is not None
        ]
        if q_tox_scores:
            stats['avg_query_toxicity'] = sum(q_tox_scores) / len(q_tox_scores)

        r_tox_scores = [
            e.get('generation', {}).get('response_toxicity_score', 0)
            for e in evaluations 
            if e.get('generation') and e.get('generation', {}).get('response_toxicity_score') is not None
        ]
        if r_tox_scores:
            stats['avg_response_toxicity'] = sum(r_tox_scores) / len(r_tox_scores)
        
        return stats
        
    except Exception as e:
        logger.error(f"[EVAL_API] Failed to get comprehensive stats: {e}")
        raise



async def get_pipeline_health() -> Dict[str, Any]:
    """
    Analyze pipeline health from recent evaluations.
    
    Returns:
        Health report with issues and recommendations
    """
    try:
        # Get stats from last 24 hours
        stats = await get_comprehensive_stats(hours=24)
        
        if stats.get('total_evaluated', 0) == 0:
            return {
                'overall_health': 0.0,
                'stage_health': {},
                'issues': ['No recent evaluations found'],
                'recommendations': ['Make some queries to generate evaluation data']
            }
        
        # Calculate stage health scores
        stage_health = {
            'query': stats.get('avg_query_quality', 0),
            'retrieval': stats.get('avg_retrieval_hit_rate', 0),
            'reranking': stats.get('avg_reranking_effectiveness', 0),
            'context': stats.get('avg_context_quality', 0),
            'generation': stats.get('avg_generation_score', 0) / 100  # Normalize to 0-1
        }
        
        # Overall health (weighted average)
        overall_health = stats.get('avg_pipeline_health', 0)
        
        # Identify issues and recommendations
        issues = []
        recommendations = []
        
        if stage_health['query'] < 0.6:
            issues.append('Query processing quality is low')
            recommendations.append('Review query expansion logic and check for proper semantic preservation')
        
        if stage_health['retrieval'] < 0.5:
            issues.append('Retrieval is not finding relevant documents')
            recommendations.append('Add more documents, improve chunking strategy, or tune retrieval parameters')
        
        if stage_health['reranking'] < 0.6:
            issues.append('Reranking is not improving results significantly')
            recommendations.append('Consider tuning reranking model or disabling if not adding value')
        
        if stage_health['context'] < 0.6:
            issues.append('Context quality is poor (redundancy, low relevance, or gaps)')
            recommendations.append('Reduce redundancy, filter low-relevance chunks, or improve context assembly')
        
        if stage_health['generation'] < 0.6:
            issues.append('Generation quality is low (hallucinations, low relevance, or incompleteness)')
            recommendations.append('Improve prompts, use better model, or ensure context has needed information')
        
        if overall_health >= 0.8:
            if not issues:
                recommendations.append('Pipeline is healthy! Monitor for any degradation.')
        
        return {
            'overall_health': overall_health,
            'stage_health': stage_health,
            'issues': issues if issues else ['No major issues detected'],
            'recommendations': recommendations if recommendations else ['Continue monitoring pipeline performance']
        }
        
    except Exception as e:
        logger.error(f"[EVAL_API] Failed to get pipeline health: {e}")
        raise


# ============================================================================
# EVALUATION ENDPOINTS
# ============================================================================

@router.get("/stats", response_model=EvaluationStatsResponse)
async def get_evaluation_statistics(
    hours: int = Query(default=24, ge=1, le=168, description="Time window in hours"),
    current_user: UserPayload = Depends(get_current_user)
):
    """
    Get comprehensive evaluation statistics for the specified time window.
    
    Shows aggregated metrics across all 6 pipeline stages from production queries.
    
    **Parameters:**
    - **hours**: Time window in hours (default: 24, max: 168 for 1 week)
    
    **Returns:**
    - Total evaluations in time window
    - Average scores for each stage
    - Overall pipeline health score
    """
    try:
        stats = await get_comprehensive_stats(hours=hours)
        return EvaluationStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"[EVAL_API] Failed to get stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@router.get("/health", response_model=PipelineHealthResponse)
async def get_evaluation_health(
    current_user: UserPayload = Depends(get_current_user)
):
    """
    Get pipeline health analysis based on recent evaluations.
    
    Analyzes evaluation data to identify issues and provide recommendations.
    
    **Returns:**
    - Overall pipeline health score (0-1)
    - Health score for each stage
    - List of identified issues
    - Recommendations for improvement
    """
    try:
        health = await get_pipeline_health()
        return PipelineHealthResponse(**health)
        
    except Exception as e:
        logger.error(f"[EVAL_API] Failed to get health: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve pipeline health: {str(e)}"
        )
import math

@router.get("/history")
async def get_evaluation_reports(
    limit: int = Query(default=100, ge=1, le=50),
    min_health: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    current_user: UserPayload = Depends(get_current_user)
):
    try:
        collection = db['comprehensive_evaluations']
        users_collection = db['users']

        # Build query
        query = {}
        if min_health is not None:
            query['pipeline_health_score'] = {'$gte': min_health}

        # Get reports
        cursor = collection.find(query).sort('evaluated_at', -1).limit(limit)
        reports = await cursor.to_list(length=limit)

        final_results = []

        for report in reports:
            user_id = report.get("user_id")
            user_info = None

            if user_id:
                user_doc = await users_collection.find_one({"_id": ObjectId(user_id)})
                if user_doc:
                    user_info = {
                        "id": str(user_doc["_id"]),
                        "name": user_doc.get("name"),
                        "role": user_doc.get("role")
                    }

            serialized = serialize_mongo_doc(report)
            serialized["user"] = user_info  # attach user details
            final_results.append(serialized)

        return JSONResponse(content=final_results)

    except Exception as e:
        logger.error(f"[EVAL_API] Failed to get history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve evaluation history: {str(e)}"
        )


@router.get("/metrics/summary")
async def get_metrics_summary(
    current_user: UserPayload = Depends(get_current_user)
):
    """
    Get summary of comprehensive evaluation metrics.
    
    Shows aggregated statistics from recent evaluations across multiple time windows.
    
    **Returns:**
    - Stats for last 24 hours
    - Stats for last 7 days
    - Pipeline health status
    """
    try:
        # Get stats from different time windows
        stats_24h = await get_comprehensive_stats(hours=24)
        stats_7d = await get_comprehensive_stats(hours=168)
        
        # Get health analysis
        health = await get_pipeline_health()
        
        summary = {
            "stats_24h": stats_24h,
            "stats_7d": stats_7d,
            "pipeline_health": health,
            "evaluation_method": "comprehensive_6_stage",
            "stages_evaluated": [
                "query_processing",
                "retrieval",
                "reranking",
                "context",
                "generation"
            ],
            "status": "healthy" if health['overall_health'] >= 0.7 else "needs_attention"
        }
        
        # Serialize any MongoDB ObjectIds
        serialized_summary = serialize_mongo_doc(summary)
        
        return JSONResponse(content=serialized_summary)
        
    except Exception as e:
        logger.error(f"[EVAL_API] Failed to get summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve metrics summary: {str(e)}"
        )


@router.get("/metrics/by-stage/{stage_name}")
async def get_stage_metrics(
    stage_name: str,
    hours: int = Query(default=24, ge=1, le=168),
    current_user: UserPayload = Depends(get_current_user)
):
    """
    Get detailed metrics for a specific pipeline stage.
    
    **Parameters:**
    - **stage_name**: Stage to query (query_processing, retrieval, reranking, context, generation)
    - **hours**: Time window in hours
    
    **Returns:**
    - Detailed metrics for the specified stage
    - Trend data over time
    """
    valid_stages = ['query_processing', 'retrieval', 'reranking', 'context', 'generation']
    
    if stage_name not in valid_stages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage name. Must be one of: {', '.join(valid_stages)}"
        )
    
    try:
        collection = db['comprehensive_evaluations']
        
        # Calculate time threshold
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        
        # Query evaluations
        cursor = collection.find({
            'evaluated_at': {'$gte': time_threshold.isoformat()},
            stage_name: {'$exists': True}
        }).sort('evaluated_at', -1)
        
        evaluations = await cursor.to_list(length=None)
        
        if not evaluations:
            return JSONResponse(content={
                'stage': stage_name,
                'total_evaluations': 0,
                'message': f'No evaluations found for {stage_name} in last {hours} hours'
            })
        
        # Extract stage-specific metrics
        stage_metrics = []
        for eval_doc in evaluations:
            stage_data = eval_doc.get(stage_name, {})
            if stage_data:
                metric_entry = {
                    'evaluated_at': eval_doc.get('evaluated_at'),
                    'query': eval_doc.get('query', '')[:50],  # Truncate query
                    **stage_data
                }
                stage_metrics.append(metric_entry)
        
        # Serialize
        serialized_metrics = serialize_mongo_doc({
            'stage': stage_name,
            'total_evaluations': len(stage_metrics),
            'time_window_hours': hours,
            'metrics': stage_metrics
        })
        
        return JSONResponse(content=serialized_metrics)
        
    except Exception as e:
        logger.error(f"[EVAL_API] Failed to get stage metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve stage metrics: {str(e)}"
        )


@router.get("/status")
async def get_evaluation_status(
    current_user: UserPayload = Depends(get_current_user)
):
    """
    Get current comprehensive evaluation system status.
    
    **Returns:**
    - Whether evaluation is enabled
    - Evaluation method (comprehensive 6-stage)
    - Sample rate (100% - every query)
    - Recent evaluation count
    - Pipeline health status
    """
    try:
        # Get stats from last 24 hours
        stats_24h = await get_comprehensive_stats(hours=24)
        health = await get_pipeline_health()
        
        return {
            "evaluation_enabled": True,
            "evaluation_method": "comprehensive_6_stage_pipeline",
            "stages": [
                "query_processing",
                "retrieval",
                "reranking",
                "context",
                "generation"
            ],
            "sample_rate": 1.0,  # 100% of queries
            "evaluations_last_24h": stats_24h.get('total_evaluated', 0),
            "avg_pipeline_health_24h": stats_24h.get('avg_pipeline_health'),
            "overall_health": health.get('overall_health'),
            "status": "active" if stats_24h.get('total_evaluated', 0) > 0 else "waiting_for_queries",
            "issues_detected": len(health.get('issues', [])),
            "background_execution": True,
            "zero_user_latency": True
        }
        
    except Exception as e:
        logger.error(f"[EVAL_API] Failed to get status: {e}")
        return {
            "evaluation_enabled": False,
            "error": str(e)
        }


@router.get("/trends")
async def get_evaluation_trends(
    hours: int = Query(default=168, ge=24, le=720, description="Time window in hours (min: 24, max: 30 days)"),
    bucket_hours: int = Query(default=24, ge=1, le=168, description="Bucket size in hours"),
    current_user: UserPayload = Depends(get_current_user)
):
    """
    Get evaluation trends over time.
    """
    try:
        collection = db['comprehensive_evaluations']
        
        # FIX 1: Use timezone-aware UTC current time
        # This matches the format of the data you stored earlier
        now_utc = datetime.now(timezone.utc)
        time_threshold = now_utc - timedelta(hours=hours)
        
        # Get all evaluations in window
        cursor = collection.find({
            'evaluated_at': {'$gte': time_threshold.isoformat()}
        }).sort('evaluated_at', 1)
        
        evaluations = await cursor.to_list(length=None)
        
        if not evaluations:
            return JSONResponse(content={
                'buckets': [],
                'message': f'No evaluations found in last {hours} hours'
            })
        
        # Bucket evaluations
        buckets = defaultdict(list)
        
        for eval_doc in evaluations:
            # FIX 2: Parse ISO string safely
            eval_time = datetime.fromisoformat(eval_doc['evaluated_at'])
            
            # Ensure the parsed time is timezone-aware (UTC) to match our logic
            if eval_time.tzinfo is None:
                eval_time = eval_time.replace(tzinfo=timezone.utc)
            
            # Calculate bucket key (floor to nearest bucket_hours)
            # We reset minutes/seconds first
            bucket_base = eval_time.replace(minute=0, second=0, microsecond=0)
            
            # Math to floor the hour to the bucket size (e.g., hour 15 with bucket 6 -> hour 12)
            floored_hour = (bucket_base.hour // bucket_hours) * bucket_hours
            bucket_key = bucket_base.replace(hour=floored_hour)
            
            buckets[bucket_key].append(eval_doc)
        
        # Aggregate by bucket
        trend_data = []
        for bucket_time in sorted(buckets.keys()):
            bucket_evals = buckets[bucket_time]
            
            # Calculate averages (Safely getting 0 if key is missing)
            pipeline_healths = [e.get('pipeline_health_score', 0) for e in bucket_evals]
            
            # Helper to safely extract nested scores
            def get_score(doc, stage, field):
                stage_data = doc.get(stage)
                return stage_data.get(field, 0) if stage_data else 0

            query_scores = [get_score(e, 'query_processing', 'expansion_quality_score') for e in bucket_evals]
            retrieval_hits = [get_score(e, 'retrieval', 'estimated_hit_rate_at_3') for e in bucket_evals]
            reranking_scores = [get_score(e, 'reranking', 'reranking_effectiveness_score') for e in bucket_evals]
            context_scores = [get_score(e, 'context', 'context_quality_score') for e in bucket_evals]
            generation_scores = [get_score(e, 'generation', 'overall_score') for e in bucket_evals]
            
            trend_data.append({
                'timestamp': bucket_time.isoformat(),
                'num_evaluations': len(bucket_evals),
                'avg_pipeline_health': sum(pipeline_healths) / len(pipeline_healths) if pipeline_healths else 0,
                'avg_query_quality': sum(query_scores) / len(query_scores) if query_scores else 0,
                'avg_retrieval_hit_rate': sum(retrieval_hits) / len(retrieval_hits) if retrieval_hits else 0,
                'avg_reranking_effectiveness': sum(reranking_scores) / len(reranking_scores) if reranking_scores else 0,
                'avg_context_quality': sum(context_scores) / len(context_scores) if context_scores else 0,
                'avg_generation_score': sum(generation_scores) / len(generation_scores) if generation_scores else 0
            })
        
        return JSONResponse(content={
            'time_window_hours': hours,
            'bucket_hours': bucket_hours,
            'num_buckets': len(trend_data),
            'buckets': trend_data
        })
        
    except Exception as e:
        logger.error(f"[EVAL_API] Failed to get trends: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve evaluation trends: {str(e)}"
        )