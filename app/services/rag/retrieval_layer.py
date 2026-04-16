import json
import time
import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes
from openinference.semconv.trace import OpenInferenceSpanKindValues

from app.services.rag.sector_registry import get_available_sectors

# Retrieval Layer 4: Multi-Source Retrieval
from app.services.retrieval.query_processor import get_query_processor
from app.services.retrieval.bm25_retriever import get_bm25_retriever
from app.services.retrieval.vector_retriever import get_vector_retriever
from app.services.retrieval.rrf_fusion import get_rrf_fusion
from app.core.llm_provider import get_bge_reranker as get_reranker

# Setup tracing and logging
tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


def _should_rerank(fused_docs: List[Dict]) -> bool:
    """
    Only rerank if top results are close.
    Skip reranking if there's a clear winner (saves 80% of reranking time).
    """
    if len(fused_docs) < 2:
        return False
    
    scores = [d.get('rrf_score', 0) for d in fused_docs[:3]]
    if not scores:
        return False
    
    top = scores[0]
    second = scores[1] if len(scores) > 1 else 0
    gap = top - second
    
    # Only rerank if it's a close race (gap < 15%)
    should_rerank = gap < 0.15
    
    if not should_rerank:
        logger.info(f"[RERANK] Skipping - clear winner (gap: {gap:.2f})")
    
    return should_rerank


def _format_for_phoenix(docs: List[Dict]) -> List[Dict]:
    """Format documents for Phoenix telemetry."""
    phoenix_docs = []
    for doc in docs:
        safe_metadata = doc.get('metadata', {})
        if not isinstance(safe_metadata, dict):
            safe_metadata = {}
        
        safe_content = doc.get('content') or doc.get('page_content', "")
        if not isinstance(safe_content, str):
            safe_content = str(safe_content)
        
        raw_id = doc.get('id') or safe_metadata.get('id') or "unknown"
        
        phoenix_docs.append({
            "document.content": safe_content,
            "document.metadata": safe_metadata,
            "document.score": float(doc.get('rerank_score', 0)),
            "document.id": str(raw_id)
        })
    
    return phoenix_docs


async def _retrieve_by_sector_comparative(
    query: str,
    project_id: str,
    sectors: List[str],
    results_per_sector: int,
    excluded_files: Optional[List[str]] = None,
    # --- NEW ARGUMENTS ---
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    use_reranking: bool = True
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    """
     PARALLEL retrieval from all sectors at once.
    Applies weights to determine strategy (Vector vs Keyword vs Hybrid).
    Integrates Query Understanding for dynamic sector selection and filtering.
    """
    logger.info(f"[RETRIEVAL START] Query='{query}' | Project='{project_id}' | Sectors={sectors}")
    logger.info(f"[RETRIEVAL CONFIG] Results/Sector={results_per_sector} | DenseW={dense_weight} | SparseW={sparse_weight} | Rerank={use_reranking}")

    with tracer.start_as_current_span(
        "custom_retriever_fusion",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
            SpanAttributes.INPUT_VALUE: query,
            "retrieval.dense_weight": dense_weight,
            "retrieval.sparse_weight": sparse_weight,
            "retrieval.reranking": use_reranking
        }
    ) as span:
        logger.info("[RETRIEVAL STEP] Initializing Retriever Services...")
        query_processor = get_query_processor()
        bm25_retriever = get_bm25_retriever()
        vector_retriever = get_vector_retriever()
        rrf = get_rrf_fusion()
        reranker = get_reranker()
        logger.info("[RETRIEVAL STEP] Services Initialized.")
        
        # --- NEW: Extract Query Understanding Once ---
        # logger.info("[RETRIEVAL STEP] Extracting Query Understanding (Intent & Filters)...")
        # query_understanding = await query_processor.extract_query_understanding(query)
        # logger.info(f"[RETRIEVAL STEP] Query Understanding Result: {query_understanding.dict(exclude_none=True)}")
        
        # # Add to span attributes for telemetry
        # span.set_attribute("retrieval.query_understanding", query_understanding.json())

        # # Auto-add sector if authority is detected with high confidence (e.g., User asks "RBI guidelines" -> Add "RBI" sector)
        # if query_understanding.authority and query_understanding.confidence > 0.8:
        #     if query_understanding.authority not in sectors:
        #         logger.info(f"[Auto-Sector] Dynamically adding sector '{query_understanding.authority}' based on intent.")
        #         sectors.append(query_understanding.authority)
        
        #  PARALLEL: Process all sectors at once!
        async def process_single_sector(sector: Optional[str]) -> Optional[Dict[str, Any]]:
            display_name = sector if sector else "Project-Only (Global)"
            logger.info(f"[PARALLEL TASK START] Processing sector: {display_name}")
            sector_start = time.time()
            
            try:
                # 1. Build Filter (NOW WITH QUERY UNDERSTANDING)
                logger.debug(f"[{display_name}] Building search filters...")
                current_sectors = [sector] if sector else None
                
                # Pass query_understanding to build_search_filter to apply specific metadata filters (e.g. Doc Type)
                search_filter = query_processor.build_search_filter(
                    project_id=project_id,
                    sectors=current_sectors,
                    excluded_files=excluded_files,
                    # query_understanding=query_understanding # <--- NEW: Passed here
                )
                logger.debug(f"[{display_name}] Filter built: Sources={search_filter.sources}, QU_Filters={bool(search_filter.query_understanding)}")
                
                targets = [project_id]
                if sector: targets.append(sector)
                
                # 2.  SELECT STRATEGY BASED ON WEIGHTS
                bm25_results = []
                vector_results = []
                
                tasks = []
                run_bm25 = sparse_weight > 0.0
                run_vector = dense_weight > 0.0
                
                logger.info(f"[{display_name}] Strategy Selection -> BM25: {run_bm25}, Vector: {run_vector}")

                if run_bm25:
                    logger.debug(f"[{display_name}] Queuing BM25 Search task...")
                    tasks.append(bm25_retriever.retrieve(
                        query=query,
                        search_filter=search_filter,
                        sources_to_search=targets
                    ))
                
                if run_vector:
                    logger.debug(f"[{display_name}] Queuing Vector Search task...")
                    tasks.append(vector_retriever.retrieve(
                        query=query,
                        search_filter=search_filter
                    ))
                
                # Execute gathered tasks
                if not tasks:
                    logger.warning(f"[{display_name}] No search tasks scheduled (Weights are 0?). Returning None.")
                    return None

                logger.info(f"[{display_name}] Awaiting {len(tasks)} search tasks...")
                results = await asyncio.gather(*tasks)
                logger.info(f"[{display_name}] Search tasks completed.")
                
                # Unpack results based on what ran
                idx = 0
                if run_bm25:
                    bm25_results = results[idx]
                    idx += 1
                if run_vector:
                    vector_results = results[idx]
                
                logger.info(f"[{display_name}] Raw Results -> BM25: {len(bm25_results)} docs, Vector: {len(vector_results)} docs")

                if not bm25_results and not vector_results:
                    logger.info(f"[{display_name}] No results found in either retriever. Returning None.")
                    return None
                
                # 3. Fusion Logic
                logger.debug(f"[{display_name}] Starting Fusion Logic...")
                if run_bm25 and run_vector:
                    # Hybrid: Use RRF
                    #  FIX: Allow getting up to RRF_TOP_K candidates (e.g. 50) 
                    # so the reranker has plenty of options to pick the top 10.
                    from app.config import settings # Ensure settings is imported
                    
                    fused = rrf.fuse(
                        bm25_results, 
                        vector_results, 
                        top_k=settings.RRF_TOP_K  # Use the config value (50)
                    )
                    logger.info(f"[{display_name}] RRF Fusion complete. Fused count: {len(fused)}")
                else:
                    # Single Strategy: Just take the results from the active one
                    logger.info(f"[{display_name}] Single Strategy used. Passthrough results.")
                    raw_docs = bm25_results if run_bm25 else vector_results
                    
                    #  FIX: Remove hard limit of 8 here too
                    fused = raw_docs[:settings.RRF_TOP_K] 
                    
                    for i, d in enumerate(fused):
                        d['rrf_score'] = d.get('score', 1.0 - (i*0.01))
                    logger.info(f"[{display_name}] Passthrough complete. Count: {len(fused)}")
                # Capture docs before reranking
                docs_before = [{
                    "page_content": doc.get('content', ''),
                    "metadata": doc.get('metadata', {}),
                    "rrf_score": doc.get('rrf_score', 0),
                    "relevance_score": doc.get('relevance_score', 0)
                } for doc in fused]
                
                # 4.  Smart Reranking (Conditional)
                rerank_start = time.time()
                rerank_time = 0
                top_chunks = []

                logger.debug(f"[{display_name}] Checking Reranking logic (Enabled={use_reranking})...")
                if use_reranking:
                    if _should_rerank(fused):
                        logger.info(f"[{display_name}] Reranking triggered for {len(fused)} candidates...")
                        top_chunks = await reranker.arerank(query=query, documents=fused, top_k=results_per_sector)
                        rerank_time = (time.time() - rerank_start) * 1000
                        logger.info(f"[{display_name}] Reranking done in {rerank_time:.0f}ms. Top chunks: {len(top_chunks)}")
                    else:
                        logger.info(f"[{display_name}] Skipped rerank (Optimization/Clear Winner). Taking top {results_per_sector}.")
                        top_chunks = fused[:results_per_sector]
                else:
                    logger.info(f"[{display_name}] Reranking DISABLED by config. Taking top {results_per_sector}.")
                    top_chunks = fused[:results_per_sector]
                
                # Format for App Logic
                formatted_chunks = []

                for doc in top_chunks:
                    metadata = doc.get("metadata", {}) or {}

                    file_id = (
                        metadata.get("file_id")
                        or metadata.get("id")
                        or metadata.get("file_uuid")
                    )

                    file_view_url = None
                    if file_id:
                        # RELATIVE URL — no port, no config
                        file_view_url = f"/files/view/{file_id}"

                    formatted_chunks.append({
                        "page_content": doc.get("content") or doc.get("page_content"),
                        "metadata": {
                            **metadata,
                            "file_view_url": file_view_url
                        },
                        "relevance_score": doc.get("rerank_score", doc.get("rrf_score", 0)),
                        "rerank_score": doc.get("rerank_score", doc.get("rrf_score", 0))
                    })

                
                sector_time = (time.time() - sector_start) * 1000
                logger.info(f"[PARALLEL TASK END] {display_name} finished in {sector_time:.0f}ms. Found {len(formatted_chunks)} chunks.")
                
                return {
                    'sector': display_name,
                    'chunks_found': len(top_chunks),
                    'chunks': formatted_chunks,
                    'docs_before': docs_before,
                    'rerank_time': rerank_time,
                    'phoenix_docs': _format_for_phoenix(top_chunks),
                    'sector_time_ms': sector_time
                }
                
            except Exception as e:
                logger.error(f"[PARALLEL ERROR] {display_name} failed: {e}", exc_info=True)
                return None
        
        #  Execute all sectors + Project-Only in PARALLEL
        logger.info("[RETRIEVAL STEP] Preparing parallel sector tasks...")
        tasks = [process_single_sector(sector) for sector in sectors]
        tasks.append(process_single_sector(None)) # Project Only
        
        logger.info(f"[RETRIEVAL STEP] Launching {len(tasks)} parallel sector searches (Sectors + Global)...")
        parallel_start = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        parallel_time = (time.time() - parallel_start) * 1000
        logger.info(f"[RETRIEVAL STEP] All parallel tasks finished in {parallel_time:.0f}ms.")
        
        # Aggregate Results
        logger.info("[RETRIEVAL STEP] Aggregating results...")
        sector_results = []
        all_docs_before = []
        total_reranking_time = 0
        all_phoenix_docs = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed with exception: {result}")
                continue
                
            if result and result['chunks_found'] > 0:
                logger.debug(f"Aggregation: Adding {result['chunks_found']} chunks from '{result['sector']}'")
                sector_results.append({
                    'sector': result['sector'],
                    'chunks_found': result['chunks_found'],
                    'chunks': result['chunks']
                })
                all_docs_before.extend(result['docs_before'])
                total_reranking_time += result['rerank_time']
                all_phoenix_docs.extend(result['phoenix_docs'])
            else:
                logger.debug(f"Aggregation: Task {i} returned empty or None.")
        
        # Log to Phoenix
        if all_phoenix_docs:
            logger.debug(f"[RETRIEVAL TELEMETRY] Logging {len(all_phoenix_docs)} docs to Phoenix.")
            span.set_attribute(SpanAttributes.RETRIEVAL_DOCUMENTS, json.dumps(all_phoenix_docs))
        
        span.set_attribute("retrieval.total_sectors", len(sector_results))
        span.set_attribute("retrieval.total_reranking_time_ms", total_reranking_time)
        
        logger.info(f"[RETRIEVAL COMPLETE] Returning {len(sector_results)} sector groups. Total Rerank Time: {total_reranking_time:.0f}ms. Total Parallel Time: {parallel_time:.0f}ms")
        
        return sector_results, all_docs_before, total_reranking_time