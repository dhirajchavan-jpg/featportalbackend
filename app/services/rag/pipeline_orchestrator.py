# app/services/rag/pipeline_orchestrator.py

import asyncio
import time
import json
import logging
import re
from typing import Optional, List, Any, Dict

from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes
from openinference.semconv.trace import OpenInferenceSpanKindValues

from app.database import cache_collection
from app.dependencies import UserPayload
from app.middleware.prompt_validation import get_prompt_validator
from app.middleware.moderation import is_compliance_related
from app.config import settings 

from app.services.rag.history_and_cache import (
    _save_to_history,
    _cache_result,
    _build_cache_key,
    _get_chat_history,
)
from app.services.rag.greeting_handler import _handle_greeting
from app.services.rag.retrieval_layer import _retrieve_by_sector_comparative
from app.services.rag.evaluation_runner import _run_comprehensive_evaluation_background

# Layer 3: Embedding & Indexing
from app.core.llm_provider import cleanup_vram, get_llm ,get_simple_llm

# Layer 4: Multi-Source Retrieval
from app.services.retrieval.query_processor import get_query_processor
from app.services.retrieval.model_router import get_model_router

# Context Management
from app.services.context_manager import get_context_manager

# Setup tracing and logging
tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)

PER_SECTOR_CHUNK_LIMIT = 5


async def _normalize_response(text: str, llm_instance: Any) -> str:
    """
    Parses the LLM response to detect and fix Chinese output.
    - Checks for Chinese characters.
    - Retries normalization up to 2 times.
    - If it fails after 2 tries, returns 'Please ask your question again.'
    """
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    
    # 1. Fast Check: If no Chinese, return immediately
    if not chinese_pattern.search(text):
        return text

    logger.warning("[RESPONSE PARSER] Chinese characters detected. Starting normalization (Limit: 2 attempts)...")
    
    corrector_llm = get_simple_llm()

    current_text = text
    max_retries = 2
    
    for i in range(max_retries):
        try:
            logger.info(f"[RESPONSE PARSER] Normalization Attempt {i+1}/{max_retries}...")
            
            repair_messages = [
                {
                    "role": "system", 
                    "content": "You are a translator. The previous answer was accidentally generated in Chinese. Translate it to professional English maintaining all formatting. Output ONLY the English translation."
                },
                {
                    "role": "user", 
                    "content": f"Translate this text to English:\n\n{current_text}"
                }
            ]
            
            # Invoke LLM
            response = await corrector_llm.ainvoke(repair_messages)
            normalized_text = response.content if hasattr(response, 'content') else str(response)
            
            # Check if the new text is clean (No Chinese)
            if not chinese_pattern.search(normalized_text):
                logger.info("[RESPONSE PARSER] Normalization successful.")
                return normalized_text
            
            # If still Chinese, update text and try again (loop continues)
            current_text = normalized_text
            
        except Exception as e:
            logger.error(f"[RESPONSE PARSER] Attempt {i+1} failed: {e}")
            continue # Try next attempt if available

    # 2. Failure Case: If loop finishes and text is still Chinese
    logger.error("[RESPONSE PARSER] Failed to normalize after 2 attempts. Returning fallback message.")
    return "Please ask your question again."

async def query_rag_pipeline(
    query: str,
    current_user: UserPayload,
    project_id: str,
    sectors: Optional[List[str]] = None,
    comparative_mode: bool = True,
    results_per_sector: int = PER_SECTOR_CHUNK_LIMIT,
    excluded_files: Optional[List[str]] = None,
    style: Optional[str] = "Detailed",
    top_k: int = 5,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    use_reranking: bool = True,
    sector: Optional[str] = None,
    background_tasks: Optional[Any] = None,
    ai_config: Optional[Dict[str, Any]] = None ,
    force_sync_evaluation: bool = False,
    skip_evaluation: bool = False
):
    """
    Complete multi-source query pipeline with DYNAMIC CONFIGURATION.
    """
    # <--- ADDED: Normalize query to lowercase immediately
    query = query.strip().lower()

    logger.info(f" [INIT] Pipeline Started | Project: {project_id} | User: {current_user.user_id}")
    history_limit = 5
    results_per_sector = PER_SECTOR_CHUNK_LIMIT
    
    # --- 0. APPLY CONFIGURATION OVERRIDES ---
    if ai_config:
        logger.info(f" [CONFIG] Loading Custom AI Configuration: {json.dumps(ai_config, default=str)}")
        
        # 1. Retrieval Depth
        if "retrieval_depth" in ai_config:
            top_k = ai_config["retrieval_depth"]
            logger.info(
                f" Top-K requested: {top_k} (Per Sector enforced to {results_per_sector})"
            )

        if "chat_history_limit" in ai_config:
            history_limit = ai_config["chat_history_limit"]
            logger.info(f" [CONFIG] Chat History Limit set to: {history_limit}")   

        # 2. Reranking
        if "enable_reranking" in ai_config:
            use_reranking = ai_config["enable_reranking"]
            logger.info(f" Reranking: {'ENABLED' if use_reranking else 'DISABLED'}")

        # 3. Search Strategy (Map to Weights)
        strategy = ai_config.get("search_strategy", "hybrid")
        if strategy == "vector":
            dense_weight = 1.0
            sparse_weight = 0.0
            logger.info("   Search Strategy: VECTOR ONLY (Dense=1.0, Sparse=0.0)")
        elif strategy == "keyword":
            dense_weight = 0.0
            sparse_weight = 1.0
            logger.info("   Search Strategy: KEYWORD ONLY (Dense=0.0, Sparse=1.0)")
        else: # hybrid
            dense_weight = 0.5
            sparse_weight = 0.5
            logger.info("  Search Strategy: HYBRID (Dense=0.5, Sparse=0.5)")
    else:
        logger.info("[CONFIG] Using System Defaults (No custom config found)")
    # ----------------------------------------

    with tracer.start_as_current_span(
        "rag_execution_pipeline",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            SpanAttributes.INPUT_VALUE: query,
            "user.id": current_user.user_id,
            "project.id": project_id,
            "config.strategy": ai_config.get("search_strategy", "default") if ai_config else "default"
        }
    ) as span:
        total_start = time.time()
        logger.info(f"\n{'='*70}")
        logger.info(f"QUERY: {query}")
        logger.info(f"{'='*70}\n")
        
        # ========== EVALUATION DATA CAPTURE ==========
        query_processing_start = 0
        query_processing_time_ms = 0
        retrieval_start = 0
        retrieval_time_ms = 0
        reranking_time_ms = 0
        generation_start = 0
        generation_time_ms = 0
        
        original_query = query 
        expanded_query = None
        docs_before_rerank = []
        docs_after_rerank = []
        assembled_context = ""
        context_tokens = 0
        llm_answer = ""
        # ============================================
        
        try:
            normalized_project_id = project_id.lower().strip()
            
            if sector and not sectors:
                sectors = [sector]
            
            # 1. SETUP SECTORS (Universal Search)
            if sectors is None or len(sectors) == 0:
                raise RuntimeError("Organization sector is required for HR single-sector retrieval")
            
            normalized_sectors = [s.strip().upper() for s in sectors if s and s.strip()]
            logger.info(f" [SECTORS] Active Sectors: {normalized_sectors}")
            
            # ================= CACHE CHECK =================
            cache_key = _build_cache_key(
                user_id=current_user.user_id,
                project_id=normalized_project_id,
                query=query,
                sectors=normalized_sectors,
                excluded_files=excluded_files,
                style=style  # <--- UPDATED: Pass style here
            )
            chat_id = f"user_{current_user.user_id}_project_{normalized_project_id}"
            
            logger.debug(f" [CACHE] Checking cache key: {cache_key}")
            existing_cache = await cache_collection.find_one({"cache_key": cache_key})
            if existing_cache:
                logger.info(" [CACHE] HIT! Returning cached response.")
                
                
                await _save_to_history(
                    chat_id=chat_id, 
                    user_id=current_user.user_id, 
                    query=query, 
                    project_id=normalized_project_id, 
                    sectors=normalized_sectors, 
                    answer=existing_cache['llm_answer'],
                    style=style # <--- UPDATED: Pass style to history
                )
                span.set_attribute("is_cache_hit", True)
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, existing_cache['llm_answer'])
                return {
                    "result": existing_cache['llm_answer'],
                    "source_documents": [],
                    "is_comparative": False,
                    "model_used": "cached_response",
                    "processing_time": time.time() - total_start,
                    "from_cache": True
                }
            logger.info("[CACHE] MISS. Proceeding to processing.")
            # ===============================================

            chat_id = f"user_{current_user.user_id}_project_{normalized_project_id}"
            history_list = await _get_chat_history(chat_id, limit=history_limit)

            logger.info(f" [HISTORY] Using {len(history_list)} messages from history (Limit: {history_limit})")
            
            # ========== STAGE 1: QUERY PROCESSING ==========
            logger.info(" [STEP 1] Starting Query Processing & Smart Greeting Check...")
            query_processing_start = time.time()
            
            query_processor = get_query_processor()
            processed_query = await query_processor.process_query(
                query=query, 
                chat_history=history_list
            )
            
            query_processing_time_ms = (time.time() - query_processing_start) * 1000
            logger.info(f"[STEP 1] Processing completed in {query_processing_time_ms:.0f}ms")
            
            # Gate 1: Invalid Check
            if not processed_query.get('is_valid', True):
                msg = "Please ask a valid compliance related question."
                logger.warning(" [VALIDATION] Query blocked as invalid/gibberish.")
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, msg)
                return { "result": msg, "source_documents": [], "is_comparative": False, "model_used": "validation_guard" }
            
            # Gate 2: Smart Greeting
            if processed_query.get('is_greeting', False):
                logger.info(f" [GREETING] Detected greeting intent.")
                return await _handle_greeting(
                    query, chat_id, current_user, normalized_project_id, 
                    normalized_sectors, total_start,
                    greeting_response=processed_query.get('greeting_response')
                )

            expanded_query = processed_query['expanded_query'] 
            query_metadata = processed_query['metadata']
            
            logger.info(f" [STEP 1] Expanded Query: '{expanded_query}'")
            
            # STEP 2: Model Routing (Logic Update)
            logger.info("[STEP 2] Routing Query to Appropriate Model...")
            router = get_model_router()
            
            # --- FIX START: Extract custom router model from config ---
            custom_router_model = None
            if ai_config and "router_model" in ai_config:
                custom_router_model = ai_config["router_model"]
                logger.info(f" [ROUTER] Using Custom Router Model: {custom_router_model}")
            # --- FIX END ---

            # 1. Determine Complexity (Simple vs Complex)
            # Pass the custom_router_model to the route_query method
            model_complexity = await router.route_query(
                query=expanded_query, 
                query_metadata=query_metadata,
                router_model_name=custom_router_model
            )
            logger.info(f" [STEP 2] Classified Complexity: {model_complexity.upper()}")
            
            # 2. Select Model Name from Config (Dynamic Selection)
            selected_model_name = settings.LLM_MODEL # Default Fallback
            
            if ai_config:
                if model_complexity == "simple":
                    selected_model_name = ai_config.get("simple_model", settings.LLM_MODEL_SIMPLE)
                    logger.info(f" Config selected SIMPLE model: {selected_model_name}")
                else:
                    selected_model_name = ai_config.get("complex_model", settings.LLM_MODEL_COMPLEX)
                    logger.info(f" Config selected COMPLEX model: {selected_model_name}")
            else:
                # Use Router's defaults if no config
                info = router.get_model_info(model_complexity)
                selected_model_name = info['model_name']
                logger.info(f"  Default Router selected model: {selected_model_name}")

            # 3. Get the LLM Instance
            selected_model = get_llm(selected_model_name)
            
            # ========== STAGE 3 & 4: RETRIEVAL & RERANKING ==========
            logger.info(f" [STEP 3] Starting Retrieval (Depth={top_k}, Strategy={ai_config.get('search_strategy', 'hybrid')})...")
            retrieval_start = time.time()
            
            sector_results, docs_before_rerank, reranking_time_ms = await _retrieve_by_sector_comparative(
                query=expanded_query,
                project_id=normalized_project_id,
                sectors=normalized_sectors,
                results_per_sector=results_per_sector,
                excluded_files=excluded_files,
                # Pass weights derived from config
                dense_weight=dense_weight, 
                sparse_weight=sparse_weight,
                use_reranking=use_reranking
            )
            
            retrieval_time_ms = (time.time() - retrieval_start) * 1000
            logger.info(f" [RETRIEVAL] Finished in {retrieval_time_ms:.0f}ms (Reranking: {reranking_time_ms:.0f}ms)")
            logger.info(f" [RETRIEVAL] Retrieved {len(docs_before_rerank)} docs initially.")

            logger.info(f"\n{'='*20} INITIAL CHUNKS (BEFORE RERANK) {'='*20}")
            for i, doc in enumerate(docs_before_rerank):
                content = doc.get('page_content', '') if isinstance(doc, dict) else getattr(doc, 'page_content', str(doc))
                meta = doc.get('metadata', {}) if isinstance(doc, dict) else getattr(doc, 'metadata', {})
                # Use 'rrf_score' which represents the retrieval score before reranking
                logger.info(f" [Chunk {i+1}] Source: {meta.get('source', 'unknown')} | Score: {doc.get('rrf_score', 'N/A')}")
                logger.info(f" Content: {content[:200]}...") # Logs first 200 chars to avoid flooding

            
            final_results = [s for s in sector_results if s['chunks_found'] > 0]
            
            if not final_results:
                logger.warning(" [RETRIEVAL] No relevant documents found across all sectors.")
                no_results_msg = "I searched your project documents but could not find relevant information."
                # UPDATED: Pass style to history
                await _save_to_history(chat_id, current_user.user_id, query, project_id, normalized_sectors, no_results_msg, style=style)
                return { "result": no_results_msg, "source_documents": [], "is_comparative": False, "model_used": "none" }

            docs_after_rerank = []
            for sector_result in final_results:
                docs_after_rerank.extend(sector_result['chunks'])
                
            logger.info(f" [RETRIEVAL] Final Document Count: {len(docs_after_rerank)}")
            logger.info(f"\n{'='*20} FINAL SELECTED CHUNKS (CONTEXT) {'='*20}")
            for i, doc in enumerate(docs_after_rerank):
                content = doc.get('page_content', '') if isinstance(doc, dict) else getattr(doc, 'page_content', str(doc))
                meta = doc.get('metadata', {}) if isinstance(doc, dict) else getattr(doc, 'metadata', {})
                rerank_score = doc.get('rerank_score', 'N/A')
    
                logger.info(f" [Final Chunk {i+1}] Source: {meta.get('source', 'unknown')} | Rerank Score: {rerank_score}")
                logger.info(f" Content: {content} \n") # Logs full content for the final context
            
            # ========== STAGE 5: CONTEXT ASSEMBLY ==========
            logger.info(" [STEP 5] Assembling Context for LLM...")
            context_mgr = get_context_manager()
            comparative_context = context_mgr.build_comparative_context(
                query=query,
                sector_results=final_results,
                chat_history=history_list,
                include_history=True,
                chat_history_limit=history_limit
            )
            
            assembled_context = comparative_context
            context_tokens = int(len(comparative_context.split()) / 0.75) 
            logger.info(f" [CONTEXT] Context Length: {len(assembled_context)} chars (~{context_tokens} tokens)")
            
            # STEP 5: Prompt
            messages = context_mgr.prepare_prompt(
                query=query,
                context=comparative_context,
                complexity=model_complexity ,
                style=style
            )
            
            
            # ========== STAGE 6: GENERATION ==========
            logger.info(f" [STEP 6] Generating Answer with {selected_model_name}...")
            generation_start = time.time()
            
            try:
                # 1. Try the Complex Model
                response = await selected_model.ainvoke(messages)
                llm_answer = response.content if hasattr(response, 'content') else str(response)
                llm_answer = await _normalize_response(llm_answer, selected_model)
                
                generation_time_ms = (time.time() - generation_start) * 1000
                logger.info(f" [GENERATION] Complete! Time: {generation_time_ms:.0f}ms")
                logger.info(f" [GENERATION] Output length: {len(llm_answer)} chars")
                
            except Exception as e:
                # 2. Handle Crash/Error
                logger.error(f" [GENERATION ERROR] {e}. Attempting cleanup and fallback...")
                
                # --- CRITICAL FIX STARTS HERE ---
                
                # A. Send signal to unload models
                cleanup_vram()
                
                # B. WAIT for the GPU to actually free the memory (Essential!)
                logger.info(" [MEMORY] Waiting 3s for VRAM release...")
                await asyncio.sleep(3.0) 
                
                # --- CRITICAL FIX ENDS HERE ---

                try:
                    # C. Try the Simple Model (Fallback)
                    # Use system default simple model as fallback
                    fallback_model = get_llm(settings.LLM_MODEL_SIMPLE)
                    logger.info(f" [FALLBACK] Switching to {settings.LLM_MODEL_SIMPLE}...")
                    
                    response = await fallback_model.ainvoke(messages)
                    llm_answer = response.content if hasattr(response, 'content') else str(response)
                    llm_answer = await _normalize_response(llm_answer, fallback_model)
                    selected_model_name += " (Fallback)"
                    generation_time_ms = (time.time() - generation_start) * 1000
                    logger.info(f" [FALLBACK GENERATION] Complete. Time: {generation_time_ms:.0f}ms")
                    
                except Exception as fallback_e:
                    # D. If even the simple model fails, fail gracefully
                    logger.critical(f" [CRITICAL] Fallback failed: {fallback_e}")
                    llm_answer = "I apologize, but the system is currently under heavy load. Please try your query again in a few moments."
                    selected_model_name = "ERROR"
            
            # Collect and Save
            all_docs = docs_after_rerank
            total_time = time.time() - total_start
            
            stats_dict = {
                "total_chunks_searched": len(all_docs),
                "chunks_retrieved": len(all_docs),
                "sources_queried": [normalized_project_id] + [s['sector'] for s in final_results],
                "retrieval_method": ai_config.get("search_strategy", "hybrid") if ai_config else "hybrid",
                "reranking_applied": use_reranking,
                "processing_time_ms": total_time * 1000
            }

            meta_dict = {
                "model_used": selected_model_name,
                "query_complexity": model_complexity,
                "from_cache": False
            }
            
            logger.info(" [SAVE] Saving conversation history and cache...")
            # UPDATED: Pass style parameter
            await _save_to_history(chat_id, current_user.user_id, query, project_id, normalized_sectors, llm_answer, source_documents=all_docs, retrieval_stats=stats_dict, meta_data=meta_dict, style=style)
           # 2. Conditional Cache: Check if response is a refusal or error
            # Define the patterns that should NOT be cached
            no_cache_patterns = [
                "system is currently under heavy load",          # Fallback error
                "could not find any valid results",              # Empty retrieval message
                "provided documents do not contain information", # LLM refusal message
                "I apologize, but the provided documents"        # Polite refusal variation
            ]
            
            should_cache = True
            for pattern in no_cache_patterns:
                if pattern.lower() in llm_answer.lower():
                    should_cache = False
                    logger.info(f" [CACHE] Skipping cache for negative response. Detected: '{pattern}'")
                    break

            # 🔴 VERIFICATION LOG: Add this line to prove we are skipping immediate cache

            logger.info(f" [CACHE] Immediate cache skipped. Delegating to background evaluation. (Valid: {should_cache})")            
            
            logger.info(f" [COMPLETE] Total Pipeline Time: {total_time:.2f}s")
            
            rag_result = {
                "result": llm_answer,
                "source_documents": all_docs,
                "comparative_analysis": final_results,
                "is_comparative": False,
                "sources_queried": [normalized_project_id] + [s['sector'] for s in final_results],
                "model_used": selected_model_name,
                "processing_time": total_time,
                "from_cache": False,
                "retrieval_stats": stats_dict, 
                "meta": meta_dict
            }
            
        except Exception as e:
            logger.error(f" [CRITICAL ERROR] Pipeline Failed: {e}", exc_info=True)
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Query processing failed: {e}")
    
    # ========== BACKGROUND EVALUATION ==========

    hallucination_threshold = 100.0
    if ai_config and "hallucination_threshold" in ai_config:
        hallucination_threshold = float(ai_config["hallucination_threshold"])


    eval_data = {
        'original_query': original_query,
        'expanded_query': expanded_query,
        'query_processing_time_ms': query_processing_time_ms,
        'docs_before_rerank': docs_before_rerank,
        'retrieval_time_ms': retrieval_time_ms - reranking_time_ms,
        'docs_after_rerank': docs_after_rerank,
        'reranking_time_ms': reranking_time_ms,
        'assembled_context': assembled_context,
        'context_tokens': context_tokens,
        'generated_response': llm_answer,
        'generation_time_ms': generation_time_ms,
        'user_id': current_user.user_id,
        'project_id': normalized_project_id,
        'model_config': ai_config, # Pass config for logging
        'hallucination_threshold': hallucination_threshold,
        'cache_key': cache_key if should_cache else None, 
        'sectors': normalized_sectors,
        'style': style  # <--- UPDATED: Include style for background worker
    }
    

    # --- NEW LOGIC STARTS HERE ---
    if skip_evaluation:
        logger.info("[EVAL] Skipping internal execution (Delegating to Worker).")
        # Attach the eval data to the result so the worker can retrieve it
        rag_result["_eval_data"] = eval_data 
        return rag_result
    # --- NEW LOGIC ENDS HERE ---
    if background_tasks:
        logger.info("[EVAL] Scheduling Background Evaluation (FastAPI BackgroundTasks)")
        background_tasks.add_task(_run_comprehensive_evaluation_background, eval_data=eval_data)
    else:
        logger.info("[EVAL] Scheduling Background Evaluation (AsyncIO Task)")
        asyncio.create_task(_run_comprehensive_evaluation_background(eval_data))
    
    return rag_result


