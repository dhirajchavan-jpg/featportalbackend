# app/core/phoenix_utils.py
import time
import json
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from app.utils.logger import setup_logger
from app.services.evaluation.phoenix_publisher import push_evaluation_to_phoenix


logger = setup_logger()

# ============================================================================
# CONFIGURATION
# ============================================================================
OLLAMA_BASE_URL = "http://localhost:11434"
EVAL_MODEL = "phi3"

def evaluate_rag_interaction(
    query: str,
    response: str,
    retrieved_docs: list,
    span_id: str
):
    """
    Evaluate RAG interaction inline (not as separate span).
    This will make evaluations appear as attributes in the main trace.
    """
    
    try:
        logger.info(f"[EVAL] ")
        logger.info(f"[EVAL] Starting evaluation for Span: {span_id}")
        start_time = time.time()
        
        # ====================================================================
        # 1. PREPARE CONTEXT
        # ====================================================================
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5]):
            if hasattr(doc, 'page_content'):
                text = doc.page_content
            elif isinstance(doc, dict):
                text = doc.get('page_content', '')
            else:
                text = str(doc)
            
            context_parts.append(text[:500])
        
        context_text = "\n\n".join(context_parts)
        
        if not context_text.strip():
            logger.warning(f"[EVAL] No context available")
            return
        
        # ====================================================================
        # 2. RUN EVALUATIONS
        # ====================================================================
        
        # Hallucination Check
        hall_result = _evaluate_hallucination(query, response, context_text)
        logger.info(f"[EVAL]  Hallucination: {hall_result['label'].upper()}")
        
        # QA Correctness Check
        qa_result = _evaluate_qa_correctness(query, response, context_text)
        logger.info(f"[EVAL]  QA Correctness: {qa_result['label'].upper()}")
        
        # ====================================================================
        # 3. ATTACH TO CURRENT SPAN (NOT A CHILD SPAN!)
        # ====================================================================
        try:
            current_span = trace.get_current_span()
            
            if current_span and current_span.is_recording():
                # Add as top-level attributes (visible in columns)
                current_span.set_attribute("hallucination", hall_result['label'])
                current_span.set_attribute("hallucination_score", hall_result['score'])
                current_span.set_attribute("qa_correctness", qa_result['label'])
                current_span.set_attribute("qa_correctness_score", qa_result['score'])
                
                # Add detailed attributes (visible in trace details)
                current_span.set_attribute("eval.hallucination.label", hall_result['label'])
                current_span.set_attribute("eval.hallucination.score", hall_result['score'])
                current_span.set_attribute("eval.hallucination.explanation", hall_result['explanation'][:500])
                
                current_span.set_attribute("eval.qa_correctness.label", qa_result['label'])
                current_span.set_attribute("eval.qa_correctness.score", qa_result['score'])
                current_span.set_attribute("eval.qa_correctness.explanation", qa_result['explanation'][:500])
                
                # Add metadata
                current_span.set_attribute("eval.model", "ollama-phi3")
                current_span.set_attribute("eval.timestamp", int(time.time()))
                
                # Set span status based on quality
                if hall_result['label'] == "hallucinated":
                    current_span.set_status(Status(StatusCode.ERROR, "Hallucination detected"))
                elif qa_result['label'] == "incorrect":
                    current_span.set_status(Status(StatusCode.ERROR, "Incorrect answer"))
                else:
                    current_span.set_status(Status(StatusCode.OK))
                
                logger.info("[EVAL]  Added to current span (inline)")
            else:
                logger.warning("[EVAL] No active span to attach to")
                
        except Exception as e:
            logger.warning(f"[EVAL] Could not attach to span: {e}")
        
        # ====================================================================
        # 4. SAVE TO MONGODB
        # ====================================================================
        try:
            _save_to_mongodb(
                span_id=span_id,
                query=query,
                response=response,
                hallucination_label=hall_result['label'],
                hallucination_score=hall_result['score'],
                hallucination_explanation=hall_result['explanation'],
                qa_label=qa_result['label'],
                qa_score=qa_result['score'],
                qa_explanation=qa_result['explanation'],
                timestamp=time.time()
            )
            
            logger.info("[EVAL]  Saved to MongoDB")

            phoenix_eval_row = {
    "input": query,                 #  REQUIRED
    "output": response,             #  REQUIRED

    "span_id": span_id,

    "hallucination_label": hall_result["label"],
    "hallucination_score": hall_result["score"],

    "qa_correctness_label": qa_result["label"],
    "qa_correctness_score": qa_result["score"],

    "eval_model": "ollama-phi3",
    "timestamp": int(time.time())
}

            push_evaluation_to_phoenix(phoenix_eval_row)


            
            

            
        except Exception as e:
            logger.warning(f"[EVAL] MongoDB save failed: {e}")
        
        # ====================================================================
        # 5. LOG SUMMARY
        # ====================================================================
        duration = time.time() - start_time
        logger.info(f"[EVAL] ")
        logger.info(f"[EVAL]  COMPLETE ({duration:.2f}s)")
        logger.info(f"[EVAL]   • Hallucination: {hall_result['label'].upper()}")
        logger.info(f"[EVAL]   • QA Correctness: {qa_result['label'].upper()}")
        logger.info(f"[EVAL]   • Attached inline to span")
        logger.info(f"[EVAL] ")
        
    except Exception as e:
        logger.error(f"[EVAL] CRITICAL ERROR: {e}", exc_info=True)


def _evaluate_hallucination(query: str, response: str, context: str) -> dict:
    """Evaluate if response contains hallucinations"""
    
    prompt = f"""You are an evaluator. Is this response FACTUAL or HALLUCINATED?

Context: {context[:700]}

Question: {query}
Response: {response}

Answer ONLY with JSON: {{"label": "factual"}} OR {{"label": "hallucinated"}}

Your answer:"""

    try:
        result = _call_ollama(prompt, max_tokens=50)
        parsed = _parse_eval(result)
        
        label = parsed.get("label", "unknown")
        score = 1.0 if label == "factual" else 0.0
        
        return {
            "label": label,
            "score": score,
            "explanation": f"Evaluated as {label}"
        }
    except Exception as e:
        logger.error(f"[EVAL] Hallucination failed: {e}")
        return {
            "label": "error",
            "score": 0.0,
            "explanation": f"Error: {str(e)[:100]}"
        }


def _evaluate_qa_correctness(query: str, response: str, context: str) -> dict:
    """Evaluate if response correctly answers the question"""
    
    prompt = f"""You are an evaluator. Is this answer CORRECT or INCORRECT?

Context: {context[:700]}

Question: {query}
Response: {response}

Answer ONLY with JSON: {{"label": "correct"}} OR {{"label": "incorrect"}}

Your answer:"""

    try:
        result = _call_ollama(prompt, max_tokens=50)
        parsed = _parse_eval(result)
        
        label = parsed.get("label", "unknown")
        score = 1.0 if label == "correct" else 0.0
        
        return {
            "label": label,
            "score": score,
            "explanation": f"Evaluated as {label}"
        }
    except Exception as e:
        logger.error(f"[EVAL] QA correctness failed: {e}")
        return {
            "label": "error",
            "score": 0.0,
            "explanation": f"Error: {str(e)[:100]}"
        }


def _save_to_mongodb(span_id, query, response, hallucination_label, hallucination_score, 
                     hallucination_explanation, qa_label, qa_score, qa_explanation, timestamp):
    """Save evaluations to MongoDB"""
    
    try:
        # from app.database import get_database
        from datetime import datetime
        
        # db = get_database()
        # evaluations_collection = db["evaluations"]

        from app.database import db
        evaluations_collection = db["live_evaluations"]

        
        evaluation_doc = {
            "span_id": span_id,
            "query": query,
            "response": response,
            "hallucination": {
                "label": hallucination_label,
                "score": hallucination_score,
                "explanation": hallucination_explanation
            },
            "qa_correctness": {
                "label": qa_label,
                "score": qa_score,
                "explanation": qa_explanation
            },
            "timestamp": timestamp,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        evaluations_collection.insert_one(evaluation_doc)
        
    except Exception as e:
        logger.error(f"[EVAL] MongoDB save failed: {e}")
        raise




def _call_ollama(prompt: str, max_tokens: int = 100) -> str:
    """Call Ollama API"""
    import requests
    
    payload = {
        "model": EVAL_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
        }
    }
    
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=60.0
    )
    
    if response.status_code == 200:
        return response.json().get("response", "").strip()
    
    raise Exception(f"Ollama returned {response.status_code}")


def _parse_eval(text: str) -> dict:
    """Parse evaluation response"""
    import re
    
    # Extract JSON
    try:
        json_match = re.search(r'\{.*?\}', text)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    # Fallback
    text_lower = text.lower()
    
    if "factual" in text_lower and "hallucinated" not in text_lower:
        return {"label": "factual"}
    elif "hallucinated" in text_lower:
        return {"label": "hallucinated"}
    elif "correct" in text_lower and "incorrect" not in text_lower:
        return {"label": "correct"}
    elif "incorrect" in text_lower:
        return {"label": "incorrect"}
    
    return {"label": "unknown"}


def test_ollama_connection():
    """Test Ollama connection"""
    import requests
    
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m.get("name") for m in response.json().get("models", [])]
            logger.info(f"[OLLAMA] Available: {models}")
            return any("phi3" in m for m in models)
        return False
    except:
        logger.error("[OLLAMA] Not accessible")
        return False