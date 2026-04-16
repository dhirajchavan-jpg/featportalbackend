import phoenix as px
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Initialize Phoenix client
client = px.Client(endpoint="http://localhost:6006")

def push_evaluation_to_phoenix(evaluation: dict):
    """
    Push evaluation to Phoenix using the modern DataFrame API.
    """
    try:
        # 1. Flatten the data into a structure suitable for a DataFrame
        eval_record = {
            # 'span_id' is CRITICAL to link this evaluation to the trace
            "span_id": evaluation.get("span_id"),
            
            # Context (Input/Output) - Optional but good for debugging
            "input": evaluation.get("input"),
            "output": evaluation.get("output"),
            
            # Metrics
            "hallucination_label": evaluation.get("hallucination_label"),
            "hallucination_score": evaluation.get("hallucination_score"),
            "qa_correctness_label": evaluation.get("qa_correctness_label"),
            "qa_correctness_score": evaluation.get("qa_correctness_score"),
            
            # Metadata
            "model": evaluation.get("model"),
            "timestamp": evaluation.get("timestamp"),
        }

        # 2. Create a Pandas DataFrame 
        df = pd.DataFrame([eval_record])

        # 3. Log the evaluation
        # CHANGED: 'dataframe' -> 'evaluations'
        client.log_evaluations(evaluations=df)

        logger.info("[PHOENIX] Evaluation pushed successfully via DataFrame")

    except Exception as e:
        logger.error(f"[PHOENIX] Push failed: {e}", exc_info=True)