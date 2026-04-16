# app/monitoring/drift/thresholds.py

THRESHOLDS = {
    "hallucination_avg": {
        "warn": 20,      # early signal
        "high": 40,
        "critical": 60
    },

    # Retrieval is unstable early → forgiving
    "retrieval_avg": {
        "warn": 40,
        "high": 30,
        "critical": 20
    },

    # This is a RATE (0–1), not score
    "knowledge_gap_rate": {
        "warn": 0.20,     # learning phase
        "high": 0.40,
        "critical": 0.60
    },
    
    "knowledge_coverage_gap_rate": {
        "warn": 0.05,
        "high": 0.15,
        "critical": 0.30
    },

    # Toxicity stays strict
    "query_toxicity_avg": {
        "warn": 10,
        "high": 20,
        "critical": 30
    },

    "response_toxicity_avg": {
        "warn": 5,
        "high": 10,
        "critical": 20
    }
}
