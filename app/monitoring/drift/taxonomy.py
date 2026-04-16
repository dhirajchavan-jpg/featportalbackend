# app/monitoring/drift/taxonomy.py
DRIFT_TAXONOMY = {
    "HALLUCINATION": {
        "risk_categories": ["QUALITY", "TRUST"],
        "impact_areas": ["ACCURACY", "COMPLIANCE"]
    },
    "RETRIEVAL": {
        "risk_categories": ["DATA", "PIPELINE"],
        "impact_areas": ["ACCURACY"]
    },
    "KNOWLEDGE_GAP": {
        "risk_categories": ["DATA", "KNOWLEDGE"],
        "impact_areas": ["ACCURACY", "TRUST"]
    },
    "KNOWLEDGE_COVERAGE_GAP": {
        "risk_categories": ["DATA", "COVERAGE"],
        "impact_areas": ["ACCURACY", "USER_EXPERIENCE"]
    },
    "QUERY_TOXICITY": {
        "risk_categories": ["SAFETY", "USER_BEHAVIOR"],
        "impact_areas": ["COMPLIANCE"]
    },
    "RESPONSE_TOXICITY": {
        "risk_categories": ["SAFETY"],
        "impact_areas": ["COMPLIANCE", "TRUST"]
    }
}
