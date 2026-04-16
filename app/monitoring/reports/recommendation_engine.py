# app/monitoring/reports/recommendation_engine.py


def generate_recommendations(drifts: list) -> list:
    recs = []

    for drift in drifts:
        dtype = drift["drift_type"]
        severity = drift["severity"]
        scope = drift["scope_pct"]

        if dtype == "HALLUCINATION" and severity in ["HIGH", "CRITICAL"]:
            recs.append(
                "Reduce model temperature, enforce stricter answer grounding, and "
                "validate citations against retrieved documents."
            )

        if dtype == "RETRIEVAL" and scope > 30:
            recs.append(
                "Re-evaluate embedding quality, chunk size, and vector similarity thresholds."
            )

        if dtype == "KNOWLEDGE_GAP":
            recs.append(
                "Expand the knowledge base by adding documents for frequently unanswered queries."
            )
        
        if dtype == "KNOWLEDGE_COVERAGE_GAP":
            recs.append(
                "Ingest missing documents, expand the knowledge base, and "
                "analyze unanswered queries to improve coverage."
            )

        if dtype == "QUERY_TOXICITY":
            recs.append(
                "Apply stricter input moderation, rate limiting, or user guidance prompts."
            )

        if dtype == "RESPONSE_TOXICITY":
            recs.append(
                "Introduce output moderation layers and tighten system safety prompts."
            )

    return list(set(recs))  # remove duplicates
