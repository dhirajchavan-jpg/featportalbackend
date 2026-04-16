from datetime import datetime
from app.database import db

DRIFT_ATTRIBUTION_RULES = {
    "HALLUCINATION": {
        "field": "generation.hallucination_score",
        "operator": "$gt",
        "definition_threshold": 60
    },
    "RETRIEVAL": {
        "field": "retrieval.avg_score",
        "operator": "$lt",
        "definition_threshold": 0.2
    },
    "KNOWLEDGE_GAP": {
        "field": "retrieval.avg_score",
        "operator": "$lt",
        "definition_threshold": 0.15
    },
    "KNOWLEDGE_COVERAGE_GAP": {
        "field": "retrieval.avg_score",
        "operator": "$lt",
        "definition_threshold": 0.4
    },
    "QUERY_TOXICITY": {
        "field": "generation.query_toxicity_score",
        "operator": "$gt",
        "definition_threshold": 20
    },
    "RESPONSE_TOXICITY": {
        "field": "generation.response_toxicity_score",
        "operator": "$gt",
        "definition_threshold": 10
    }
}


async def attribute_drift(
    drift_type: str,
    threshold: float,
    since: datetime,
    until: datetime,
    report_reference_id: str
):
    rule = DRIFT_ATTRIBUTION_RULES[drift_type]
    field = rule["field"]
    operator = rule["operator"]
    definition_threshold = rule["definition_threshold"]

    base_match = {
        "$expr": {
            "$and": [
                {"$gte": [{"$toDate": "$evaluated_at"}, since]},
                {"$lte": [{"$toDate": "$evaluated_at"}, until]},
                {
                    operator: [
                        f"${field}",
                        definition_threshold
                    ]
                }
            ]
        }
    }

    affected_count = await db["comprehensive_evaluations"].count_documents(base_match)

    total_count = await db["comprehensive_evaluations"].count_documents({
        "$expr": {
            "$and": [
                {"$gte": [{"$toDate": "$evaluated_at"}, since]},
                {"$lte": [{"$toDate": "$evaluated_at"}, until]}
            ]
        }
    })

    examples_cursor = db["comprehensive_evaluations"].find(
        base_match,
        {"query": 1, field: 1}
    )

    examples = []
    async for doc in examples_cursor:
        examples.append({
            "query": doc.get("query"),
            "score": doc.get(field.split(".")[0], {}).get(field.split(".")[1]),
            "reason": f"{drift_type} definition threshold breached"
        })

    
    await db["drift_query_references"].insert_one({
        "report_reference_id": report_reference_id,
        "drift_type": drift_type,
        "definition_threshold": definition_threshold,
        "severity_threshold": threshold,
        "since": since,
        "until": until,
        "created_at": datetime.utcnow(),
        "query_filter": {
            "field": field,
            "operator": operator,
            "threshold": definition_threshold
        }
    })

    return {
        "count": affected_count,
        "total": total_count,
        "top_examples": examples
    }
