# app/monitoring/repositories/drift_query_repository.py

from app.database import db

async def get_drift_queries(
    report_reference_id: str,
):
    refs_cursor = db["drift_query_references"].find(
        {"report_reference_id": report_reference_id}
    )

    refs = []
    async for ref in refs_cursor:
        refs.append(ref)

    if not refs:
        return {
            "report_reference_id": report_reference_id,
            "total": 0,
            "items": []
        }

    items = []

    for ref in refs:
        field = ref["query_filter"]["field"]
        operator = ref["query_filter"]["operator"]
        threshold = ref["query_filter"]["threshold"]
        since = ref["since"]
        until = ref["until"]

        query = {
            "$expr": {
                "$and": [
                    {"$gte": [{"$toDate": "$evaluated_at"}, since]},
                    {"$lte": [{"$toDate": "$evaluated_at"}, until]},
                    {
                        operator: [
                            f"${field}",
                            threshold
                        ]
                    }
                ]
            }
        }

        cursor = db["comprehensive_evaluations"].find(
            query,
            {"query": 1, field: 1, "evaluated_at": 1}
        )

        async for doc in cursor:
            items.append({
                "query": doc.get("query"),
                "score": doc.get(field.split(".")[0], {}).get(field.split(".")[1]),
                "evaluated_at": doc.get("evaluated_at"),
                "drift_type": ref["drift_type"]
            })

    return {
        "report_reference_id": report_reference_id,
        "total": len(items),
        "items": items
    }
