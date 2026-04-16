#app/monitoring/aggregation/evaluation_aggregator.py
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from app.database import db

async def aggregate_evaluations(since: datetime, until: datetime):
    pipeline = [
        {
            "$addFields": {
                "evaluated_at_dt": {
                    "$convert": {
                        "input": "$evaluated_at",
                        "to": "date",
                        "onError": None,
                        "onNull": None
                    }
                }
            }
        },
        {
            "$match": {
                "evaluated_at_dt": {
                    "$gte": since,
                    "$lte": until
                }
            }
        },
        {
            "$group": {
                "_id": None,
                "faithfulness_avg": {"$avg": "$generation.faithfulness_score"},
                "hallucination_avg": {"$avg": "$generation.hallucination_score"},
                "overall_score_avg": {"$avg": "$generation.overall_score"},
                "retrieval_avg": {"$avg": "$retrieval.avg_score"},
                "query_toxicity_avg": {"$avg": "$generation.query_toxicity_score"},
                "response_toxicity_avg": {"$avg": "$generation.response_toxicity_score"},
                "pipeline_health_avg": {"$avg": "$pipeline_health_score"},
                "total_queries": {"$sum": 1},
                "knowledge_gap_rate": {
                    "$avg": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$gte": ["$retrieval.avg_score", 0.6]},
                                    {"$gt": [{"$size": "$retrieval.top_k_docs"}, 0]},
                                    {"$gt": ["$generation.hallucination_score", 0.4]}
                                ]
                            },
                            1,
                            0
                        ]
                    }
                },
                "knowledge_coverage_gap_rate": {
                    "$avg": {
                        "$cond": [
                            {
                                "$or": [
                                    {"$lt": ["$retrieval.avg_score", 0.4]},
                                    {"$eq": [{"$size": "$retrieval.top_k_docs"}, 0]}
                                ]
                            },
                            1,
                            0
                        ]
                    }
                }
            }
        },
        {"$project": {"_id": 0}}
    ]

    result = await db["comprehensive_evaluations"].aggregate(pipeline).to_list(1)
    return result[0] if result else {}

