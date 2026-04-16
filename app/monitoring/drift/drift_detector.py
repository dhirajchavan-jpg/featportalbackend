#app/monitoring/drift/drift_detector.py
from app.monitoring.drift.taxonomy import DRIFT_TAXONOMY
from app.monitoring.drift.thresholds import THRESHOLDS
from app.monitoring.drift.severity import resolve_severity
from app.monitoring.drift.confidence import calculate_confidence
from app.monitoring.drift.attribution import attribute_drift
from app.monitoring.drift.status import resolve_status
from datetime import datetime


async def detect_drifts(metrics: dict,since: datetime,until: datetime,report_reference_id: str) -> list:
    drifts = []
    total_queries = metrics.get("total_queries", 0)

    DRIFT_MAP = {
        "HALLUCINATION": ("hallucination_avg", THRESHOLDS["hallucination_avg"]),
        "RETRIEVAL": ("retrieval_avg", THRESHOLDS["retrieval_avg"]),
        "KNOWLEDGE_GAP": ("knowledge_gap_rate", THRESHOLDS["knowledge_gap_rate"]),
        "KNOWLEDGE_COVERAGE_GAP": ("knowledge_coverage_gap_rate",THRESHOLDS["knowledge_coverage_gap_rate"]),
        "QUERY_TOXICITY": ("query_toxicity_avg", THRESHOLDS["query_toxicity_avg"]),
        "RESPONSE_TOXICITY": ("response_toxicity_avg", THRESHOLDS["response_toxicity_avg"]),
    }

    for drift_type, (metric_key, threshold_cfg) in DRIFT_MAP.items():
        value = metrics.get(metric_key, 0)

        severity = resolve_severity(value, threshold_cfg)
        status = resolve_status(severity) if severity else "OK"

        # NO DRIFT
        if not severity:
            drifts.append({
                "drift_type": drift_type,
                "status": "OK",
                "severity": "NONE",
                "risk_categories": DRIFT_TAXONOMY[drift_type]["risk_categories"],
                "impact_areas": DRIFT_TAXONOMY[drift_type]["impact_areas"],
                "confidence": "HIGH",
                "scope_pct": 0,
                "threshold": None,
                "observed_value": value,
                "affected_queries": {
                    "count": 0,
                    "total": total_queries,
                    "top_examples": [],
                    "note": "No drift detected within configured thresholds"
                }
            })
            continue

        attribution = await attribute_drift(
            drift_type=drift_type,
            threshold=threshold_cfg[severity.lower()],
            since=since,
            until=until,
            report_reference_id=report_reference_id
        )


        scope_pct = (
            (attribution["count"] / total_queries) * 100
            if total_queries else 0
        )

        drifts.append({
            "drift_type": drift_type,
            "status": status,
            "severity": severity,
            "risk_categories": DRIFT_TAXONOMY[drift_type]["risk_categories"],
            "impact_areas": DRIFT_TAXONOMY[drift_type]["impact_areas"],
            "confidence": calculate_confidence(scope_pct, total_queries),
            "scope_pct": round(scope_pct, 2),
            "threshold": threshold_cfg[severity.lower()],
            "observed_value": value,
            "affected_queries": attribution
        })

    return drifts

