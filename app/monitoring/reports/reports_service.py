from datetime import datetime
from app.monitoring.aggregation.evaluation_aggregator import aggregate_evaluations
from app.monitoring.drift.drift_detector import detect_drifts
from app.monitoring.reports.narrative_generator import generate_narrative
from app.monitoring.reports.recommendation_engine import generate_recommendations
import app.monitoring.repositories.report_repository as rp
from datetime import timedelta
from zoneinfo import ZoneInfo
import uuid



REPORT_WINDOWS = {
    "DAILY": 1,
    "WEEKLY": 7,
    "MONTHLY": 30,
    "YEARLY": 365
}


async def generate_report(report_type: str):
    if report_type not in REPORT_WINDOWS:
        raise ValueError(f"Unsupported report type: {report_type}")

    days = REPORT_WINDOWS[report_type]
    
    report_reference_id = f"RPT-{report_type}-{uuid.uuid4().hex[:8]}"

    #  SINGLE SOURCE OF TRUTH
    IST = ZoneInfo("Asia/Kolkata")
    UTC = ZoneInfo("UTC")

    until_ist = datetime.now(IST)
    since_ist = until_ist - timedelta(days=days)

    # Mongo always UTC
    since = since_ist.astimezone(UTC)
    until = until_ist.astimezone(UTC)

    #  pass frozen window
    metrics = await aggregate_evaluations(since, until)
    drifts = await detect_drifts(metrics, since, until,report_reference_id)

    narrative = generate_narrative(drifts)

    recommendations = []
    if report_type in ["WEEKLY", "MONTHLY", "YEARLY"]:
        recommendations = generate_recommendations(drifts)

    report = {
        "report_type": report_type,
        "window_days": days,
        "since": since,          #  IMPORTANT
        "until": until,          #  IMPORTANT
        "metrics": metrics,
        "drifts": drifts,
        "narrative": narrative,
        "recommendations": recommendations,
        "generated_at": until,
        "report_reference_id":report_reference_id
    }

    await rp.ReportRepository.save(report)
    return report
