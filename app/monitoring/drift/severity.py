# app/monitoring/drift/severity.py
def resolve_severity(value: float, thresholds: dict) -> str | None:
    if value >= thresholds["critical"]:
        return "CRITICAL"
    if value >= thresholds["high"]:
        return "HIGH"
    if value >= thresholds["warn"]:
        return "WARN"
    return None
