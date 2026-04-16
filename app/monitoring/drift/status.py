def resolve_status(severity: str) -> str:
    if severity in ("WARN", "HIGH"):
        return "DEGRADED"
    if severity == "CRITICAL":
        return "CRITICAL"
    return "OK"
