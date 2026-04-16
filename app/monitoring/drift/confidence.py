def calculate_confidence(scope_pct: float, total_queries: int) -> str:
    if total_queries >= 500 and scope_pct >= 50:
        return "STRONG"
    if total_queries >= 100 and scope_pct >= 20:
        return "MODERATE"
    return "WEAK"
