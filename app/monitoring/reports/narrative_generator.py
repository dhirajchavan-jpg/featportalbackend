# app/monitoring/reports/narrative_generator.py


def generate_narrative(drifts: list) -> dict:
    """
    Generates executive-level narrative from drift objects
    """

    if not drifts:
        return {
            "health": "HEALTHY",
            "summary": (
                "System performance remains stable. No significant risk signals "
                "were detected across quality, safety, or retrieval dimensions."
            )
        }

    lines = ["System health shows degradation trends:"]

    for drift in drifts:
        if drift.get("severity") in (None, "NONE"):
            continue

        lines.append(
            f"{drift['drift_type']} drift detected with {drift['severity']} severity. "
            f"{drift['scope_pct']}% of queries were affected, indicating impact on "
            f"{', '.join(drift['impact_areas']).lower()}."
        )


        if drift["affected_queries"]["top_examples"]:
            lines.append(
                "This drift is primarily driven by recurring patterns in user queries "
                "that exceed configured thresholds."
            )

    return {
        "health": "DEGRADED",
        "summary": " ".join(lines)
    }
