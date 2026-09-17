"""
Alert rules engine - Phase 1.

IMPORTANT: these thresholds are general reference values and are NOT
medical criteria. The agent only detects signals to escalate to a human
(caregiver/health professional) - it never diagnoses or decides treatment.
This separation becomes an explicit "guardrail" later in the roadmap
(Phase 7).
"""
from models import CheckIn

EMERGENCY_KEYWORDS = [
    "can't breathe",
    "cannot breathe",
    "chest pain",
    "fainted",
    "fainting",
    "bleeding",
]


def detect_alerts(checkin: CheckIn) -> list[str]:
    """Returns a list of alert reasons. Empty list = everything normal."""
    alerts = []
    v = checkin.vitals

    if v.temperature_c is not None:
        if v.temperature_c >= 38.0:
            alerts.append(f"Fever: {v.temperature_c}°C")
        elif v.temperature_c <= 35.0:
            alerts.append(f"Hypothermia: {v.temperature_c}°C")

    if v.spo2_pct is not None and v.spo2_pct < 92:
        alerts.append(f"Low SpO2: {v.spo2_pct}%")

    if v.heart_rate_bpm is not None:
        if v.heart_rate_bpm < 50:
            alerts.append(f"Low heart rate: {v.heart_rate_bpm} bpm")
        elif v.heart_rate_bpm > 120:
            alerts.append(f"High heart rate: {v.heart_rate_bpm} bpm")

    if v.systolic_bp is not None:
        if v.systolic_bp > 180 or v.systolic_bp < 90:
            alerts.append(f"Systolic blood pressure out of range: {v.systolic_bp}")

    if v.diastolic_bp is not None:
        if v.diastolic_bp > 120 or v.diastolic_bp < 60:
            alerts.append(f"Diastolic blood pressure out of range: {v.diastolic_bp}")

    if checkin.pain_0_10 is not None and checkin.pain_0_10 >= 8:
        alerts.append(f"Severe pain: {checkin.pain_0_10}/10")

    if checkin.breathing_difficulty:
        alerts.append("Breathing difficulty reported")

    notes_lower = checkin.notes.lower()
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in notes_lower:
            alerts.append(f"Emergency keyword detected: '{keyword}'")
            break

    return alerts
