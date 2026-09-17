"""
Data models for CareAI - Phase 1 (plain Python)

We use simple dataclasses here; in Phase 2 (LangChain) we'll migrate this
to Pydantic so the framework itself validates the model's structured output.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class VitalSigns:
    temperature_c: Optional[float] = None
    spo2_pct: Optional[float] = None
    heart_rate_bpm: Optional[int] = None
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None


@dataclass
class CheckIn:
    patient_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    vitals: VitalSigns = field(default_factory=VitalSigns)
    pain_0_10: Optional[int] = None
    breathing_difficulty: Optional[bool] = None
    dizziness: Optional[bool] = None
    took_medication: Optional[bool] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "timestamp": self.timestamp,
            "temperature_c": self.vitals.temperature_c,
            "spo2_pct": self.vitals.spo2_pct,
            "heart_rate_bpm": self.vitals.heart_rate_bpm,
            "systolic_bp": self.vitals.systolic_bp,
            "diastolic_bp": self.vitals.diastolic_bp,
            "pain_0_10": self.pain_0_10,
            "breathing_difficulty": self.breathing_difficulty,
            "dizziness": self.dizziness,
            "took_medication": self.took_medication,
            "notes": self.notes,
        }
