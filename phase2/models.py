"""
Data models for CareAI - Phase 2 (LangChain)

Migrated from plain dataclasses (Phase 1) to Pydantic models. This is what
lets LangChain validate the model's structured output automatically instead
of us parsing raw JSON by hand.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class VitalSigns(BaseModel):
    temperature_c: Optional[float] = None
    spo2_pct: Optional[float] = None
    heart_rate_bpm: Optional[int] = None
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None


class CheckIn(BaseModel):
    patient_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    vitals: VitalSigns = Field(default_factory=VitalSigns)
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
