"""
SQLite persistence - Phase 1.
No ORM yet: raw SQL so it's clear what's happening under the hood.
"""
import sqlite3
from models import CheckIn

DB_PATH = "careai.db"


def init_db(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            temperature_c REAL,
            spo2_pct REAL,
            heart_rate_bpm INTEGER,
            systolic_bp INTEGER,
            diastolic_bp INTEGER,
            pain_0_10 INTEGER,
            breathing_difficulty INTEGER,
            dizziness INTEGER,
            took_medication INTEGER,
            notes TEXT,
            alerts TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_checkin(checkin: CheckIn, alerts: list[str], db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    d = checkin.to_dict()
    conn.execute("""
        INSERT INTO checkins (
            patient_id, timestamp, temperature_c, spo2_pct,
            heart_rate_bpm, systolic_bp, diastolic_bp,
            pain_0_10, breathing_difficulty, dizziness, took_medication,
            notes, alerts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        d["patient_id"], d["timestamp"], d["temperature_c"], d["spo2_pct"],
        d["heart_rate_bpm"], d["systolic_bp"], d["diastolic_bp"],
        d["pain_0_10"], int(bool(d["breathing_difficulty"])), int(bool(d["dizziness"])),
        int(bool(d["took_medication"])), d["notes"], "; ".join(alerts)
    ))
    conn.commit()
    conn.close()


def patient_history(patient_id: str, db_path: str = DB_PATH) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM checkins WHERE patient_id = ? ORDER BY timestamp DESC",
        (patient_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
