from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.config import DATABASE_PATH


def get_connection(db_path: str | Path = DATABASE_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: str | Path = DATABASE_PATH) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_filename TEXT,
                crop_name TEXT,
                predicted_disease TEXT,
                confidence_score REAL,
                top_predictions TEXT,
                recommendation_summary TEXT,
                analysis_timestamp TEXT NOT NULL
            )
            '''
        )
        conn.commit()
    finally:
        conn.close()


def save_analysis_result(
    image_filename: str,
    crop_name: str,
    predicted_disease: str,
    confidence_score: float,
    top_predictions: list[dict[str, Any]] | str,
    recommendation_summary: str,
    db_path: str | Path = DATABASE_PATH,
) -> int:
    if isinstance(top_predictions, list):
        payload = json.dumps(top_predictions, ensure_ascii=False)
    else:
        payload = str(top_predictions)

    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            '''
            INSERT INTO analysis_history (
                image_filename,
                crop_name,
                predicted_disease,
                confidence_score,
                top_predictions,
                recommendation_summary,
                analysis_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                image_filename,
                crop_name,
                predicted_disease,
                float(confidence_score),
                payload,
                recommendation_summary,
                datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def fetch_history(db_path: str | Path = DATABASE_PATH) -> list[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            '''
            SELECT *
            FROM analysis_history
            ORDER BY id DESC
            '''
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def fetch_analysis_by_id(
    record_id: int,
    db_path: str | Path = DATABASE_PATH,
) -> dict[str, Any] | None:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            '''
            SELECT *
            FROM analysis_history
            WHERE id = ?
            ''',
            (record_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def fetch_history_stats(db_path: str | Path = DATABASE_PATH) -> dict[str, Any]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            '''
            SELECT crop_name, predicted_disease, confidence_score
            FROM analysis_history
            '''
        ).fetchall()

        total = len(rows)
        if total == 0:
            return {
                "total_analyses": 0,
                "healthy_count": 0,
                "disease_count": 0,
                "avg_confidence": 0.0,
                "most_analyzed_crop": "N/A",
                "most_detected_disease": "N/A",
            }

        healthy_count = sum(1 for r in rows if "healthy" in str(r["predicted_disease"]).lower())
        disease_count = total - healthy_count

        confidences = [float(r["confidence_score"]) for r in rows if r["confidence_score"] is not None]
        avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0.0

        crops = [r["crop_name"] for r in rows if r["crop_name"]]
        most_crop = max(set(crops), key=crops.count) if crops else "N/A"

        diseases = [
            r["predicted_disease"]
            for r in rows
            if r["predicted_disease"] and "healthy" not in str(r["predicted_disease"]).lower()
        ]
        most_disease = max(set(diseases), key=diseases.count) if diseases else "None"

        return {
            "total_analyses": total,
            "healthy_count": healthy_count,
            "disease_count": disease_count,
            "avg_confidence": avg_conf,
            "most_analyzed_crop": most_crop,
            "most_detected_disease": most_disease,
        }
    finally:
        conn.close()


def clear_history(db_path: str | Path = DATABASE_PATH) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute('DELETE FROM analysis_history')
        conn.commit()
    finally:
        conn.close()
