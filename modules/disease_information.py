from __future__ import annotations

from typing import Any

from modules.recommendation_engine import get_disease_record


def get_disease_summary(crop_name: str, disease_name: str) -> dict[str, Any]:
    record = get_disease_record(crop_name, disease_name)
    return {
        'crop': record.get('crop', crop_name),
        'disease': record.get('disease', disease_name),
        'scientific_name': record.get('scientific_name', 'Not specified'),
        'description': record.get('description', 'No description is currently available.'),
        'symptoms': record.get('symptoms', []),
        'causes': record.get('causes', []),
        'references': record.get('references', []),
    }
