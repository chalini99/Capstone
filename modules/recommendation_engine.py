
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.config import KNOWLEDGE_BASE_DIR
from modules.validators import validate_disease_record


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)

    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing knowledge base file: {json_path}"
        )

    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_disease_key(text: str) -> str:
    """
    Normalize crop and disease strings for robust lookup across dataset typos,
    parenthetical suffixes, underscores, and spacing differences.
    """
    import re
    cleaned = str(text or "").strip().lower()
    # Normalize known PlantVillage dataset typo (Haunglongbing -> Huanglongbing)
    cleaned = cleaned.replace("haung", "huang")
    return re.sub(r"[^a-z0-9]", "", cleaned)


def get_disease_record(
    crop_name: str,
    disease_name: str,
    knowledge_base_dir: str | Path = KNOWLEDGE_BASE_DIR
) -> dict[str, Any]:

    root = Path(knowledge_base_dir)

    treatments = load_json(
        root / "disease_treatments.json"
    )

    diseases = treatments.get("diseases", [])

    target_crop = str(crop_name or "").strip().lower()
    target_disease = str(disease_name or "").strip().lower()

    # Pass 1: Exact case-insensitive match
    for record in diseases:
        record_crop = str(record.get("crop", "")).strip().lower()
        record_disease = str(record.get("disease", "")).strip().lower()

        if target_crop == record_crop and target_disease == record_disease:
            validate_disease_record(record)
            return record

    # Pass 2: Canonical normalized match (resolves spelling variants, punctuation, etc.)
    target_crop_norm = normalize_disease_key(crop_name)
    target_disease_norm = normalize_disease_key(disease_name)

    for record in diseases:
        record_crop_norm = normalize_disease_key(record.get("crop", ""))
        record_disease_norm = normalize_disease_key(record.get("disease", ""))

        if target_crop_norm == record_crop_norm and target_disease_norm == record_disease_norm:
            validate_disease_record(record)
            return record

    # Pass 3: Normalized substring matching when crop matches
    for record in diseases:
        record_crop_norm = normalize_disease_key(record.get("crop", ""))
        record_disease_norm = normalize_disease_key(record.get("disease", ""))

        if target_crop_norm == record_crop_norm and (
            target_disease_norm in record_disease_norm or record_disease_norm in target_disease_norm
        ):
            validate_disease_record(record)
            return record

    raise KeyError(
        f'No disease knowledge record found for crop '
        f'"{crop_name}" and disease "{disease_name}".'
    )


def get_preventive_measures(
    knowledge_base_dir: str | Path = KNOWLEDGE_BASE_DIR
) -> list[dict[str, Any]]:

    payload = load_json(
        Path(knowledge_base_dir) / "preventive_measures.json"
    )

    return payload.get("preventive_practices", [])


def get_sources(
    knowledge_base_dir: str | Path = KNOWLEDGE_BASE_DIR
) -> list[dict[str, Any]]:

    payload = load_json(
        Path(knowledge_base_dir) / "agricultural_sources.json"
    )

    return payload.get("sources", [])


def is_healthy_prediction(disease_name: str) -> bool:
    """Identify healthy predictions from the model."""

    normalized = str(disease_name or "").strip().lower()

    healthy_labels = {
        "healthy",
        "healthy leaf",
        "healthy plant",
        "no disease",
        "disease free",
        "disease-free",
    }

    return normalized in healthy_labels


def build_recommendation(
    crop_name: str,
    disease_name: str,
    growth_stage: str | None = None,
    disease_severity: str | None = None,
    location: str | None = None,
    soil_conditions: str | None = None,
    temperature: str | None = None,
    humidity: str | None = None,
    knowledge_base_dir: str | Path = KNOWLEDGE_BASE_DIR,
) -> dict[str, Any]:

    # Load shared preventive guidance and sources.
    global_preventive = get_preventive_measures(
        knowledge_base_dir
    )

    source_list = get_sources(
        knowledge_base_dir
    )

    # Build contextual explanation.
    explanation_items = []

    if growth_stage:
        explanation_items.append(
            f"Growth stage considered: {growth_stage}."
        )

    if disease_severity:
        explanation_items.append(
            f"Disease severity considered: {disease_severity}."
        )

    if location:
        explanation_items.append(
            f"Location considered: {location}."
        )

    if soil_conditions:
        explanation_items.append(
            f"Soil conditions considered: {soil_conditions}."
        )

    if temperature:
        explanation_items.append(
            f"Temperature context: {temperature}."
        )

    if humidity:
        explanation_items.append(
            f"Humidity context: {humidity}."
        )

    explanation = (
        " ".join(explanation_items)
        if explanation_items
        else
        "Default management guidance was applied using "
        "the available crop and disease knowledge base."
    )

    # -------------------------------------------------
    # CASE 1: HEALTHY CROP
    # -------------------------------------------------

    if is_healthy_prediction(disease_name):

        return {
            "crop": crop_name,
            "disease": "Healthy",
            "scientific_name": "Not applicable",
            "description": (
                f"The model classified the uploaded {crop_name} "
                "leaf as healthy. No recognized disease was "
                "predicted by the classification model."
            ),
            "symptoms": [],
            "causes": [],
            "chemical_management": [],
            "biological_management": [],
            "cultural_management": [],
            "preventive_measures": [
                "Continue regular crop monitoring.",
                "Maintain appropriate irrigation practices.",
                "Ensure balanced crop nutrition.",
                "Remove dead or decaying plant material.",
                "Monitor leaves regularly for unusual symptoms.",
            ],
            "explanation": explanation,
            "references": [],
            "global_preventive_measures": global_preventive,
            "source_list": source_list,
            "status": "healthy",
            "message": (
                "The crop was classified as healthy. "
                "Disease-specific chemical treatment "
                "is not required based on this prediction."
            ),
        }

    # -------------------------------------------------
    # CASE 2: KNOWN DISEASE
    # -------------------------------------------------

    try:
        record = get_disease_record(
            crop_name,
            disease_name,
            knowledge_base_dir
        )

    except (KeyError, ValueError):

        # -------------------------------------------------
        # CASE 3: DISEASE NOT IN KNOWLEDGE BASE
        # -------------------------------------------------

        return {
            "crop": crop_name,
            "disease": disease_name,
            "scientific_name": "Not available",
            "description": (
                "The AI model detected this disease class, "
                "but a verified disease-specific knowledge "
                "record is not currently available."
            ),
            "symptoms": [],
            "causes": [],
            "chemical_management": [
                {
                    "option": (
                        "No verified chemical option is available."
                    ),
                    "reason": (
                        "A disease-specific verified product "
                        "record or official label reference "
                        "is not available in the knowledge base."
                    ),
                    "verified": False,
                    "source": (
                        "Project safety and verification requirement."
                    ),
                }
            ],
            "biological_management": [],
            "cultural_management": [],
            "preventive_measures": [
                "Monitor the crop for changes in leaf symptoms.",
                "Maintain field sanitation.",
                "Avoid unnecessary pesticide application.",
                "Consult a qualified agricultural expert "
                "for disease confirmation and management."
            ],
            "explanation": explanation,
            "references": [],
            "global_preventive_measures": global_preventive,
            "source_list": source_list,
            "status": "knowledge_not_available",
            "message": (
                "Prediction completed, but disease-specific "
                "recommendations are not available."
            ),
        }

    # -------------------------------------------------
    # CASE 4: BUILD KNOWN DISEASE RECOMMENDATIONS
    # -------------------------------------------------

    chemical_options = record.get(
        "chemical_management", []
    )

    # Only show chemical recommendations explicitly
    # marked as verified.
    verified_chemical_options = [
        item
        for item in chemical_options
        if item.get("verified") is True
    ]

    if not verified_chemical_options:
        verified_chemical_options = [
            {
                "option": (
                    "No verified chemical option is available "
                    "in the current knowledge base."
                ),
                "reason": (
                    "No verified chemical recommendation "
                    "could be included without a documented "
                    "product label or local registration reference."
                ),
                "verified": False,
                "source": (
                    "Safety requirement: no unsupported "
                    "pesticide guidance."
                ),
            }
        ]

    return {
        "crop": record.get("crop", crop_name),
        "disease": record.get("disease", disease_name),
        "scientific_name": record.get(
            "scientific_name",
            "Not specified"
        ),
        "description": record.get(
            "description",
            "No description available."
        ),
        "symptoms": record.get("symptoms", []),
        "causes": record.get("causes", []),
        "chemical_management": verified_chemical_options,
        "biological_management": record.get(
            "biological_management", []
        ),
        "cultural_management": record.get(
            "cultural_management", []
        ),
        "preventive_measures": record.get(
            "preventive_measures", []
        ),
        "explanation": explanation,
        "references": record.get("references", []),
        "global_preventive_measures": global_preventive,
        "source_list": source_list,
        "status": "disease_detected",
        "message": (
            "Disease-specific knowledge-base recommendations "
            "were retrieved successfully."
        ),
    }
