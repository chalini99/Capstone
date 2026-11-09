from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def is_supported_image(filename: str | None) -> bool:
    if not filename:
        return False
    return Path(filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def load_image_file(uploaded_file) -> Image.Image:
    if uploaded_file is None:
        raise ValueError('No image file was uploaded.')

    filename = getattr(uploaded_file, 'name', '')
    if not is_supported_image(filename):
        raise ValueError('Unsupported file type. Please upload a JPG, JPEG, or PNG image.')

    try:
        image = Image.open(uploaded_file)
        image = image.convert('RGB')
        return image
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError('The uploaded image could not be opened or appears to be corrupted.') from exc


def validate_disease_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError('Disease knowledge entry must be a JSON object.')

    required_fields = ['crop', 'disease', 'symptoms', 'causes', 'preventive_measures', 'references']
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise ValueError(f'Disease record is missing required fields: {missing}')
    return record


def safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ''
