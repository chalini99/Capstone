from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from config.config import EXPECTED_NUM_CLASSES
from database.database import fetch_history, initialize_database, save_analysis_result
from modules.disease_detection import build_efficientnet_b0, load_class_mapping, parse_label
from modules.gradcam import GradCAM, overlay_heatmap_on_image
from modules.recommendation_engine import build_recommendation, get_disease_record, is_healthy_prediction
from utils.report_generator import generate_pdf_report


def test_recommendation_lookup_for_known_disease() -> None:
    record = get_disease_record('Tomato', 'Early Blight')
    assert record['crop'] == 'Tomato'
    assert 'symptoms' in record
    assert len(record['symptoms']) > 0


def test_orange_huanglongbing_lookup_both_spellings() -> None:
    # 1. Test lookup with the standard spelling
    rec_standard = get_disease_record('Orange', 'Huanglongbing (Citrus greening)')
    assert rec_standard['crop'] == 'Orange'
    assert 'Huanglongbing' in rec_standard['disease']

    # 2. Test lookup with the PlantVillage typo spelling ('Haunglongbing')
    rec_typo = get_disease_record('Orange', 'Haunglongbing (Citrus greening)')
    assert rec_typo['crop'] == 'Orange'
    assert 'Huanglongbing' in rec_typo['disease']

    # 3. Test build_recommendation with typo
    recommendation = build_recommendation('Orange', 'Haunglongbing (Citrus greening)')
    assert recommendation['status'] == 'disease_detected'
    assert len(recommendation['symptoms']) > 0
    assert len(recommendation['cultural_management']) > 0
    assert len(recommendation['references']) > 0


def test_all_26_disease_classes_retrieval() -> None:
    mapping = load_class_mapping()
    disease_count = 0
    for idx_str, label in mapping.items():
        crop, disease = parse_label(label)
        if is_healthy_prediction(disease):
            continue
        disease_count += 1
        rec = build_recommendation(crop, disease)
        assert rec['status'] == 'disease_detected', f"Failed retrieval for {label}: got {rec['status']}"
        assert len(rec['symptoms']) > 0, f"Missing symptoms for {label}"
        assert len(rec['causes']) > 0, f"Missing causes for {label}"
        assert len(rec['preventive_measures']) > 0, f"Missing preventive measures for {label}"
        assert len(rec['references']) > 0, f"Missing references for {label}"
    assert disease_count == 26


def test_all_12_healthy_classes_handling() -> None:
    mapping = load_class_mapping()
    healthy_count = 0
    for idx_str, label in mapping.items():
        crop, disease = parse_label(label)
        if not is_healthy_prediction(disease):
            continue
        healthy_count += 1
        rec = build_recommendation(crop, disease)
        assert rec['status'] == 'healthy'
        assert 'healthy' in rec['message'].lower()
        assert len(rec['preventive_measures']) > 0
    assert healthy_count == 12


def test_missing_disease_entry_fallback() -> None:
    with pytest.raises(KeyError):
        get_disease_record('UnknownCrop', 'UnknownDisease')

    rec = build_recommendation('UnknownCrop', 'UnknownDisease')
    assert rec['status'] == 'knowledge_not_available'
    assert 'not available' in rec['message'].lower()


def test_unverified_chemical_treatment_rejection() -> None:
    recommendation = build_recommendation('Tomato', 'Early Blight')
    chemical = recommendation['chemical_management']
    assert chemical[0]['verified'] is False
    assert 'No verified chemical option is available' in chemical[0]['option']


def test_database_insertion_and_fetch(tmp_path: Path) -> None:
    db_path = tmp_path / 'history.db'
    initialize_database(db_path)
    row_id = save_analysis_result(
        image_filename='leaf.jpg',
        crop_name='Tomato',
        predicted_disease='Early Blight',
        confidence_score=92.5,
        top_predictions=[{'crop': 'Tomato', 'disease': 'Early Blight', 'confidence': 0.925}],
        recommendation_summary='Healthy management guidance',
        db_path=db_path,
    )
    history = fetch_history(db_path)
    assert row_id > 0
    assert len(history) == 1
    assert history[0]['predicted_disease'] == 'Early Blight'


def test_pdf_generation_creates_report_file(tmp_path: Path) -> None:
    output_path = tmp_path / 'report.pdf'
    generated_path = generate_pdf_report(
        {
            'crop': 'Tomato',
            'disease': 'Early Blight',
            'confidence': '92.50',
            'timestamp': '2026-11-07',
            'recommendation_summary': 'Follow integrated disease management guidance.',
            'symptoms': ['Leaf spots'],
            'preventive_measures': ['Remove infected debris'],
            'references': [{'name': 'Example source', 'url': 'https://example.com'}],
        },
        output_path=output_path,
    )
    assert Path(generated_path).exists()


def test_gradcam_heatmap_and_overlay() -> None:
    model = build_efficientnet_b0(num_classes=EXPECTED_NUM_CLASSES)
    model.eval()

    gradcam = GradCAM(model, target_layer='features.8')
    dummy_input = torch.randn(1, 3, 224, 224)

    heatmap = gradcam.generate(dummy_input, target_class=15)
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6

    test_img = Image.new('RGB', (224, 224), color=(80, 140, 80))
    overlay = overlay_heatmap_on_image(test_img, heatmap)
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (224, 224)
