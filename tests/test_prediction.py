from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from config.config import EXPECTED_NUM_CLASSES, ensure_model_artifacts
from modules.disease_detection import build_efficientnet_b0, load_class_mapping, parse_label, predict_image
from modules.image_preprocessing import preprocess_image_for_model


def test_model_artifacts_missing_message_when_files_do_not_exist() -> None:
    exists, message = ensure_model_artifacts()
    if Path('models/efficientnet_b0_best.pth').exists() and Path('config/class_mapping.json').exists():
        assert exists is True
    else:
        assert exists is False
        assert 'Missing' in message


def test_class_mapping_raises_when_file_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / 'missing.json'
    with pytest.raises(FileNotFoundError):
        load_class_mapping(missing_path)


def test_parse_label_handles_underscore_replacement() -> None:
    crop, disease = parse_label('Tomato___Early_Blight')
    assert crop == 'Tomato'
    assert disease == 'Early Blight'


def test_parse_label_canonicalizes_orange_huanglongbing() -> None:
    crop, disease = parse_label('Orange___Haunglongbing_(Citrus_greening)')
    assert crop == 'Orange'
    assert disease == 'Huanglongbing (Citrus greening)'


def test_all_38_classes_parse_correctly() -> None:
    mapping = load_class_mapping()
    assert len(mapping) == EXPECTED_NUM_CLASSES
    for idx_str, label in mapping.items():
        crop, disease = parse_label(label)
        assert crop != ''
        assert disease != ''
        assert '_' not in crop
        assert '_' not in disease


def test_preprocess_image_returns_model_ready_tensor() -> None:
    image = Image.new('RGB', (224, 224), color='green')
    tensor = preprocess_image_for_model(image, image_size=224)
    assert tensor.shape == (1, 3, 224, 224)


def test_predict_image_includes_class_index() -> None:
    model = build_efficientnet_b0(num_classes=EXPECTED_NUM_CLASSES)
    model.eval()
    mapping = load_class_mapping()
    image = Image.new('RGB', (224, 224), color='green')
    device = torch.device('cpu')

    result = predict_image(image, model, mapping, device, top_k=3)
    assert len(result.top_predictions) == 3
    for pred in result.top_predictions:
        assert 'class_index' in pred
        assert isinstance(pred['class_index'], int)
        assert 0 <= pred['class_index'] < EXPECTED_NUM_CLASSES
        assert 'crop' in pred
        assert 'disease' in pred
        assert 'confidence' in pred
