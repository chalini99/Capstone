from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from database.database import (
    fetch_analysis_by_id,
    fetch_history,
    fetch_history_stats,
    initialize_database,
    save_analysis_result,
)
from modules.gradcam import get_heatmap_image
from utils.report_generator import generate_pdf_report


def test_fetch_history_stats_structure(tmp_path: Path) -> None:
    db_path = tmp_path / "test_stats.db"
    initialize_database(db_path)

    # Empty stats
    stats_empty = fetch_history_stats(db_path)
    assert stats_empty["total_analyses"] == 0
    assert stats_empty["healthy_count"] == 0
    assert stats_empty["disease_count"] == 0
    assert stats_empty["avg_confidence"] == 0.0

    # Insert 1 healthy and 1 diseased
    save_analysis_result(
        image_filename="test_healthy.jpg",
        crop_name="Apple",
        predicted_disease="healthy",
        confidence_score=98.5,
        top_predictions=[{"crop": "Apple", "disease": "healthy", "confidence": 0.985}],
        recommendation_summary="Healthy leaf",
        db_path=db_path,
    )
    save_analysis_result(
        image_filename="test_disease.jpg",
        crop_name="Tomato",
        predicted_disease="Early Blight",
        confidence_score=91.5,
        top_predictions=[{"crop": "Tomato", "disease": "Early Blight", "confidence": 0.915}],
        recommendation_summary="Early blight detected",
        db_path=db_path,
    )

    stats = fetch_history_stats(db_path)
    assert stats["total_analyses"] == 2
    assert stats["healthy_count"] == 1
    assert stats["disease_count"] == 1
    assert stats["avg_confidence"] == 95.0
    assert stats["most_analyzed_crop"] in ["Apple", "Tomato"]
    assert stats["most_detected_disease"] == "Early Blight"


def test_fetch_analysis_by_id(tmp_path: Path) -> None:
    db_path = tmp_path / "test_id.db"
    initialize_database(db_path)

    row_id = save_analysis_result(
        image_filename="leaf.jpg",
        crop_name="Grape",
        predicted_disease="Black rot",
        confidence_score=88.2,
        top_predictions=[],
        recommendation_summary="Black rot advisory",
        db_path=db_path,
    )

    record = fetch_analysis_by_id(row_id, db_path=db_path)
    assert record is not None
    assert record["id"] == row_id
    assert record["crop_name"] == "Grape"
    assert record["predicted_disease"] == "Black rot"

    # Non-existent ID returns None
    missing = fetch_analysis_by_id(99999, db_path=db_path)
    assert missing is None


def test_gradcam_standalone_heatmap_image() -> None:
    dummy_cam = np.random.rand(224, 224).astype(np.float32)
    heat_img = get_heatmap_image(dummy_cam, size=(300, 300))
    assert isinstance(heat_img, Image.Image)
    assert heat_img.size == (300, 300)
    assert heat_img.mode == "RGB"


def test_ui_modules_import() -> None:
    import modules.ui_components
    import modules.ui_dashboard
    import modules.ui_detection
    import modules.ui_recommendations
    import modules.ui_analytics
    import modules.ui_history
    import modules.ui_reports
    import modules.ui_knowledge_base
    import modules.ui_about

    assert hasattr(modules.ui_dashboard, "render_dashboard_page")
    assert hasattr(modules.ui_detection, "render_detection_page")
    assert hasattr(modules.ui_recommendations, "render_recommendations_page")
    assert hasattr(modules.ui_analytics, "render_analytics_page")
    assert hasattr(modules.ui_history, "render_history_page")
    assert hasattr(modules.ui_reports, "render_reports_page")
    assert hasattr(modules.ui_knowledge_base, "render_knowledge_base_page")
    assert hasattr(modules.ui_about, "render_about_page")
