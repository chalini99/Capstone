from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
import streamlit as st

from config.config import CLASS_MAPPING_PATH, MODEL_PATH
from database.database import save_analysis_result
from modules.disease_detection import load_trained_model, predict_image
from modules.gradcam import GradCAM, get_heatmap_image, overlay_heatmap_on_image
from modules.image_preprocessing import preprocess_image_for_model, resize_for_display
from modules.recommendation_engine import build_recommendation
from modules.ui_components import (
    render_disclaimer,
    render_hero,
    render_management_cards,
    render_status_badge,
    render_top_predictions_bars,
)
from modules.validators import load_image_file
from utils.report_generator import generate_pdf_report


@st.cache_resource(show_spinner=False)
def get_cached_model():
    return load_trained_model(MODEL_PATH, CLASS_MAPPING_PATH)


def render_detection_page() -> None:
    render_hero(
        title="Crop Disease Detection Workspace",
        subtitle="Upload leaf imagery to run real-time deep learning classification via EfficientNet-B0. "
        "Review top class probabilities, Grad-CAM visual attention regions, and retrieve verified management recommendations.",
        badge="Inference & Explainability",
    )

    # File Upload & Configuration Columns
    col_input, col_context = st.columns([1.1, 1])

    with col_input:
        st.markdown("#### 1. Upload Leaf Image")
        uploaded_file = st.file_uploader(
            "Select leaf photograph (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="High-contrast, well-focused photos of affected or healthy leaves give optimal accuracy.",
        )

        image = None
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                image = load_image_file(uploaded_file)
                st.image(
                    resize_for_display(image, width=480),
                    caption=f"Uploaded: {getattr(uploaded_file, 'name', 'leaf.jpg')}",
                    use_container_width=True,
                )
            except Exception as err:
                st.error(f"Invalid or corrupted image: {err}")
                image = None

    with col_context:
        st.markdown("#### 2. Optional Agronomic Context")
        st.caption("Provide optional field context to personalize the management guidance.")

        with st.expander("🌾 Field & Climate Parameters", expanded=True):
            growth_stage = st.selectbox(
                "Crop Growth Stage",
                ["Not specified", "Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity / Harvest"],
            )

            severity = st.selectbox(
                "Observed Disease Severity",
                ["Not specified", "Early / Mild (< 10% leaf area)", "Moderate (10% - 30%)", "Severe (> 30%)"],
            )

            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                location = st.text_input("Region / Location", placeholder="e.g. Karnataka, India")
                temperature = st.text_input("Temperature", placeholder="e.g. 28°C")

            with col_sub2:
                soil_conditions = st.text_input("Soil Condition", placeholder="e.g. Well-drained loamy")
                humidity = st.text_input("Relative Humidity", placeholder="e.g. 75% RH")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Primary Inference Trigger Button
    btn_col1, btn_col2, _ = st.columns([2, 2, 4])
    with btn_col1:
        analyze_clicked = st.button("🔬 Analyze Crop Leaf", type="primary", use_container_width=True)

    if analyze_clicked:
        if uploaded_file is None or image is None:
            st.warning("Please upload a valid JPG or PNG leaf image before requesting analysis.")
            return

        with st.spinner("Executing neural network inference & agronomic intelligence..."):
            try:
                model, class_mapping, device = get_cached_model()
            except Exception as exc:
                st.error(f"Failed to load trained model artifacts: {exc}")
                return

            # Compute unique hash to prevent duplicate database inserts on page rerun
            uploaded_file.seek(0)
            img_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            img_hash = hashlib.md5(img_bytes).hexdigest()

            # Execute Model Prediction
            try:
                prediction = predict_image(image, model, class_mapping, device, top_k=3)
            except Exception as exc:
                st.error(f"Inference failed during image classification: {exc}")
                return

            # Retrieve Recommendations
            recommendation = build_recommendation(
                crop_name=prediction.crop,
                disease_name=prediction.disease,
                growth_stage=growth_stage if growth_stage != "Not specified" else None,
                disease_severity=severity if severity != "Not specified" else None,
                location=location or None,
                soil_conditions=soil_conditions or None,
                temperature=temperature or None,
                humidity=humidity or None,
            )

            # Store in Session State
            prediction_dict = {
                "crop": prediction.crop,
                "disease": prediction.disease,
                "confidence": round(prediction.confidence_pct, 2),
                "top_predictions": prediction.top_predictions,
                "recommendation_summary": recommendation.get("description", recommendation.get("message", "")),
                "description": recommendation.get("description", ""),
                "symptoms": recommendation.get("symptoms", []),
                "causes": recommendation.get("causes", []),
                "chemical_management": recommendation.get("chemical_management", []),
                "biological_management": recommendation.get("biological_management", []),
                "cultural_management": recommendation.get("cultural_management", []),
                "preventive_measures": recommendation.get("preventive_measures", []),
                "references": recommendation.get("references", []),
                "explanation": recommendation.get("explanation", ""),
                "status": recommendation.get("status", ""),
                "image_filename": getattr(uploaded_file, "name", "uploaded_leaf.jpg"),
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            }

            st.session_state.latest_prediction = prediction_dict
            st.session_state.latest_recommendation = recommendation
            st.session_state.latest_image = image

            # Generate Grad-CAM Heatmap
            try:
                gradcam = GradCAM(model)
                preprocessed = preprocess_image_for_model(image, 224).to(device)
                target_cls = prediction.top_predictions[0].get("class_index", None)
                heatmap = gradcam.generate(preprocessed, target_class=target_cls)
                st.session_state.latest_heatmap = heatmap
                st.session_state.latest_overlay = overlay_heatmap_on_image(image, heatmap)
                st.session_state.latest_heatmap_img = get_heatmap_image(heatmap, size=image.size)
            except Exception as cam_err:
                st.session_state.latest_heatmap = None
                st.session_state.latest_overlay = None
                st.session_state.latest_heatmap_img = None
                st.session_state.cam_error = str(cam_err)

            # Save to database only if not already saved in this session
            if st.session_state.get("last_saved_hash") != img_hash:
                db_id = save_analysis_result(
                    image_filename=prediction_dict["image_filename"],
                    crop_name=prediction.crop,
                    predicted_disease=prediction.disease,
                    confidence_score=prediction_dict["confidence"],
                    top_predictions=prediction.top_predictions,
                    recommendation_summary=prediction_dict["recommendation_summary"],
                )
                st.session_state.last_saved_hash = img_hash
                st.session_state.last_saved_id = db_id

    # Display Active Analysis Results if Available
    if "latest_prediction" in st.session_state and st.session_state.latest_prediction:
        res = st.session_state.latest_prediction
        rec = st.session_state.latest_recommendation
        img = st.session_state.latest_image
        is_healthy = res.get("status") == "healthy" or "healthy" in res["disease"].lower()

        st.markdown("---")
        st.markdown("### 📋 Preliminary Screening Results")

        res_left, res_right = st.columns([1.1, 1.3])

        with res_left:
            st.image(
                resize_for_display(img, width=480),
                caption=f"Evaluated Specimen: {res['image_filename']}",
                use_container_width=True,
            )
            if is_healthy:
                render_status_badge(True, f"Healthy {res['crop']} Specimen")
            else:
                render_status_badge(False, f"Condition: {res['disease']}")

        with res_right:
            st.markdown(
                f"""
                <div class="agro-card">
                    <div class="agro-card-header">
                        <div class="agro-card-title">{res['crop']} — {res['disease']}</div>
                        <span class="agro-badge {'agro-badge-healthy' if is_healthy else 'agro-badge-disease'}">
                            {res['confidence']:.2f}% Confidence
                        </span>
                    </div>
                    <p style="color: #475569; font-size: 0.95rem; margin-bottom: 0.8rem;">
                        <strong>Pathogen / Status:</strong> {rec.get('scientific_name', 'N/A')}<br/>
                        <strong>Record ID:</strong> #{st.session_state.get('last_saved_id', 'Logged')}<br/>
                        <strong>Evaluated:</strong> {res['timestamp']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Low confidence uncertainty alert
            if res["confidence"] < 40.0:
                st.warning(
                    f"⚠️ **Low Confidence Notice ({res['confidence']:.1f}%)**: The model exhibited significant uncertainty. "
                    "Ensure the leaf is well-lit and centered. Laboratory diagnostic confirmation is strongly advised."
                )

            # Top 3 predictions with visual progress bars
            render_top_predictions_bars(res["top_predictions"])

        # Tabbed Detail View
        st.markdown("<br/>", unsafe_allow_html=True)
        tab_rec, tab_context, tab_cam, tab_report = st.tabs([
            "🌾 Smart AgroCare Recommendations",
            "🌍 Contextual Field Guidance",
            "🧠 Explainable AI (Grad-CAM)",
            "📄 Export Diagnostic Report",
        ])

        with tab_rec:
            st.markdown(f"#### Disease Overview: {res['disease']}")
            st.write(rec.get("description", "No detailed description available."))

            if rec.get("symptoms"):
                st.markdown("##### Characteristic Symptoms")
                for s in rec["symptoms"]:
                    st.markdown(f"• {s}")

            if rec.get("causes"):
                st.markdown("##### Underlying Causes & Vector Information")
                for c in rec["causes"]:
                    st.markdown(f"• {c}")

            st.markdown("---")
            render_management_cards(rec)

        with tab_context:
            st.markdown("#### Agronomic Context Integration")
            explanation = rec.get("explanation")
            if explanation:
                st.info(f"🌱 **Applied Agronomic Context**: {explanation}")
            else:
                st.write("Default baseline guidance was applied from the verified agricultural knowledge base.")

        with tab_cam:
            st.markdown("#### Visual Attention Attribution (Grad-CAM)")
            st.caption(
                "Grad-CAM computes spatial gradients from the final convolutional block (`features.8`) "
                "to pinpoint regions of the leaf that most strongly activated the model's prediction."
            )

            overlay = st.session_state.get("latest_overlay")
            heat_img = st.session_state.get("latest_heatmap_img")

            if overlay is not None and heat_img is not None:
                cam_c1, cam_c2, cam_c3 = st.columns(3)
                with cam_c1:
                    st.image(resize_for_display(img, width=320), caption="1. Original Leaf Specimen", use_container_width=True)
                with cam_c2:
                    st.image(resize_for_display(heat_img, width=320), caption="2. Attention Heatmap (JET)", use_container_width=True)
                with cam_c3:
                    st.image(resize_for_display(overlay, width=320), caption="3. Blended Diagnostic Overlay", use_container_width=True)
            else:
                st.info(f"Grad-CAM visualization note: {st.session_state.get('cam_error', 'Not generated.')}")

        with tab_report:
            st.markdown("#### Generate & Download Diagnostic Report")
            st.write("Generate a formal, publication-ready PDF report containing diagnostic summary, top predictions, management advice, and references.")

            try:
                report_path = generate_pdf_report(res)
                with open(report_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download Official Diagnostic Report (PDF)",
                        data=pdf_file,
                        file_name=Path(report_path).name,
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                    )
                st.success(f"Report compiled successfully: `{Path(report_path).name}`")
            except Exception as rep_err:
                st.error(f"Report generation error: {rep_err}")

        render_disclaimer()
