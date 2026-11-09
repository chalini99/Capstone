from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd
import streamlit as st

from database.database import fetch_history
from modules.recommendation_engine import build_recommendation
from modules.ui_components import render_disclaimer, render_hero
from utils.report_generator import generate_pdf_report


def render_reports_page() -> None:
    render_hero(
        title="Diagnostic Report Generation Center",
        subtitle="Export formal, publication-quality agricultural diagnostic screening reports with full provenance, "
        "model probabilities, AgroCare chemical/biological/cultural management, and agricultural extension citations.",
        badge="Reporting & Documentation",
    )

    rows = fetch_history()
    has_latest = "latest_prediction" in st.session_state and st.session_state.latest_prediction is not None

    if not rows and not has_latest:
        st.info("No analyses recorded yet. Upload a leaf image in Disease Detection to generate diagnostic reports.")
        render_disclaimer()
        return

    # Report Source Selection
    options = []
    if has_latest:
        options.append("Latest Active Inspection Session")
    if rows:
        options.append("Select from Historical Database Records")

    report_source = st.radio("Choose Report Source", options, horizontal=True)

    report_payload = None

    if report_source == "Latest Active Inspection Session" and has_latest:
        latest = st.session_state.latest_prediction
        rec = st.session_state.latest_recommendation
        report_payload = {
            "crop": latest["crop"],
            "disease": latest["disease"],
            "confidence": latest["confidence"],
            "timestamp": latest["timestamp"],
            "image_filename": latest["image_filename"],
            "top_predictions": latest["top_predictions"],
            "recommendation_summary": latest.get("recommendation_summary", ""),
            "description": rec.get("description", ""),
            "symptoms": rec.get("symptoms", []),
            "causes": rec.get("causes", []),
            "chemical_management": rec.get("chemical_management", []),
            "biological_management": rec.get("biological_management", []),
            "cultural_management": rec.get("cultural_management", []),
            "preventive_measures": rec.get("preventive_measures", []),
            "references": rec.get("references", []),
        }
        st.success(f"Loaded live session data for **{latest['crop']} — {latest['disease']}**")

    elif rows:
        selected_id = st.selectbox(
            "Select Inspection Record",
            [r["id"] for r in rows],
            format_func=lambda x: f"Record #{x} — {next((r['crop_name'] + ' (' + r['predicted_disease'] + ')' for r in rows if r['id'] == x), '')}",
        )
        selected_row = next((r for r in rows if r["id"] == selected_id), rows[0])

        crop = selected_row["crop_name"]
        dis = selected_row["predicted_disease"]
        rec = build_recommendation(crop, dis)

        report_payload = {
            "crop": crop,
            "disease": dis,
            "confidence": selected_row["confidence_score"],
            "timestamp": selected_row["analysis_timestamp"],
            "image_filename": selected_row["image_filename"],
            "top_predictions": selected_row["top_predictions"],
            "recommendation_summary": selected_row["recommendation_summary"] or rec.get("description"),
            "description": rec.get("description"),
            "symptoms": rec.get("symptoms", []),
            "causes": rec.get("causes", []),
            "chemical_management": rec.get("chemical_management", []),
            "biological_management": rec.get("biological_management", []),
            "cultural_management": rec.get("cultural_management", []),
            "preventive_measures": rec.get("preventive_measures", []),
            "references": rec.get("references", []),
        }

    # Report Preview & Download Card
    if report_payload:
        st.markdown("---")
        st.markdown("### 📋 Document Preview")

        prev_col1, prev_col2 = st.columns([2, 1])

        with prev_col1:
            st.markdown(
                f"""
                <div class="agro-card">
                    <div class="agro-card-header">
                        <div class="agro-card-title">Diagnostic Screening Report Preview</div>
                        <span class="agro-badge agro-badge-neutral">{report_payload['crop']}</span>
                    </div>
                    <table style="width: 100%; font-size: 0.92rem; line-height: 1.8;">
                        <tr><td style="width: 35%; color: #64748b;"><strong>Target Crop:</strong></td><td>{report_payload['crop']}</td></tr>
                        <tr><td style="color: #64748b;"><strong>Predicted Condition:</strong></td><td><strong>{report_payload['disease']}</strong></td></tr>
                        <tr><td style="color: #64748b;"><strong>Confidence Level:</strong></td><td>{float(report_payload['confidence']):.2f}%</td></tr>
                        <tr><td style="color: #64748b;"><strong>Specimen File:</strong></td><td>{report_payload['image_filename']}</td></tr>
                        <tr><td style="color: #64748b;"><strong>Timestamp:</strong></td><td>{report_payload['timestamp']}</td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with prev_col2:
            st.markdown("#### Export Actions")
            st.caption("Generate a high-resolution, branded PDF document incorporating ReportLab typography.")

            try:
                pdf_file_path = generate_pdf_report(report_payload)
                with open(pdf_file_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download Official Report (PDF)",
                        data=pdf_file,
                        file_name=Path(pdf_file_path).name,
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                    )
                st.caption(f"✓ Formatted file ready: `{Path(pdf_file_path).name}`")
            except Exception as e:
                st.error(f"PDF creation failed: {e}")

    render_disclaimer()
