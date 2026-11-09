from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd
import streamlit as st

from database.database import fetch_analysis_by_id, fetch_history
from modules.recommendation_engine import build_recommendation
from modules.ui_components import render_disclaimer, render_hero, render_status_badge
from utils.report_generator import generate_pdf_report


def render_history_page() -> None:
    render_hero(
        title="Prediction History & Audit Log",
        subtitle="Permanent historical repository of past leaf inspections. "
        "Review previous diagnoses, inspect top model predictions, and export historical PDF screening reports.",
        badge="Audit Repository",
    )

    rows = fetch_history()

    if not rows:
        st.info("No prior analyses recorded in database. Upload and evaluate leaf images in Disease Detection.")
        render_disclaimer()
        return

    df = pd.DataFrame(rows)
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    df["is_healthy"] = df["predicted_disease"].str.contains("healthy", case=False, na=False)

    # Search & Filter Controls
    f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 2])
    with f_col1:
        crop_list = ["All Crops"] + sorted(df["crop_name"].dropna().unique().tolist())
        sel_crop = st.selectbox("Crop Filter", crop_list)

    with f_col2:
        diag_filter = st.selectbox("Status Filter", ["All Records", "Diseased Cases", "Healthy Only"])

    with f_col3:
        search_kw = st.text_input("Search Records", placeholder="Filter by disease or filename...")

    filtered_df = df.copy()
    if sel_crop != "All Crops":
        filtered_df = filtered_df[filtered_df["crop_name"] == sel_crop]

    if diag_filter == "Diseased Cases":
        filtered_df = filtered_df[~filtered_df["is_healthy"]]
    elif diag_filter == "Healthy Only":
        filtered_df = filtered_df[filtered_df["is_healthy"]]

    if search_kw.strip():
        kw = search_kw.strip().lower()
        filtered_df = filtered_df[
            filtered_df["predicted_disease"].str.lower().str.contains(kw, na=False)
            | filtered_df["image_filename"].str.lower().str.contains(kw, na=False)
            | filtered_df["crop_name"].str.lower().str.contains(kw, na=False)
        ]

    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} recorded inspections**")

    # Interactive Table
    table_cols = ["id", "analysis_timestamp", "crop_name", "predicted_disease", "confidence_score", "image_filename"]
    clean_display = filtered_df[[c for c in table_cols if c in filtered_df.columns]].rename(
        columns={
            "id": "ID",
            "analysis_timestamp": "Timestamp",
            "crop_name": "Crop",
            "predicted_disease": "Condition",
            "confidence_score": "Confidence (%)",
            "image_filename": "Image Reference",
        }
    )
    st.dataframe(clean_display, use_container_width=True)

    # Detailed Record Inspector
    st.markdown("---")
    st.markdown("### 🔍 Historical Record Inspector")
    st.caption("Select any historical record ID to review diagnostic details, top predictions, and download a report.")

    record_ids = filtered_df["id"].tolist()
    if not record_ids:
        st.warning("No records match the current filter selection.")
        render_disclaimer()
        return

    selected_id = st.selectbox("Select Record ID to Inspect", record_ids, format_func=lambda x: f"Record #{x}")

    selected_row = fetch_analysis_by_id(selected_id)
    if selected_row:
        c_meta, c_actions = st.columns([2, 1.2])

        with c_meta:
            is_healthy_rec = "healthy" in str(selected_row["predicted_disease"]).lower()
            st.markdown(
                f"""
                <div class="agro-card">
                    <div class="agro-card-header">
                        <div class="agro-card-title">Inspection #{selected_row['id']} — {selected_row['crop_name']}</div>
                        <span class="agro-badge {'agro-badge-healthy' if is_healthy_rec else 'agro-badge-disease'}">
                            {selected_row['confidence_score']:.2f}% Confidence
                        </span>
                    </div>
                    <p style="margin-bottom: 0.5rem;">
                        <strong>Condition:</strong> {selected_row['predicted_disease']}<br/>
                        <strong>Evaluated:</strong> {selected_row['analysis_timestamp']}<br/>
                        <strong>Source File:</strong> {selected_row['image_filename']}
                    </p>
                    <p style="color: #475569; font-size: 0.9rem;">
                        <strong>Advisory Summary:</strong><br/>
                        {selected_row['recommendation_summary'] or 'Standard agricultural management protocol applies.'}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c_actions:
            st.markdown("##### 📄 Export Report")
            st.caption("Generate a formatted PDF diagnostic report from this historical record.")

            # Reconstruct recommendation for PDF
            crop = selected_row["crop_name"]
            dis = selected_row["predicted_disease"]
            rec = build_recommendation(crop, dis)

            report_data = {
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

            try:
                pdf_path = generate_pdf_report(report_data)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label=f"📥 Download Report #{selected_row['id']}",
                        data=f,
                        file_name=Path(pdf_path).name,
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary",
                    )
            except Exception as err:
                st.error(f"Report generation error: {err}")

        # Top predictions breakdown
        raw_top = selected_row.get("top_predictions")
        if raw_top:
            try:
                parsed_top = json.loads(raw_top) if isinstance(raw_top, str) else raw_top
                if isinstance(parsed_top, list) and parsed_top:
                    with st.expander("📊 View Top 3 Class Probabilities for this Inspection", expanded=False):
                        top_df = pd.DataFrame(parsed_top)
                        if "confidence" in top_df.columns:
                            top_df["confidence %"] = top_df["confidence"].apply(lambda x: f"{float(x)*100:.2f}%")
                        cols_to_show = [c for c in ["crop", "disease", "confidence %"] if c in top_df.columns]
                        st.dataframe(top_df[cols_to_show], use_container_width=True)
            except Exception:
                pass

    render_disclaimer()
