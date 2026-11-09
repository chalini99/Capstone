from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import streamlit as st

from config.config import KNOWLEDGE_BASE_DIR
from modules.recommendation_engine import build_recommendation, get_disease_record
from modules.ui_components import render_disclaimer, render_hero, render_management_cards, render_status_badge


def render_recommendations_page() -> None:
    render_hero(
        title="Smart AgroCare Advisory System",
        subtitle="Comprehensive agronomic and agrochemical guidance curated from state university extension programs "
        "and international plant health authorities. Verified chemical, biological, cultural and preventive protocols.",
        badge="Agronomic Decision Support",
    )

    # Load Knowledge Base and Class Mapping for Comprehensive Crop Selection
    kb_path = Path(KNOWLEDGE_BASE_DIR) / "disease_treatments.json"
    disease_records = []
    if kb_path.exists():
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                disease_records = payload.get("diseases", [])
        except Exception:
            disease_records = []

    from config.config import CLASS_MAPPING_PATH
    from modules.disease_detection import parse_label

    crops_set = {entry.get("crop", "Unknown") for entry in disease_records if entry.get("crop")}
    mapping_path = Path(CLASS_MAPPING_PATH)
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                for label in mapping.values():
                    c_name, _ = parse_label(label)
                    crops_set.add(c_name)
        except Exception:
            pass

    crops = sorted(list(crops_set))

    # Mode Selector: View Latest Analysis vs Browse Knowledge Base
    has_latest = "latest_prediction" in st.session_state and st.session_state.latest_prediction is not None
    view_mode = st.radio(
        "Advisory Source",
        ["Latest Analyzed Specimen", "Browse Specific Crop & Disease Catalog"] if has_latest else ["Browse Specific Crop & Disease Catalog"],
        horizontal=True,
    )

    selected_crop = None
    selected_disease = None
    recommendation = None

    if view_mode == "Latest Analyzed Specimen" and has_latest:
        latest = st.session_state.latest_prediction
        recommendation = st.session_state.latest_recommendation
        selected_crop = latest["crop"]
        selected_disease = latest["disease"]

        is_healthy = recommendation.get("status") == "healthy" or "healthy" in selected_disease.lower()

        st.markdown(
            f"""
            <div class="agro-card">
                <div class="agro-card-header">
                    <div>
                        <span class="agro-badge {'agro-badge-healthy' if is_healthy else 'agro-badge-disease'}">
                            {'Healthy Specimen' if is_healthy else 'Active Disease Alert'}
                        </span>
                        <h2 style="margin: 0.5rem 0 0.2rem 0; color: #0d3813;">{selected_crop} — {selected_disease}</h2>
                        <span style="color: #64748b; font-size: 0.9rem;">From recent leaf analysis #{st.session_state.get('last_saved_id', 'Logged')} ({latest.get('confidence', 0):.2f}% confidence)</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown("#### Explore Curated Recommendations by Crop")
        sel_c1, sel_c2 = st.columns(2)

        with sel_c1:
            selected_crop = st.selectbox("Select Crop Species", crops, index=0 if crops else None)

        diseases_for_crop = [
            d.get("disease", "Unknown")
            for d in disease_records
            if d.get("crop") == selected_crop
        ]
        diseases_for_crop.append("Healthy")

        with sel_c2:
            selected_disease = st.selectbox("Select Disease or Condition", diseases_for_crop)

        if selected_crop and selected_disease:
            recommendation = build_recommendation(selected_crop, selected_disease)

    # Render Recommendations if Available
    if recommendation:
        st.markdown("<br/>", unsafe_allow_html=True)

        st.markdown(f"### 🔬 Disease Profile: {recommendation.get('disease', selected_disease)}")
        st.markdown(f"**Scientific / Pathogen Designation:** *{recommendation.get('scientific_name', 'Not specified')}*")
        st.write(recommendation.get("description", "No description available."))

        if recommendation.get("symptoms"):
            with st.expander("🔍 Observed Symptoms & Pathology", expanded=True):
                for s in recommendation["symptoms"]:
                    st.markdown(f"• {s}")

        if recommendation.get("causes"):
            with st.expander("🌱 Pathogen Biology & Spread Vectors", expanded=True):
                for c in recommendation["causes"]:
                    st.markdown(f"• {c}")

        st.markdown("---")
        render_management_cards(recommendation)

    render_disclaimer()
