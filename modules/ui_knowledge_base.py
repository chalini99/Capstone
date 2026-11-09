from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import streamlit as st

from config.config import CLASS_MAPPING_PATH, KNOWLEDGE_BASE_DIR
from modules.ui_components import render_disclaimer, render_hero, render_kpi_card, render_management_cards


def render_knowledge_base_page() -> None:
    render_hero(
        title="Plant Pathology Knowledge Base",
        subtitle="Curated compendium of agricultural crop diseases, biological pathogen vectors, "
        "cultural practices, preventive measures, and reference university extension guidelines.",
        badge="Agronomic Catalog",
    )

    # Load Knowledge Base Data
    kb_path = Path(KNOWLEDGE_BASE_DIR) / "disease_treatments.json"
    disease_records = []
    if kb_path.exists():
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                disease_records = payload.get("diseases", [])
        except Exception as e:
            st.error(f"Failed to load disease treatments knowledge base: {e}")

    # Summary Metrics Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Total Model Classes", "38", footer="Deep learning outputs", badge="CNN")
    with k2:
        render_kpi_card("Disease Records", len(disease_records), footer="Curated monographs", badge="Pathology")
    with k3:
        render_kpi_card("Healthy Categories", "12", footer="Baseline reference classes", badge="Healthy")
    with k4:
        unique_crops = len({d.get("crop") for d in disease_records if d.get("crop")})
        render_kpi_card("Crop Species", unique_crops, footer="Agricultural hosts", badge="Botany")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Search & Filter Controls
    st.markdown("#### 🔍 Search & Filter Knowledge Base")
    c_f1, c_f2 = st.columns([1, 2])

    with c_f1:
        all_crops = ["All Crops"] + sorted(list({d.get("crop") for d in disease_records if d.get("crop")}))
        sel_crop = st.selectbox("Crop Filter", all_crops)

    with c_f2:
        search_query = st.text_input("Search disease, pathogen, or symptom", placeholder="e.g. greening, blight, rust, virus...")

    # Filter Records
    filtered = disease_records.copy()
    if sel_crop != "All Crops":
        filtered = [d for d in filtered if d.get("crop") == sel_crop]

    if search_query.strip():
        q = search_query.strip().lower()
        filtered = [
            d
            for d in filtered
            if q in d.get("disease", "").lower()
            or q in d.get("scientific_name", "").lower()
            or q in d.get("description", "").lower()
            or any(q in s.lower() for s in d.get("symptoms", []))
            or any(q in c.lower() for c in d.get("causes", []))
        ]

    st.markdown(f"**Found {len(filtered)} matching disease entries**")

    # Disease Catalog Accordions
    for entry in filtered:
        crop = entry.get("crop", "Unknown Crop")
        disease = entry.get("disease", "Unknown Disease")
        sci_name = entry.get("scientific_name", "Not specified")

        with st.expander(f"🌱 {crop} — {disease} (*{sci_name}*)", expanded=False):
            st.markdown(f"**Description:** {entry.get('description', 'No description.')}")

            # Symptoms & Causes Columns
            col_sym, col_cause = st.columns(2)
            with col_sym:
                st.markdown("##### 🔍 Symptoms")
                for s in entry.get("symptoms", []):
                    st.markdown(f"• {s}")

            with col_cause:
                st.markdown("##### 🔬 Pathogen & Spread Vectors")
                for c in entry.get("causes", []):
                    st.markdown(f"• {c}")

            st.markdown("---")
            render_management_cards(entry)

    render_disclaimer()
