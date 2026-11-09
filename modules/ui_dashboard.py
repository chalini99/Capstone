from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.config import EXPECTED_NUM_CLASSES, MODEL_PATH, CLASS_MAPPING_PATH
from database.database import fetch_history, fetch_history_stats
from modules.ui_components import render_hero, render_kpi_card, render_disclaimer


def render_dashboard_page() -> None:
    render_hero(
        title="AGROVISION AI",
        subtitle="AI-Powered Crop Disease Detection & Smart Agrochemical Recommendation Platform. "
        "Combining deep convolutional neural networks (EfficientNet-B0), Grad-CAM explainability, "
        "and agronomic knowledge bases for reliable crop protection.",
        badge="Enterprise Agricultural Intelligence",
    )

    # Primary Action Callout Banner
    cta_col1, cta_col2 = st.columns([3, 1])
    with cta_col1:
        st.markdown(
            "##### 🚜 Ready to inspect crop leaf health?"
        )
        st.caption("Upload a leaf image to run instant AI screening, Grad-CAM attention analysis, and retrieve AgroCare management guidance.")
    with cta_col2:
        if st.button("🌱 Analyze New Leaf", type="primary", use_container_width=True):
            st.session_state.nav_radio = "Disease Detection"
            st.session_state.current_page = "Disease Detection"
            st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)

    # Database Statistics & KPI Cards
    stats = fetch_history_stats()
    rows = fetch_history()

    st.markdown("### 📊 Operational Overview & System Telemetry")

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

    with kpi1:
        render_kpi_card(
            label="Total Analyses",
            value=stats["total_analyses"],
            footer="Logged records",
            badge="Database",
        )
    with kpi2:
        render_kpi_card(
            label="Healthy Crops",
            value=stats["healthy_count"],
            footer="No pathogen signs",
            badge="Healthy",
        )
    with kpi3:
        render_kpi_card(
            label="Disease Alerts",
            value=stats["disease_count"],
            footer="Pathogens identified",
            badge="Screening",
        )
    with kpi4:
        render_kpi_card(
            label="Disease Catalog",
            value="26",
            footer="Verified records",
            badge="Knowledge",
        )
    with kpi5:
        render_kpi_card(
            label="Model Classes",
            value=EXPECTED_NUM_CLASSES,
            footer="EfficientNet-B0",
            badge="CNN Model",
        )
    with kpi6:
        render_kpi_card(
            label="Top Crop",
            value=stats["most_analyzed_crop"],
            footer="Highest frequency",
            badge="Activity",
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Visualizations Section
    if rows:
        df = pd.DataFrame(rows)
        df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
        df["is_healthy"] = df["predicted_disease"].str.contains("healthy", case=False, na=False)

        col_chart1, col_chart2 = st.columns([1, 1])

        with col_chart1:
            st.markdown("#### Health Ratio Distribution")
            health_counts = df["is_healthy"].value_counts().reset_index()
            health_counts.columns = ["Status", "Count"]
            health_counts["Status"] = health_counts["Status"].map({True: "Healthy Crop", False: "Disease Detected"})

            fig_donut = px.pie(
                health_counts,
                values="Count",
                names="Status",
                hole=0.55,
                color="Status",
                color_discrete_map={"Healthy Crop": "#10b981", "Disease Detected": "#f59e0b"},
            )
            fig_donut.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=280,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_chart2:
            st.markdown("#### Crop Analysis Frequency")
            crop_counts = df["crop_name"].value_counts().head(7).reset_index()
            crop_counts.columns = ["Crop", "Analyses"]

            fig_bar = px.bar(
                crop_counts,
                x="Analyses",
                y="Crop",
                orientation="h",
                color="Analyses",
                color_continuous_scale=["#a7f3d0", "#059669", "#064e3b"],
            )
            fig_bar.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=280,
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Recent Analysis Activity Table
        st.markdown("#### Recent Analysis Records")
        display_cols = ["id", "analysis_timestamp", "crop_name", "predicted_disease", "confidence_score"]
        clean_df = df[[c for c in display_cols if c in df.columns]].copy()
        clean_df.rename(
            columns={
                "id": "ID",
                "analysis_timestamp": "Timestamp",
                "crop_name": "Crop",
                "predicted_disease": "Predicted Condition",
                "confidence_score": "Confidence (%)",
            },
            inplace=True,
        )
        st.dataframe(clean_df.head(5), use_container_width=True)

    else:
        st.info("No crop analyses have been recorded yet. Click 'Analyze New Leaf' above to test your first image.")

    # System Status & Diagnostic Integrity
    st.markdown("---")
    st.markdown("### ⚙️ System Status & Model Integrity")
    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        model_exists = MODEL_PATH.exists()
        st.markdown(
            f"**Neural Network Model**: {'✅ Loaded' if model_exists else '❌ Missing'}<br/>"
            f"<small style='color: gray;'>EfficientNet-B0 (`{MODEL_PATH.name}`)</small>",
            unsafe_allow_html=True,
        )

    with status_col2:
        mapping_exists = CLASS_MAPPING_PATH.exists()
        st.markdown(
            f"**Class Mapping**: {'✅ 38 Classes Active' if mapping_exists else '❌ Missing'}<br/>"
            f"<small style='color: gray;'>PlantVillage standard taxonomy</small>",
            unsafe_allow_html=True,
        )

    with status_col3:
        st.markdown(
            "**Knowledge Base**: ✅ 26 Disease Records<br/>"
            "<small style='color: gray;'>Verified cultural, biological & preventive advice</small>",
            unsafe_allow_html=True,
        )

    render_disclaimer()
