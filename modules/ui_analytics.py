from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database.database import fetch_history
from modules.ui_components import render_disclaimer, render_hero, render_kpi_card


def render_analytics_page() -> None:
    render_hero(
        title="Crop Health Analytics & Screening Insights",
        subtitle="Empirical operational intelligence derived from logged field screenings. "
        "Track pathogen incidence, crop distribution trends, and model prediction confidence distributions.",
        badge="Telemetry & Analytics",
    )

    rows = fetch_history()

    if not rows:
        st.info("No historical analysis records found in database. Run disease screenings to populate analytics telemetry.")
        render_disclaimer()
        return

    df = pd.DataFrame(rows)
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    df["is_healthy"] = df["predicted_disease"].str.contains("healthy", case=False, na=False)
    df["diagnosis_type"] = df["is_healthy"].map({True: "Healthy Specimen", False: "Disease Detected"})

    # Filter Controls Bar
    st.markdown("#### 🔍 Filter Operational Telemetry")
    f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 2])

    with f_col1:
        crop_options = ["All Crops"] + sorted(df["crop_name"].dropna().unique().tolist())
        selected_crop = st.selectbox("Filter by Crop", crop_options)

    with f_col2:
        diag_options = ["All Diagnoses", "Disease Cases Only", "Healthy Only"]
        selected_diag = st.selectbox("Diagnosis Category", diag_options)

    with f_col3:
        search_query = st.text_input("Search disease name or image filename", placeholder="e.g. blight, rot, leaf...")

    # Apply Filters
    filtered_df = df.copy()
    if selected_crop != "All Crops":
        filtered_df = filtered_df[filtered_df["crop_name"] == selected_crop]

    if selected_diag == "Disease Cases Only":
        filtered_df = filtered_df[~filtered_df["is_healthy"]]
    elif selected_diag == "Healthy Only":
        filtered_df = filtered_df[filtered_df["is_healthy"]]

    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["predicted_disease"].str.lower().str.contains(q, na=False)
            | filtered_df["image_filename"].str.lower().str.contains(q, na=False)
            | filtered_df["crop_name"].str.lower().str.contains(q, na=False)
        ]

    # Metrics Summary Row
    st.markdown("<br/>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)

    total_filtered = len(filtered_df)
    healthy_filtered = int(filtered_df["is_healthy"].sum()) if total_filtered > 0 else 0
    disease_filtered = total_filtered - healthy_filtered
    avg_conf = round(filtered_df["confidence_score"].mean(), 1) if total_filtered > 0 else 0.0

    with m1:
        render_kpi_card("Filtered Records", total_filtered, footer=f"Out of {len(df)} total", badge="Subset")
    with m2:
        render_kpi_card("Disease Cases", disease_filtered, footer="Pathogens identified", badge="Pathology")
    with m3:
        render_kpi_card("Healthy Crops", healthy_filtered, footer="Normal tissue", badge="Healthy")
    with m4:
        render_kpi_card("Mean Confidence", f"{avg_conf:.1f}%", footer="Prediction certainty", badge="Model Score")

    st.markdown("<br/>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No records match the selected filter criteria. Adjust the filters above.")
        render_disclaimer()
        return

    # Visualizations Grid
    v_col1, v_col2 = st.columns(2)

    with v_col1:
        st.markdown("#### Disease Incidence Frequency")
        disease_counts = (
            filtered_df[~filtered_df["is_healthy"]]["predicted_disease"]
            .value_counts()
            .head(8)
            .reset_index()
        )
        disease_counts.columns = ["Disease", "Count"]

        if not disease_counts.empty:
            fig_dis = px.bar(
                disease_counts,
                x="Count",
                y="Disease",
                orientation="h",
                color="Count",
                color_continuous_scale=["#fed7aa", "#ea580c", "#9a3412"],
            )
            fig_dis.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=320,
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dis, use_container_width=True)
        else:
            st.info("No disease cases present in this filtered selection.")

    with v_col2:
        st.markdown("#### Crop Distribution")
        crop_counts = filtered_df["crop_name"].value_counts().reset_index()
        crop_counts.columns = ["Crop", "Count"]

        fig_crop = px.pie(
            crop_counts,
            values="Count",
            names="Crop",
            hole=0.5,
            color_discrete_sequence=px.colors.sequential.Aggrnyl,
        )
        fig_crop.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_crop, use_container_width=True)

    # Confidence Score Distribution & Timeline
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("#### Model Confidence Spread")
        fig_hist = px.histogram(
            filtered_df,
            x="confidence_score",
            nbins=12,
            color="diagnosis_type",
            color_discrete_map={"Healthy Specimen": "#10b981", "Disease Detected": "#f59e0b"},
            barmode="overlay",
            opacity=0.75,
            labels={"confidence_score": "Confidence (%)", "diagnosis_type": "Condition"},
        )
        fig_hist.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c_col2:
        st.markdown("#### Analysis Timeline Trend")
        timeline_df = filtered_df.copy()
        timeline_df["date"] = pd.to_datetime(timeline_df["analysis_timestamp"], errors="coerce").dt.date
        time_counts = timeline_df.groupby(["date", "diagnosis_type"]).size().reset_index(name="count")

        fig_line = px.bar(
            time_counts,
            x="date",
            y="count",
            color="diagnosis_type",
            color_discrete_map={"Healthy Specimen": "#10b981", "Disease Detected": "#f59e0b"},
            labels={"date": "Analysis Date", "count": "Inspections", "diagnosis_type": "Result"},
        )
        fig_line.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # Filtered Records Table with CSV Export
    st.markdown("---")
    t_header_col1, t_header_col2 = st.columns([3, 1])
    with t_header_col1:
        st.markdown("#### 📑 Filtered Telemetry Records")
    with t_header_col2:
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Data (CSV)",
            data=csv_data,
            file_name="agrovision_analysis_telemetry.csv",
            mime="text/csv",
            use_container_width=True,
        )

    export_cols = ["id", "analysis_timestamp", "image_filename", "crop_name", "predicted_disease", "confidence_score"]
    clean_table = filtered_df[[c for c in export_cols if c in filtered_df.columns]].rename(
        columns={
            "id": "ID",
            "analysis_timestamp": "Timestamp",
            "image_filename": "File",
            "crop_name": "Crop",
            "predicted_disease": "Condition",
            "confidence_score": "Confidence (%)",
        }
    )
    st.dataframe(clean_table, use_container_width=True)

    render_disclaimer()
