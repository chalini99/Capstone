from __future__ import annotations

from typing import Any
import streamlit as st


def render_hero(title: str, subtitle: str, badge: str = "AGROVISION AI v2.0") -> None:
    st.markdown(
        f"""
        <div class="agro-hero">
            <div class="agro-badge-pill">{badge}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: Any, footer: str = "", badge: str = "") -> None:
    badge_html = f'<span class="agro-badge agro-badge-neutral">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="agro-kpi-card">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="agro-kpi-label">{label}</div>
                    {badge_html}
                </div>
                <div class="agro-kpi-value">{value}</div>
            </div>
            {f'<div class="agro-kpi-footer">{footer}</div>' if footer else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        """
        <div class="agro-disclaimer-box">
            <strong>⚠️ Agricultural Advisory Disclaimer:</strong>
            AGROVISION AI is an automated machine-learning screening and educational decision-support tool.
            Model predictions and preliminary classifications do not replace laboratory pathogen testing,
            PCR analysis, or on-site diagnosis by certified agronomists or plant pathology authorities.
            Always consult a local agricultural extension officer before applying regulated pesticides.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(is_healthy: bool, text: str) -> None:
    badge_class = "agro-badge-healthy" if is_healthy else "agro-badge-disease"
    icon = "✓" if is_healthy else "⚠️"
    st.markdown(
        f'<span class="agro-badge {badge_class}">{icon} {text}</span>',
        unsafe_allow_html=True,
    )


def render_top_predictions_bars(top_predictions: list[dict[str, Any]]) -> None:
    st.markdown("#### Top Model Predictions")
    for idx, pred in enumerate(top_predictions, start=1):
        crop = pred.get("crop", "Unknown")
        disease = pred.get("disease", "Unknown")
        conf_pct = pred.get("confidence_pct", 0.0)
        is_healthy = "healthy" in disease.lower()

        color = "#10b981" if is_healthy else "#e65100" if idx == 1 else "#3b82f6"

        col_meta, col_bar = st.columns([3, 5])
        with col_meta:
            st.markdown(
                f"**#{idx} {crop}** — *{disease}*"
            )
        with col_bar:
            st.progress(
                float(min(max(conf_pct / 100.0, 0.0), 1.0)),
                text=f"{conf_pct:.2f}% confidence",
            )


def render_management_cards(recommendation: dict[str, Any]) -> None:
    status = recommendation.get("status", "")
    is_healthy = status == "healthy"

    if is_healthy:
        st.success(
            "🌿 **Plant Evaluated as Healthy**: No disease treatments or chemical interventions required. "
            "Follow standard crop-care and regular field hygiene."
        )

    # 1. Chemical Management
    chemical_items = recommendation.get("chemical_management", [])
    st.markdown("### 🧪 Chemical Management Guidance")
    if chemical_items:
        for item in chemical_items:
            opt = item.get("option", "Chemical Option")
            reason = item.get("reason", "No reason provided.")
            verified = item.get("verified", False)
            source = item.get("source", "Not specified")

            tag_badge = (
                '<span class="agro-badge agro-badge-healthy">✓ Official Label Verified</span>'
                if verified
                else '<span class="agro-badge agro-badge-warning">⚠️ Unverified / Advisory Only</span>'
            )
            st.markdown(
                f"""
                <div class="agro-rec-card chemical">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
                        <div class="agro-rec-title">{opt}</div>
                        {tag_badge}
                    </div>
                    <div class="agro-rec-reason">{reason}</div>
                    <div class="agro-rec-source">Reference: {source}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No chemical treatments are recommended or verified for this condition.")

    # 2. Biological Management
    st.markdown("### 🐞 Biological Management")
    bio_items = recommendation.get("biological_management", [])
    if bio_items:
        for item in bio_items:
            opt = item.get("option", "Biological agent")
            reason = item.get("reason", "")
            source = item.get("source", "Agricultural Extension")
            st.markdown(
                f"""
                <div class="agro-rec-card biological">
                    <div class="agro-rec-title">🔬 {opt}</div>
                    <div class="agro-rec-reason">{reason}</div>
                    <div class="agro-rec-source">Source: {source}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No specific biological control agents are listed in the verified knowledge base for this condition.")

    # 3. Cultural Management
    st.markdown("### 🌾 Cultural & Agronomic Practices")
    cult_items = recommendation.get("cultural_management", [])
    if cult_items:
        for item in cult_items:
            opt = item.get("option", "Cultural practice")
            reason = item.get("reason", "")
            source = item.get("source", "IPM Practices")
            st.markdown(
                f"""
                <div class="agro-rec-card cultural">
                    <div class="agro-rec-title">🌱 {opt}</div>
                    <div class="agro-rec-reason">{reason}</div>
                    <div class="agro-rec-source">Source: {source}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Follow general field sanitation, crop rotation, and optimal plant spacing.")

    # 4. Preventive Measures
    st.markdown("### 🛡️ Recommended Preventive Practices")
    preventive = recommendation.get("preventive_measures", [])
    if preventive:
        for measure in preventive:
            st.markdown(f"- {measure}")
    else:
        st.info("Maintain regular crop scouting and irrigation sanitation.")

    # 5. References
    refs = recommendation.get("references", [])
    if refs:
        st.markdown("### 📚 Agricultural References & Guidelines")
        for ref in refs:
            name = ref.get("name", "Agricultural Resource")
            url = ref.get("url", "")
            if url:
                st.markdown(f"- 🔗 [{name}]({url})")
            else:
                st.markdown(f"- 📄 {name}")
