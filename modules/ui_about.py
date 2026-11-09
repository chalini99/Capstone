from __future__ import annotations

import streamlit as st
from modules.ui_components import render_disclaimer, render_hero


def render_about_page() -> None:
    render_hero(
        title="About AGROVISION AI",
        subtitle="Final-Year B.Tech Capstone Project: AI-Powered Crop Disease Detection and Smart Agrochemical Recommendation System. "
        "Engineered for early pathogen screening, visual model interpretability, and responsible agronomic advisory.",
        badge="System Architecture & Documentation",
    )

    # Project Mission & Objective Card
    st.markdown(
        """
        <div class="agro-card">
            <div class="agro-card-header">
                <div class="agro-card-title">🎯 Project Mission & Objective</div>
            </div>
            <p style="font-size: 1rem; color: #334155; line-height: 1.6;">
                Plant diseases cause substantial yield losses in global agriculture every year.
                Early and accurate identification of foliar symptoms allows farmers and extension specialists
                to intervene before pathogens spread across the canopy.
            </p>
            <p style="font-size: 1rem; color: #334155; line-height: 1.6;">
                <strong>AGROVISION AI</strong> bridges the gap between deep convolutional neural networks and actionable,
                responsible agronomy. By integrating <strong>EfficientNet-B0 transfer learning</strong> with
                <strong>Grad-CAM explainability</strong> and a curated <strong>plant pathology knowledge base</strong>,
                the platform delivers rapid screening, visual confidence transparency, and verified multi-tiered management protocols.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # End-to-End System Workflow Architecture
    st.markdown("### 🔄 End-to-End System Architecture")
    st.markdown(
        """
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <div style="display: flex; flex-wrap: wrap; gap: 0.8rem; justify-content: space-between; align-items: center; text-align: center;">
                <div style="flex: 1; min-width: 110px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 0.8rem;">
                    <div style="font-size: 1.4rem;">📷</div>
                    <div style="font-weight: 700; font-size: 0.82rem; color: #0f172a;">1. Leaf Upload</div>
                    <div style="font-size: 0.72rem; color: #64748b;">JPG, JPEG, PNG</div>
                </div>
                <div style="color: #10b981; font-weight: bold;">➔</div>
                <div style="flex: 1; min-width: 110px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 0.8rem;">
                    <div style="font-size: 1.4rem;">⚙️</div>
                    <div style="font-weight: 700; font-size: 0.82rem; color: #0f172a;">2. Preprocessing</div>
                    <div style="font-size: 0.72rem; color: #64748b;">224×224 Normalization</div>
                </div>
                <div style="color: #10b981; font-weight: bold;">➔</div>
                <div style="flex: 1; min-width: 110px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 0.8rem;">
                    <div style="font-size: 1.4rem;">🧠</div>
                    <div style="font-weight: 700; font-size: 0.82rem; color: #0f172a;">3. EfficientNet-B0</div>
                    <div style="font-size: 0.72rem; color: #64748b;">38 Output Classes</div>
                </div>
                <div style="color: #10b981; font-weight: bold;">➔</div>
                <div style="flex: 1; min-width: 110px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 0.8rem;">
                    <div style="font-size: 1.4rem;">🎯</div>
                    <div style="font-weight: 700; font-size: 0.82rem; color: #0f172a;">4. Grad-CAM</div>
                    <div style="font-size: 0.72rem; color: #64748b;">Spatial Attribution</div>
                </div>
                <div style="color: #10b981; font-weight: bold;">➔</div>
                <div style="flex: 1; min-width: 110px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 0.8rem;">
                    <div style="font-size: 1.4rem;">📚</div>
                    <div style="font-weight: 700; font-size: 0.82rem; color: #0f172a;">5. Knowledge Base</div>
                    <div style="font-size: 0.72rem; color: #64748b;">Multi-tier Advisory</div>
                </div>
                <div style="color: #10b981; font-weight: bold;">➔</div>
                <div style="flex: 1; min-width: 110px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 0.8rem;">
                    <div style="font-size: 1.4rem;">💾</div>
                    <div style="font-weight: 700; font-size: 0.82rem; color: #0f172a;">6. SQLite & PDF</div>
                    <div style="font-size: 0.72rem; color: #64748b;">Audit & Reports</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Core Technologies Grid
    st.markdown("### 🛠️ Core Technology Stack")
    st.caption("All technologies verified and operational in this application environment.")

    t1, t2, t3 = st.columns(3)

    with t1:
        st.markdown(
            """
            <div class="agro-card">
                <div class="agro-card-title">Deep Learning & Vision</div>
                <ul style="font-size: 0.9rem; color: #334155; margin-top: 0.6rem; padding-left: 1.2rem;">
                    <li><strong>PyTorch</strong>: Deep learning runtime & tensor engine</li>
                    <li><strong>Torchvision</strong>: EfficientNet-B0 backbone</li>
                    <li><strong>Grad-CAM</strong>: Layer <code>features.8</code> gradient attribution</li>
                    <li><strong>OpenCV</strong>: JET pseudo-coloring & heatmap overlays</li>
                    <li><strong>Pillow (PIL)</strong>: High-resolution image IO</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t2:
        st.markdown(
            """
            <div class="agro-card">
                <div class="agro-card-title">Application & Data Engine</div>
                <ul style="font-size: 0.9rem; color: #334155; margin-top: 0.6rem; padding-left: 1.2rem;">
                    <li><strong>Streamlit</strong>: Reactive enterprise UI framework</li>
                    <li><strong>SQLite</strong>: ACID-compliant local inspection history</li>
                    <li><strong>Pandas</strong>: Structured history & telemetry analysis</li>
                    <li><strong>Plotly</strong>: Interactive distribution & trend charts</li>
                    <li><strong>JSON Knowledge Base</strong>: 26 curated plant treatments</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t3:
        st.markdown(
            """
            <div class="agro-card">
                <div class="agro-card-title">Documentation & QA</div>
                <ul style="font-size: 0.9rem; color: #334155; margin-top: 0.6rem; padding-left: 1.2rem;">
                    <li><strong>ReportLab</strong>: Dynamic PDF diagnostic reports</li>
                    <li><strong>Pytest</strong>: 16-test automated regression suite</li>
                    <li><strong>Typographical Normalization</strong>: Multi-pass matcher</li>
                    <li><strong>Safe Fallbacks</strong>: Zero crash design for missing data</li>
                    <li><strong>Chemical Safety</strong>: Strict verification disclaimers</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Responsible AI & Agricultural Safety Principles
    st.markdown("---")
    st.markdown("### 🌿 Responsible Agricultural AI Principles")
    st.markdown(
        """
        - **No Fabricated Chemicals**: AGROVISION AI never invents chemical trade names, active ingredient doses, or spraying intervals.
        - **Human-in-the-Loop Verification**: All outputs are explicitly presented as preliminary AI screening results, emphasizing consultation with certified extension professionals.
        - **Visual Transparency**: Grad-CAM attention maps explain which foliar pixel patterns the neural network focused on, preventing blind over-reliance on black-box predictions.
        """
    )

    render_disclaimer()
