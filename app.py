from __future__ import annotations

import torch
import streamlit as st

from config.config import PROJECT_ROOT
from database.database import initialize_database
from modules.ui_about import render_about_page
from modules.ui_analytics import render_analytics_page
from modules.ui_dashboard import render_dashboard_page
from modules.ui_detection import render_detection_page
from modules.ui_history import render_history_page
from modules.ui_knowledge_base import render_knowledge_base_page
from modules.ui_recommendations import render_recommendations_page
from modules.ui_reports import render_reports_page


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AGROVISION AI — Crop Health & Advisory Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM ENTERPRISE STYLING
# --------------------------------------------------

styles_path = PROJECT_ROOT / "assets" / "styles.css"
if styles_path.exists():
    with open(styles_path, "r", encoding="utf-8") as style_file:
        st.markdown(
            f"<style>{style_file.read()}</style>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

initialize_database()


# --------------------------------------------------
# NAVIGATION CONTROLLER & SIDEBAR
# --------------------------------------------------

NAV_PAGES = [
    "Dashboard Overview",
    "Disease Detection",
    "Smart AgroCare Recommendations",
    "Crop Health Analytics",
    "Prediction History",
    "Reports",
    "Disease Knowledge Base",
    "About AGROVISION AI",
]

# Ensure current_page and nav_radio are initialized in session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard Overview"
if "nav_radio" not in st.session_state:
    st.session_state.nav_radio = st.session_state.current_page

# Sidebar Branding
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.8rem 0 1.2rem 0; border-bottom: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.3rem;">
                <span style="font-size: 1.8rem;">🌿</span>
                <span style="font-size: 1.35rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">AGROVISION AI</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.35;">
                Intelligent Crop Health.<br/>Smarter Agricultural Decisions.
            </div>
            <div style="margin-top: 0.6rem;">
                <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-size: 0.7rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; border: 1px solid rgba(16, 185, 129, 0.4);">
                    v2.0 Capstone Edition
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def on_nav_change():
        st.session_state.current_page = st.session_state.nav_radio

    # Synchronized navigation radio
    st.radio(
        "Navigation Menu",
        NAV_PAGES,
        key="nav_radio",
        on_change=on_nav_change,
        label_visibility="collapsed",
    )
    st.session_state.current_page = st.session_state.nav_radio

    # Sidebar Hardware & Model Telemetry Footer
    st.markdown("<br/>", unsafe_allow_html=True)
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.markdown(
        f"""
        <div style="border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 1rem; margin-top: 2rem; font-size: 0.78rem; color: #94a3b8;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span>Backbone:</span>
                <span style="color: #e2e8f0; font-weight: 600;">EfficientNet-B0</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span>Classes:</span>
                <span style="color: #e2e8f0; font-weight: 600;">38 Categories</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span>Compute Device:</span>
                <span style="color: #10b981; font-weight: 600;">{device_name}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Database:</span>
                <span style="color: #10b981; font-weight: 600;">SQLite (Active)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# ROUTER DISPATCHER
# --------------------------------------------------

page = st.session_state.current_page

if page == "Dashboard Overview":
    render_dashboard_page()
elif page == "Disease Detection":
    render_detection_page()
elif page == "Smart AgroCare Recommendations":
    render_recommendations_page()
elif page == "Crop Health Analytics":
    render_analytics_page()
elif page == "Prediction History":
    render_history_page()
elif page == "Reports":
    render_reports_page()
elif page == "Disease Knowledge Base":
    render_knowledge_base_page()
elif page == "About AGROVISION AI":
    render_about_page()
