"""Shared theme/styling helpers so all PRISM pages look like one product."""

import streamlit as st

PRIMARY = "#0B3D66"      # deep navy blue — trust/institutional
ACCENT = "#D97706"       # amber — risk/attention highlights
SUCCESS = "#15803D"
DANGER = "#B91C1C"
BG_CARD = "#F8FAFC"

def inject_css():
    st.markdown(f"""
    <style>
        .prism-header {{
            background: linear-gradient(135deg, {PRIMARY} 0%, #1E5A8E 100%);
            padding: 1.6rem 2rem; border-radius: 12px; margin-bottom: 1.2rem;
            color: white;
        }}
        .prism-header h1 {{ margin: 0; font-size: 1.7rem; }}
        .prism-header p {{ margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }}
        .metric-card {{
            background: {BG_CARD}; border: 1px solid #E2E8F0; border-radius: 10px;
            padding: 1rem 1.2rem; margin-bottom: 0.6rem;
        }}
        .disclosure-box {{
            background: #FFF7ED; border-left: 4px solid {ACCENT}; border-radius: 6px;
            padding: 0.8rem 1rem; font-size: 0.88rem; margin: 0.8rem 0;
        }}
        .fraud-flag {{ color: {DANGER}; font-weight: 600; }}
        .clean-flag {{ color: {SUCCESS}; font-weight: 600; }}
        div[data-testid="stMetricValue"] {{ color: {PRIMARY}; }}
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str):
    st.markdown(f"""
    <div class="prism-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def disclosure(text: str):
    st.markdown(f'<div class="disclosure-box">⚠️ <b>Data note:</b> {text}</div>', unsafe_allow_html=True)
