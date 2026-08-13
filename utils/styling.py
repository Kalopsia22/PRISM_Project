"""Shared theme/styling helpers so all PRISM pages look like one product."""

import streamlit as st

PRIMARY = "#0B3D66"      # deep navy blue — trust/institutional
PRIMARY_LIGHT = "#3B82F6"
ACCENT = "#D97706"       # amber — risk/attention highlights
SUCCESS = "#15803D"
DANGER = "#B91C1C"
BG_CARD = "#F8FAFC"

def inject_css():
    st.markdown(f"""
    <style>
        .prism-header {{
            background: linear-gradient(135deg, {PRIMARY} 0%, #1E5A8E 60%, #2D7DBE 100%);
            padding: 1.8rem 2.2rem; border-radius: 14px; margin-bottom: 1.4rem;
            color: white; box-shadow: 0 4px 18px rgba(11,61,102,0.25);
        }}
        .prism-header h1 {{ margin: 0; font-size: 1.9rem; letter-spacing: -0.02em; }}
        .prism-header p {{ margin: 0.35rem 0 0 0; opacity: 0.88; font-size: 0.97rem; }}
        .metric-card {{
            background: {BG_CARD}; border: 1px solid #E2E8F0; border-radius: 12px;
            padding: 1.1rem 1.3rem; margin-bottom: 0.6rem;
            transition: box-shadow 0.15s ease;
        }}
        .metric-card:hover {{ box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
        .section-title {{
            display: flex; align-items: center; gap: 0.5rem;
            font-size: 1.15rem; font-weight: 700; color: {PRIMARY};
            margin: 1.4rem 0 0.6rem 0; padding-bottom: 0.4rem;
            border-bottom: 2px solid #E2E8F0;
        }}
        .section-title .icon-badge {{
            display: inline-flex; align-items: center; justify-content: center;
            width: 28px; height: 28px; border-radius: 8px;
            background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_LIGHT} 100%);
            font-size: 0.95rem;
        }}
        .pill {{
            display: inline-block; padding: 0.15rem 0.65rem; border-radius: 999px;
            font-size: 0.78rem; font-weight: 600; margin-right: 0.3rem;
        }}
        .pill-tier1 {{ background: #DBEAFE; color: {PRIMARY}; }}
        .pill-tier2 {{ background: #FEF3C7; color: #92400E; }}
        .pill-tier3 {{ background: #DCFCE7; color: #166534; }}
        .disclosure-box {{
            background: #FFF7ED; border-left: 4px solid {ACCENT}; border-radius: 6px;
            padding: 0.8rem 1rem; font-size: 0.88rem; margin: 0.8rem 0;
        }}
        .fraud-flag {{ color: {DANGER}; font-weight: 600; }}
        .clean-flag {{ color: {SUCCESS}; font-weight: 600; }}
        div[data-testid="stMetricValue"] {{ color: {PRIMARY}; }}
        div[data-testid="stMetric"] {{
            background: white; border: 1px solid #EEF2F6; border-radius: 10px;
            padding: 0.7rem 0.9rem 0.5rem 0.9rem;
        }}
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str):
    skyline = """
    <svg viewBox="0 0 400 100" preserveAspectRatio="none" style="position:absolute; right:0; bottom:0; height:100%; width:45%; opacity:0.16;">
        <rect x="10" y="40" width="30" height="60" fill="white"/>
        <rect x="45" y="25" width="24" height="75" fill="white"/>
        <rect x="74" y="55" width="20" height="45" fill="white"/>
        <rect x="100" y="15" width="28" height="85" fill="white"/>
        <rect x="133" y="45" width="22" height="55" fill="white"/>
        <rect x="160" y="30" width="26" height="70" fill="white"/>
        <rect x="191" y="60" width="18" height="40" fill="white"/>
        <rect x="214" y="10" width="30" height="90" fill="white"/>
        <rect x="249" y="48" width="24" height="52" fill="white"/>
        <rect x="278" y="35" width="20" height="65" fill="white"/>
        <rect x="303" y="58" width="26" height="42" fill="white"/>
        <rect x="334" y="22" width="22" height="78" fill="white"/>
        <rect x="361" y="50" width="28" height="50" fill="white"/>
    </svg>
    """
    st.markdown(f"""
    <div class="prism-header" style="position:relative; overflow:hidden;">
        {skyline}
        <div style="position:relative; z-index:1;">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def section_title(icon: str, title: str):
    st.markdown(f"""
    <div class="section-title"><span class="icon-badge">{icon}</span> {title}</div>
    """, unsafe_allow_html=True)


def tier_pill(city_tier: str) -> str:
    cls = {"Tier1": "pill-tier1", "Tier2": "pill-tier2", "Tier3": "pill-tier3"}.get(city_tier, "pill-tier1")
    return f'<span class="pill {cls}">{city_tier}</span>'


def disclosure(text: str):
    st.markdown(f'<div class="disclosure-box">⚠️ <b>Data note:</b> {text}</div>', unsafe_allow_html=True)
