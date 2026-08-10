import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import joblib
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, disclosure, PRIMARY
from models.construction_monitor import (
    STAGE_NAMES, _generate_stage_image, extract_features, compare_to_promised_timeline,
)

st.set_page_config(page_title="PRISM — Construction Monitoring", page_icon="🏗️", layout="wide")
inject_css()
page_header("🏗️ Construction Progress Monitoring", "CV pipeline comparing actual construction stage vs. builder-promised timeline")

BASE = os.path.dirname(os.path.dirname(__file__))

st.markdown(
    "**Why this matters:** banks disbursing construction-linked home loans, and buyers "
    "verifying builder claims, both need an independent read on actual progress — not just "
    "the builder's self-reported update."
)

disclosure(
    "Real satellite imagery (Sentinel-2, ~10m resolution) is too coarse to detect construction "
    "stage. Production-grade monitoring needs sub-meter commercial imagery or drone footage, "
    "neither freely available here. This module demonstrates the CV pipeline end-to-end using "
    "procedurally generated stage-representative imagery, with classical feature extraction "
    "(no GPU/deep-learning framework available in this environment) standing in for a "
    "transfer-learned CNN. See README for the production path."
)

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "models", "construction_rf.joblib"))

bundle = load_model()
model, feature_cols = bundle["model"], bundle["feature_cols"]

st.markdown("#### The 5 construction stages the model recognizes")
cols = st.columns(5)
for i, (stage, name) in enumerate(STAGE_NAMES.items()):
    img = _generate_stage_image(stage, size=64)
    with cols[i]:
        st.image(Image.fromarray(img), caption=f"Stage {stage}: {name}", use_container_width=True)

st.divider()

st.markdown("#### Simulate a site inspection")
c1, c2 = st.columns(2)
with c1:
    actual_stage = st.select_slider(
        "Actual stage observed (simulated image classification)",
        options=list(STAGE_NAMES.keys()), value=2,
        format_func=lambda s: f"{s}: {STAGE_NAMES[s]}",
    )
    sample_img = _generate_stage_image(actual_stage, size=200)
    st.image(Image.fromarray(sample_img), caption=f"Simulated site image — Stage {actual_stage}", width=200)

    feats = extract_features(sample_img)
    X_input = pd.DataFrame([feats])[feature_cols]
    predicted_stage = model.predict(X_input)[0]
    confidence = model.predict_proba(X_input).max()
    st.metric("Model-classified stage", f"{predicted_stage}: {STAGE_NAMES[predicted_stage]}", f"{confidence:.0%} confidence")

with c2:
    promised_stage = st.select_slider(
        "Stage promised by builder's timeline (by this date)",
        options=list(STAGE_NAMES.keys()), value=4,
        format_func=lambda s: f"{s}: {STAGE_NAMES[s]}",
    )
    check = compare_to_promised_timeline(int(predicted_stage), promised_stage)
    status_color = "🟢" if check["delta"] >= 0 else ("🟡" if check["delta"] == -1 else "🔴")
    st.markdown(f"### {status_color} {check['status']}")
    st.metric("Schedule delta", f"{check['delta']} stage(s)")

    st.markdown("**Disbursement recommendation:**")
    if check["delta"] >= 0:
        st.success("Progress supports next construction-linked disbursement tranche.")
    elif check["delta"] == -1:
        st.warning("Minor lag — recommend a follow-up check before releasing next tranche.")
    else:
        st.error("Material lag vs. promised schedule — hold disbursement pending builder review.")

st.divider()
st.markdown("#### Model performance on held-out synthetic imagery")
st.caption("Random Forest classifier on classical CV features (color/texture/edge-density) — 750 synthetic images, 5 stages, stratified train/test split.")
st.metric("Test accuracy", "100%*")
st.caption("*Near-perfect accuracy reflects the deterministic synthetic-image generator, not real-world CV difficulty — see disclosure above.")
