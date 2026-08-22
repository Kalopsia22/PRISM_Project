import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, PRIMARY, ACCENT
from models.fraud_model import FEATURES

st.set_page_config(page_title="PRISM — Fraud Detection", page_icon="🚩", layout="wide")
inject_css()
page_header("🚩 Fraud Detection in Listings", "Classifier + duplicate-detection layer for classifieds trust & safety — sale and rental listings alike")

BASE = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_data():
    listings = pd.read_csv(os.path.join(BASE, "data", "listings.csv"))
    dups = pd.read_csv(os.path.join(BASE, "data", "flagged_duplicates.csv"))
    return listings, dups

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "models", "fraud_xgb.joblib"))

listings, dup_pairs = load_data()
model = load_model()

X_all = listings[FEATURES]
listings = listings.copy()
listings["fraud_probability"] = model.predict_proba(X_all)[:, 1]

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔎 Flagged Listings Explorer", "🔗 Duplicate Detection"])

with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total listings scanned", f"{len(listings):,}")
    with c2:
        st.metric("Sale listings", f"{(listings['listing_type']=='sale').sum():,}")
    with c3:
        st.metric("Rental listings", f"{(listings['listing_type']=='rent').sum():,}")
    with c4:
        st.metric("Model-flagged (P > 0.5)", f"{(listings['fraud_probability'] > 0.5).sum():,}")
    with c5:
        st.metric("Duplicate pairs found", f"{len(dup_pairs):,}")

    st.markdown("#### Fraud type breakdown")
    type_counts = listings[listings["is_fraud"] == 1]["fraud_type"].value_counts().reset_index()
    type_counts.columns = ["fraud_type", "count"]
    fig = px.bar(
        type_counts, x="count", y="fraud_type", orientation="h",
        title="Known fraud pattern distribution (labeled synthetic data)",
        color_discrete_sequence=[ACCENT],
    )
    fig.update_layout(yaxis_title="", xaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### What signals the model relies on most")
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False).head(8)
    label_map = {
        "description_reuse_count": "Description reused across listings", "broker_listing_count": "Broker phone reuse count",
        "num_images": "Number of images", "abs_price_deviation_pct": "Price deviation from fair value",
        "area_per_bhk": "Area-per-BHK ratio", "days_on_market": "Days on market",
        "ask_per_sqft": "Asking price/sqft", "carpet_area_sqft": "Carpet area",
        "ask_value": "Asking value", "is_rent": "Rental listing flag",
    }
    imp_df = pd.DataFrame({"feature": importances.index, "importance": importances.values})
    imp_df["label"] = imp_df["feature"].map(label_map).fillna(imp_df["feature"])
    fig2 = px.bar(
        imp_df.sort_values("importance"), x="importance", y="label", orientation="h",
        title="XGBoost feature importance",
        color_discrete_sequence=[PRIMARY],
    )
    fig2.update_layout(yaxis_title="", xaxis_title="Importance")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "The top signals mirror real-world fraud patterns: reused descriptions and repeated "
        "broker phone numbers are classic scam-ring indicators, while large price deviations "
        "from locality-implied fair value flag bait pricing — for either sale or rental listings."
    )

with tab2:
    st.markdown("#### Explore flagged listings")
    c1, c2 = st.columns(2)
    with c1:
        listing_type_filter = st.selectbox("Listing type", ["All", "sale", "rent"])
    with c2:
        min_prob = st.slider("Minimum fraud probability", 0.0, 1.0, 0.5, 0.05)

    pool = listings if listing_type_filter == "All" else listings[listings["listing_type"] == listing_type_filter]
    flagged = pool[pool["fraud_probability"] >= min_prob].sort_values("fraud_probability", ascending=False)
    st.caption(f"{len(flagged)} listings above threshold")

    display_cols = [
        "listing_id", "listing_type", "city", "locality", "bhk", "ask_value", "price_deviation_pct",
        "broker_listing_count", "description_reuse_count", "fraud_probability", "fraud_type",
    ]
    st.dataframe(
        flagged[display_cols].head(100).style.format({
            "ask_value": "₹{:,.0f}", "price_deviation_pct": "{:.1f}%", "fraud_probability": "{:.2f}",
        }),
        use_container_width=True, height=400,
    )

    if len(flagged) > 0:
        sel_id = st.selectbox("Inspect a listing", flagged["listing_id"].head(50).tolist())
        row = listings[listings["listing_id"] == sel_id].iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{row['property_type']} · {row['bhk']}BHK in {row['locality']}, {row['city']}** ({row['listing_type']})")
            st.write(row["description"])
            st.metric("Asking value", f"₹{row['ask_value']:,.0f}")
            st.metric("Fair value estimate", f"₹{row['fair_value']:,.0f}")
        with c2:
            st.metric("Fraud probability", f"{row['fraud_probability']:.1%}")
            st.markdown(f"**Ground-truth label:** `{row['fraud_type']}`" if row["is_fraud"] else "**Ground-truth label:** `clean`")
            st.metric("Broker listed this number", f"{row['broker_listing_count']}x")
            st.metric("This description used", f"{row['description_reuse_count']}x")

with tab3:
    st.markdown("#### Near-duplicate detection (TF-IDF + structural match)")
    st.caption(
        "Short, templated listing text alone saturates cosine similarity (most listings share "
        "sentence structure regardless of property), so this layer requires both high text "
        "similarity AND matching locality/price/area — catching genuine re-listed duplicates "
        "without flooding on template collisions. Runs independently of the trained classifier."
    )
    sim_threshold = st.slider("Text similarity threshold", 0.80, 1.0, 0.90, 0.01)
    filtered_dups = dup_pairs[dup_pairs["similarity"] >= sim_threshold]
    st.metric("Flagged pairs at this threshold", f"{len(filtered_dups):,}")

    if len(filtered_dups) > 0:
        sample_pairs = filtered_dups.sample(min(10, len(filtered_dups)), random_state=1)
        for _, pair in sample_pairs.iterrows():
            a = listings[listings["listing_id"] == pair["listing_id_a"]].iloc[0]
            b = listings[listings["listing_id"] == pair["listing_id_b"]].iloc[0]
            with st.expander(f"Similarity {pair['similarity']:.2f} — Listing {pair['listing_id_a']} ↔ {pair['listing_id_b']}"):
                c1, c2 = st.columns(2)
                c1.write(f"**#{a['listing_id']}** ({a['locality']}) — ₹{a['ask_value']:,.0f}")
                c1.caption(a["description"])
                c2.write(f"**#{b['listing_id']}** ({b['locality']}) — ₹{b['ask_value']:,.0f}")
                c2.caption(b["description"])
