import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, disclosure, PRIMARY, ACCENT
from models.fraud_model import FEATURES, detect_near_duplicates

st.set_page_config(page_title="PRISM — Fraud Detection", page_icon="🚩", layout="wide")
inject_css()
page_header("🚩 Fraud Detection in Listings", "Classifier + duplicate-detection layer for classifieds trust & safety")

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
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total listings scanned", f"{len(listings):,}")
    with c2:
        st.metric("Model-flagged (P > 0.5)", f"{(listings['fraud_probability'] > 0.5).sum():,}")
    with c3:
        st.metric("Model AUC (test)", "0.98")
    with c4:
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
        "price_per_sqft_listed": "Listed price/sqft", "carpet_area_sqft": "Carpet area",
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
        "from locality-implied fair value flag bait pricing."
    )

with tab2:
    st.markdown("#### Explore flagged listings")
    min_prob = st.slider("Minimum fraud probability", 0.0, 1.0, 0.5, 0.05)
    flagged = listings[listings["fraud_probability"] >= min_prob].sort_values("fraud_probability", ascending=False)
    st.caption(f"{len(flagged)} listings above threshold")

    display_cols = [
        "listing_id", "city", "locality", "bhk", "listed_price", "price_deviation_pct",
        "broker_listing_count", "description_reuse_count", "fraud_probability", "fraud_type",
    ]
    st.dataframe(
        flagged[display_cols].head(100).style.format({
            "listed_price": "₹{:,.0f}", "price_deviation_pct": "{:.1f}%", "fraud_probability": "{:.2f}",
        }),
        use_container_width=True, height=400,
    )

    if len(flagged) > 0:
        sel_id = st.selectbox("Inspect a listing", flagged["listing_id"].head(50).tolist())
        row = listings[listings["listing_id"] == sel_id].iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{row['bhk']}BHK in {row['locality']}, {row['city']}**")
            st.write(row["description"])
            st.metric("Listed price", f"₹{row['listed_price']:,.0f}")
            st.metric("Fair value estimate", f"₹{row['fair_value_est']:,.0f}")
        with c2:
            st.metric("Fraud probability", f"{row['fraud_probability']:.1%}")
            st.markdown(f"**Ground-truth label:** `{row['fraud_type']}`" if row["is_fraud"] else "**Ground-truth label:** `clean`")
            st.metric("Broker listed this number", f"{row['broker_listing_count']}x")
            st.metric("This description used", f"{row['description_reuse_count']}x")

with tab3:
    st.markdown("#### Near-duplicate detection (TF-IDF + cosine similarity)")
    st.caption(
        "Runs independently of the trained classifier — a text-similarity layer that catches "
        "duplicate/near-duplicate listings even without labeled fraud examples, the way a real "
        "moderation pipeline layers rule-based and ML-based checks."
    )
    sim_threshold = st.slider("Similarity threshold", 0.80, 1.0, 0.92, 0.01)
    filtered_dups = dup_pairs[dup_pairs["similarity"] >= sim_threshold]
    st.metric("Flagged pairs at this threshold", f"{len(filtered_dups):,}")

    if len(filtered_dups) > 0:
        sample_pairs = filtered_dups.sample(min(10, len(filtered_dups)), random_state=1)
        for _, pair in sample_pairs.iterrows():
            a = listings[listings["listing_id"] == pair["listing_id_a"]].iloc[0]
            b = listings[listings["listing_id"] == pair["listing_id_b"]].iloc[0]
            with st.expander(f"Similarity {pair['similarity']:.2f} — Listing {pair['listing_id_a']} ↔ {pair['listing_id_b']}"):
                c1, c2 = st.columns(2)
                c1.write(f"**#{a['listing_id']}** ({a['locality']}) — ₹{a['listed_price']:,.0f}")
                c1.caption(a["description"])
                c2.write(f"**#{b['listing_id']}** ({b['locality']}) — ₹{b['listed_price']:,.0f}")
                c2.caption(b["description"])

disclosure(
    "Listings and fraud labels are synthetic, built by injecting realistic fraud patterns "
    "(duplicate reposts, bait pricing, fake listings, scam-ring broker reuse, recycled "
    "descriptions) onto the synthetic property base. The near-perfect AUC partly reflects "
    "clean synthetic label separation — production deployment would need real labeled "
    "moderation data and ongoing adversarial re-training as fraud patterns evolve."
)
