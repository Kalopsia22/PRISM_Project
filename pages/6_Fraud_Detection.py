import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, section_title, insight_grid, model_metrics_bar, PRIMARY, ACCENT, SUCCESS, DANGER
from utils.charts import gauge_chart, donut_chart, PRIMARY_LIGHT
from models.fraud_model import FEATURES

st.set_page_config(page_title="PRISM — Fraud Detection", page_icon="🚩", layout="wide")
inject_css()
page_header("🚩 Fraud Detection in Listings", "Classifier + duplicate-detection layer for classifieds trust & safety — sale and rental listings alike")

BASE = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_data():
    listings = pd.read_csv(os.path.join(BASE, "data", "listings.csv"))
    dups = pd.read_csv(os.path.join(BASE, "data", "flagged_duplicates.csv"))
    metrics = pd.read_csv(os.path.join(BASE, "data", "fraud_model_metrics.csv")).iloc[0]
    return listings, dups, metrics

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "models", "fraud_xgb.joblib"))

listings, dup_pairs, metrics = load_data()
model = load_model()

X_all = listings[FEATURES]
listings = listings.copy()
listings["fraud_probability"] = model.predict_proba(X_all)[:, 1]

model_metrics_bar(
    metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"], metrics["auc"],
    caption="Held-out test set (25% split, never seen during training) — listing-level fraud classification (fraud vs. clean)."
)

tab1, tab2, tab3 = st.tabs(["📊 Overview & Performance", "🔎 Flagged Listings Explorer", "🔗 Duplicate Detection"])

with tab1:
    section_title("🎯", "Held-out test performance")
    st.caption(
        "Measured on a 25% held-out test split the model never saw during training — the honest "
        "number, not a training-set score. A classifier this clean would be suspicious; this one "
        "misses some fraud and flags a few clean listings, the way a real deployed model does."
    )

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.plotly_chart(gauge_chart(metrics["auc"] * 100, "AUC", min_val=50, max_val=100,
                                      band_edges=(70, 85, 93), height=200), use_container_width=True)
    with g2:
        st.metric("Precision", f"{metrics['precision']:.1%}")
        st.caption("Of listings flagged, how many were actually fraud.")
    with g3:
        st.metric("Recall", f"{metrics['recall']:.1%}")
        st.caption("Of actual fraud, how much the model caught.")
    with g4:
        st.metric("F1 Score", f"{metrics['f1']:.1%}")
        st.caption("Balance between precision and recall.")

    section_title("🧮", "Confusion matrix (test set)")
    cm = np.array([[metrics["true_negative"], metrics["false_positive"]],
                    [metrics["false_negative"], metrics["true_positive"]]])
    c1, c2 = st.columns([1, 1])
    with c1:
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Predicted Clean", "Predicted Fraud"], y=["Actually Clean", "Actually Fraud"],
            text=cm, texttemplate="%{text:,}", textfont={"size": 18},
            colorscale=[[0, "#F1F5F9"], [1, PRIMARY]], showscale=False,
        ))
        fig_cm.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)
    with c2:
        st.markdown(f"""
- **{int(metrics['true_positive']):,} fraud cases caught** correctly
- **{int(metrics['false_negative']):,} fraud cases missed** — the model isn't perfect, and neither is any real deployed fraud system
- **{int(metrics['false_positive']):,} clean listings incorrectly flagged** — the real-world cost of a false alarm (a legitimate seller gets a review request)
- **{int(metrics['true_negative']):,} clean listings correctly passed through**

This precision/recall trade-off is a genuine design choice, not an afterthought: a marketplace
tunes its threshold based on whether false alarms (annoying real sellers) or missed fraud
(letting scams through) cost more.
        """)

    section_title("🚩", "Fraud type breakdown")
    type_counts = listings[listings["is_fraud"] == 1]["fraud_type"].value_counts().reset_index()
    type_counts.columns = ["fraud_type", "count"]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(donut_chart(type_counts["fraud_type"].tolist(), type_counts["count"].tolist(),
                                      "Known fraud pattern distribution", colors=[DANGER, ACCENT, PRIMARY, PRIMARY_LIGHT, "#93C5FD"]),
                          use_container_width=True)
    with c2:
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
        fig2 = px.bar(imp_df.sort_values("importance"), x="importance", y="label", orientation="h",
                       title="What the model weighs most", color_discrete_sequence=[PRIMARY])
        fig2.update_layout(yaxis_title="", xaxis_title="Importance", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Note that legitimate listings also show some description reuse (builders' project "
        "templates get reused across sibling units by different agents) and broker phone reuse "
        "(popular agents genuinely list many properties) — the model has to weigh *how much* "
        "reuse is unusual, in combination with other signals, rather than treating any reuse as "
        "an automatic red flag. That overlap is what keeps recall below 100%."
    )

    top_fraud_type = type_counts.iloc[0]
    rent_fraud_rate = listings[listings["listing_type"] == "rent"]["is_fraud"].mean()
    sale_fraud_rate = listings[listings["listing_type"] == "sale"]["is_fraud"].mean()
    worse_channel = "Rental" if rent_fraud_rate > sale_fraud_rate else "Sale"
    insight_grid([
        ("Most common fraud type", f"<b>{top_fraud_type['fraud_type']}</b> accounts for <b>{top_fraud_type['count']}</b> flagged listings — the single largest pattern.", "danger"),
        ("Sale vs. rental fraud rate", f"<b>{worse_channel}</b> listings show the higher fraud rate — {rent_fraud_rate:.1%} rental vs {sale_fraud_rate:.1%} sale.", "warning"),
    ])

with tab2:
    section_title("🔎", "Explore flagged listings")
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
        }).bar(subset=["fraud_probability"], color=ACCENT),
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
            correct = (row["fraud_probability"] > 0.5) == bool(row["is_fraud"])
            if correct:
                st.success("✅ Model classification matches ground truth")
            else:
                st.warning("⚠️ Model classification disagrees with ground truth — a real miss/false-alarm case")

with tab3:
    section_title("🔗", "Near-duplicate detection (TF-IDF + structural match)")
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
