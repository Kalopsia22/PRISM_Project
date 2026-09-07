import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, section_title, insight_grid, PRIMARY, ACCENT, SUCCESS, DANGER
from utils.charts import donut_chart, PRIMARY_LIGHT
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

section_title("🌡️", "Fraud risk heatmap: city × listing type")
fraud_pivot = listings.pivot_table(index="city", columns="listing_type", values="is_fraud", aggfunc="mean") * 100
fig_fraud_heat = px.imshow(
    fraud_pivot, text_auto=".1f", aspect="auto",
    color_continuous_scale=[[0, "#F0FDF4"], [0.5, ACCENT], [1, DANGER]],
    labels=dict(color="Fraud rate %"),
    title="Fraud rate (%) by city and listing type",
)
fig_fraud_heat.update_layout(height=650, coloraxis_colorbar=dict(title=""))
st.plotly_chart(fig_fraud_heat, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["📊 Overview & Performance", "🔎 Flagged Listings Explorer", "🔗 Duplicate Detection"])

with tab1:
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

    section_title("🏙️", "Fraud rate by city — full ranking")
    city_fraud_full = listings.groupby("city")["is_fraud"].agg(["mean", "sum", "count"]).reset_index()
    city_fraud_full.columns = ["city", "fraud_rate", "flagged_count", "total_listings"]
    city_fraud_full["fraud_rate"] *= 100
    city_fraud_full = city_fraud_full.sort_values("fraud_rate")
    fig_city_fraud = px.bar(
        city_fraud_full, x="fraud_rate", y="city", orientation="h",
        color="fraud_rate", color_continuous_scale=[[0, "#DCFCE7"], [0.5, ACCENT], [1, DANGER]],
        title="Fraud rate (%) across all 18 cities", text="flagged_count",
        hover_data=["total_listings"],
    )
    fig_city_fraud.update_traces(texttemplate="%{text} flagged", textposition="outside")
    fig_city_fraud.update_layout(height=700, xaxis_title="Fraud rate (%)", yaxis_title="",
                                    plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
    st.plotly_chart(fig_city_fraud, use_container_width=True)

    section_title("💰", "Price deviation by fraud type")
    fig_dev_box = px.box(
        listings, x="fraud_type", y="price_deviation_pct", color="fraud_type",
        title="How far each listing's asking price deviates from fair value, by fraud type",
        color_discrete_sequence=[PRIMARY, PRIMARY_LIGHT, "#93C5FD", ACCENT, DANGER, "#7C3AED"],
    )
    fig_dev_box.update_layout(height=550, xaxis_title="", yaxis_title="Price deviation (%)",
                                 plot_bgcolor="rgba(0,0,0,0)", showlegend=False, xaxis_tickangle=-15)
    st.plotly_chart(fig_dev_box, use_container_width=True)

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
