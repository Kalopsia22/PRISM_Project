import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, disclosure, PRIMARY, ACCENT
from models.yield_recommender import RISK_PROFILES

st.set_page_config(page_title="PRISM — Investor Dashboard", page_icon="📊", layout="wide")
inject_css()
page_header("📊 Investor Dashboard", "Portfolio-style view combining yield, appreciation, trust, and price fairness")

BASE = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_graph():
    return pd.read_csv(os.path.join(BASE, "data", "unified_property_graph.csv"))

graph = load_graph()

st.markdown("#### Set your investment criteria")
c1, c2, c3 = st.columns(3)
with c1:
    risk_profile = st.selectbox("Risk profile", list(RISK_PROFILES.keys()), format_func=lambda x: x.replace("_", " ").title())
with c2:
    cities = st.multiselect("Cities", sorted(graph["city"].unique()), default=list(graph["city"].unique()))
with c3:
    min_score = st.slider("Minimum PRISM Score", 300, 900, 650, 10)

weights = RISK_PROFILES[risk_profile]

filtered = graph[graph["city"].isin(cities) & (graph["prism_score"] >= min_score)].copy()

# blend the investor's risk-profile weighting with the unified score's investment_score component
filtered["investor_fit_score"] = (
    filtered["investment_score"] * (weights["yield_weight"] + weights["appreciation_weight"])
    + filtered["stability_norm"].fillna(0.5) * weights["stability_weight"]
    + filtered["trust_score"] * 0.15
)

st.divider()
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Properties matching criteria", f"{len(filtered):,}")
with c2:
    st.metric("Avg PRISM Score", f"{filtered['prism_score'].mean():.0f}" if len(filtered) else "—")
with c3:
    st.metric("Avg rental yield", f"{filtered['avg_rental_yield_pct'].mean():.2f}%" if len(filtered) else "—")
with c4:
    st.metric("Avg appreciation", f"{filtered['avg_yoy_appreciation_pct'].mean():.1f}%/yr" if len(filtered) else "—")

st.markdown("#### Top recommended properties for your profile")
top = filtered.sort_values("investor_fit_score", ascending=False).head(10)

if len(top) == 0:
    st.info("No properties match these criteria — try lowering the minimum PRISM Score.")
else:
    display_cols = ["property_id", "city", "locality", "bhk", "carpet_area_sqft", "prism_score",
                     "avg_rental_yield_pct", "avg_yoy_appreciation_pct", "investor_fit_score"]
    st.dataframe(
        top[display_cols].rename(columns={
            "avg_rental_yield_pct": "Yield %", "avg_yoy_appreciation_pct": "Appreciation %/yr",
            "prism_score": "PRISM Score", "investor_fit_score": "Fit Score",
        }).style.format({"Yield %": "{:.2f}", "Appreciation %/yr": "{:.1f}", "Fit Score": "{:.2f}"}),
        use_container_width=True,
    )

st.markdown("#### Portfolio view: risk vs. return across matching properties")
if len(filtered) > 0:
    fig = px.scatter(
        filtered, x="prism_score", y="avg_rental_yield_pct", color="tier",
        size="avg_yoy_appreciation_pct", hover_data=["locality", "city", "property_id"],
        title="PRISM Score (trust/quality) vs. rental yield — bubble size = appreciation",
        color_discrete_sequence=[PRIMARY, "#3B82F6", "#93C5FD", ACCENT],
    )
    fig.update_layout(xaxis_title="PRISM Score (trust + price fairness + delivery + investment)", yaxis_title="Rental yield (%)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Locality-level rollup")
locality_rollup = filtered.groupby(["city", "locality", "tier"]).agg(
    n_properties=("property_id", "count"),
    avg_prism_score=("prism_score", "mean"),
    avg_yield=("avg_rental_yield_pct", "mean"),
    avg_appreciation=("avg_yoy_appreciation_pct", "mean"),
).reset_index().sort_values("avg_prism_score", ascending=False)
st.dataframe(locality_rollup.style.format({"avg_prism_score": "{:.0f}", "avg_yield": "{:.2f}", "avg_appreciation": "{:.1f}"}), use_container_width=True)

disclosure(
    "The Investor Dashboard blends the Rental Yield Recommender's risk-profile weighting with "
    "the unified PRISM Score's trust and price-fairness signals — so a high-yield locality full "
    "of flagged listings won't outrank a slightly-lower-yield locality with cleaner inventory."
)
