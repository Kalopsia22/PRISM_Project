import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, PRIMARY, ACCENT
from models.yield_recommender import RISK_PROFILES

st.set_page_config(page_title="PRISM — Investor Dashboard", page_icon="📊", layout="wide")
inject_css()
page_header("📊 Investor Dashboard", "Two lenses on the same market — buy-to-rent income vs. buy-to-hold appreciation")

BASE = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_graph():
    return pd.read_csv(os.path.join(BASE, "data", "unified_property_graph.csv"))

graph = load_graph()

tab_rent, tab_purchase = st.tabs(["🏠 Rental Income Dashboard", "🏗️ Purchase / Appreciation Dashboard"])

# =============================================================
# TAB 1 — Rental income focused
# =============================================================
with tab_rent:
    st.caption("For investors buying property specifically to rent it out — weighted toward yield and rent-side trust.")

    c1, c2, c3 = st.columns(3)
    with c1:
        risk_profile_r = st.selectbox("Risk profile", list(RISK_PROFILES.keys()), format_func=lambda x: x.replace("_", " ").title(), key="rp_rent")
    with c2:
        cities_r = st.multiselect("Cities", sorted(graph["city"].unique()), default=list(graph["city"].unique()), key="cities_rent")
    with c3:
        min_score_r = st.slider("Minimum rental PRISM Score", 300, 900, 600, 10, key="score_rent")

    weights_r = RISK_PROFILES[risk_profile_r]
    rentable = graph[graph["rent_ask_value"].notna()].copy()
    filtered_r = rentable[rentable["city"].isin(cities_r) & (rentable["rent_prism_score"] >= min_score_r)].copy()

    filtered_r["investor_fit_score"] = (
        filtered_r["investment_score"] * (weights_r["yield_weight"] + weights_r["appreciation_weight"])
        + filtered_r["stability_norm"].fillna(0.5) * weights_r["stability_weight"]
        + filtered_r["rent_trust_score"].fillna(0.5) * 0.15
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rentable properties matching", f"{len(filtered_r):,}")
    with c2:
        st.metric("Avg rental PRISM Score", f"{filtered_r['rent_prism_score'].mean():.0f}" if len(filtered_r) else "—")
    with c3:
        st.metric("Avg rental yield", f"{filtered_r['avg_rental_yield_pct'].mean():.2f}%" if len(filtered_r) else "—")
    with c4:
        st.metric("Avg monthly rent", f"₹{filtered_r['rent_ask_value'].mean():,.0f}" if len(filtered_r) else "—")

    st.markdown("#### Top rental-income properties for your profile")
    top_r = filtered_r.sort_values("investor_fit_score", ascending=False).head(10)
    if len(top_r) == 0:
        st.info("No properties match — try lowering the minimum score.")
    else:
        cols_r = ["property_id", "city", "locality", "property_type", "bhk", "rent_ask_value",
                   "rent_prism_score", "avg_rental_yield_pct", "investor_fit_score"]
        st.dataframe(
            top_r[cols_r].rename(columns={
                "rent_ask_value": "Monthly Rent", "rent_prism_score": "PRISM Score",
                "avg_rental_yield_pct": "Yield %", "investor_fit_score": "Fit Score",
            }).style.format({"Monthly Rent": "₹{:,.0f}", "Yield %": "{:.2f}", "Fit Score": "{:.2f}"}),
            use_container_width=True,
        )

    st.markdown("#### Yield vs. rental trust across matching properties")
    if len(filtered_r) > 0:
        fig_r = px.scatter(
            filtered_r, x="rent_prism_score", y="avg_rental_yield_pct", color="tier",
            size="avg_yoy_appreciation_pct", hover_data=["locality", "city", "property_type"],
            title="Rental PRISM Score vs. yield — bubble size = appreciation",
            color_discrete_sequence=[PRIMARY, "#3B82F6", "#93C5FD", ACCENT],
            render_mode="svg",
        )
        fig_r.update_layout(xaxis_title="Rental PRISM Score", yaxis_title="Rental yield (%)")
        st.plotly_chart(fig_r, use_container_width=True)

# =============================================================
# TAB 2 — Purchase / appreciation focused
# =============================================================
with tab_purchase:
    st.caption("For investors buying to hold and resell — weighted toward appreciation, price fairness, and delivery risk.")

    c1, c2, c3 = st.columns(3)
    with c1:
        risk_profile_p = st.selectbox("Risk profile", list(RISK_PROFILES.keys()), format_func=lambda x: x.replace("_", " ").title(), key="rp_purchase")
    with c2:
        cities_p = st.multiselect("Cities", sorted(graph["city"].unique()), default=list(graph["city"].unique()), key="cities_purchase")
    with c3:
        min_score_p = st.slider("Minimum PRISM Score", 300, 900, 650, 10, key="score_purchase")

    weights_p = RISK_PROFILES[risk_profile_p]
    filtered_p = graph[graph["city"].isin(cities_p) & (graph["prism_score"] >= min_score_p)].copy()

    filtered_p["investor_fit_score"] = (
        filtered_p["investment_score"] * (weights_p["yield_weight"]*0.3 + weights_p["appreciation_weight"]*1.2)
        + filtered_p["stability_norm"].fillna(0.5) * weights_p["stability_weight"]
        + filtered_p["price_fairness_score"] * 0.20
        + filtered_p["delivery_score"] * 0.10
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Properties matching", f"{len(filtered_p):,}")
    with c2:
        st.metric("Avg PRISM Score", f"{filtered_p['prism_score'].mean():.0f}" if len(filtered_p) else "—")
    with c3:
        st.metric("Avg appreciation", f"{filtered_p['avg_yoy_appreciation_pct'].mean():.1f}%/yr" if len(filtered_p) else "—")
    with c4:
        under_construction_pct = filtered_p["under_construction"].mean() if len(filtered_p) else 0
        st.metric("Under construction", f"{under_construction_pct:.0%}")

    st.markdown("#### Top purchase/appreciation properties for your profile")
    top_p = filtered_p.sort_values("investor_fit_score", ascending=False).head(10)
    if len(top_p) == 0:
        st.info("No properties match — try lowering the minimum PRISM Score.")
    else:
        cols_p = ["property_id", "city", "locality", "property_type", "bhk", "sale_ask_value",
                   "prism_score", "avg_yoy_appreciation_pct", "investor_fit_score"]
        st.dataframe(
            top_p[cols_p].rename(columns={
                "sale_ask_value": "Asking Price", "prism_score": "PRISM Score",
                "avg_yoy_appreciation_pct": "Appreciation %/yr", "investor_fit_score": "Fit Score",
            }).style.format({"Asking Price": "₹{:,.0f}", "Appreciation %/yr": "{:.1f}", "Fit Score": "{:.2f}"}),
            use_container_width=True,
        )

    st.markdown("#### PRISM Score vs. appreciation across matching properties")
    if len(filtered_p) > 0:
        fig_p = px.scatter(
            filtered_p, x="prism_score", y="avg_yoy_appreciation_pct", color="tier",
            size="avg_rental_yield_pct", hover_data=["locality", "city", "property_type"],
            title="PRISM Score vs. appreciation — bubble size = yield",
            color_discrete_sequence=[PRIMARY, "#3B82F6", "#93C5FD", ACCENT],
            render_mode="svg",
        )
        fig_p.update_layout(xaxis_title="PRISM Score", yaxis_title="YoY appreciation (%)")
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("#### Locality-level rollup")
    if len(filtered_p) > 0:
        rollup = filtered_p.groupby(["city", "locality", "tier"]).agg(
            n_properties=("property_id", "count"),
            avg_prism_score=("prism_score", "mean"),
            avg_appreciation=("avg_yoy_appreciation_pct", "mean"),
        ).reset_index().sort_values("avg_prism_score", ascending=False)
        st.dataframe(rollup.style.format({"avg_prism_score": "{:.0f}", "avg_appreciation": "{:.1f}"}), use_container_width=True)
