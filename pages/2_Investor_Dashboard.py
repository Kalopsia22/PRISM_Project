import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, section_title, insight_grid, PRIMARY, ACCENT
from utils.charts import gauge_chart, treemap_chart, multi_radar_chart, map_chart, PRIMARY_LIGHT
from utils.geo import add_jittered_coords
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
    st.caption("For investors buying property specifically to rent it out — weighted toward yield and rent-side trust. Yield figures calibrated to the Magicbricks Rental Index (JFM'26).")

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

    g1, g2, g3 = st.columns([1, 1, 2])
    with g1:
        st.plotly_chart(gauge_chart(filtered_r["rent_prism_score"].mean() if len(filtered_r) else 300,
                                      "Avg rental PRISM Score", height=210), use_container_width=True)
    with g2:
        st.metric("Rentable properties matching", f"{len(filtered_r):,}")
        st.metric("Avg rental yield", f"{filtered_r['avg_rental_yield_pct'].mean():.2f}%" if len(filtered_r) else "—")
    with g3:
        st.metric("Avg monthly rent", f"₹{filtered_r['rent_ask_value'].mean():,.0f}" if len(filtered_r) else "—")
        st.caption(f"**{risk_profile_r.replace('_',' ').title()}** weighting → Yield {weights_r['yield_weight']:.0%} · Appreciation {weights_r['appreciation_weight']:.0%} · Stability {weights_r['stability_weight']:.0%}")

    section_title("🏆", "Top rental-income properties for your profile")
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
            }).style.format({"Monthly Rent": "₹{:,.0f}", "Yield %": "{:.2f}", "Fit Score": "{:.2f}"})
              .bar(subset=["Fit Score"], color="#93C5FD"),
            use_container_width=True,
        )

        top_pick = top_r.iloc[0]
        city_concentration = top_r["city"].value_counts()
        yield_spread = top_r["avg_rental_yield_pct"].max() - top_r["avg_rental_yield_pct"].min()
        insight_grid([
            ("Top pick", f"<b>{top_pick['locality']}, {top_pick['city']}</b> — PRISM Score {top_pick['rent_prism_score']:.0f}, yield {top_pick['avg_rental_yield_pct']:.2f}%.", "primary"),
            ("Geographic spread", f"Top 10 span <b>{len(city_concentration)}</b> cities — {city_concentration.index[0]} contributes the most ({city_concentration.iloc[0]} of 10).", "success" if len(city_concentration) >= 4 else "warning"),
            ("Yield spread in shortlist", f"Ranges from <b>{top_r['avg_rental_yield_pct'].min():.2f}%</b> to <b>{top_r['avg_rental_yield_pct'].max():.2f}%</b> — a {yield_spread:.1f}pp gap even within your top matches.", "primary"),
        ])

        section_title("🧱", "What's driving each top pick's score")
        comp_r = top_r[["locality", "rent_price_fairness_score", "rent_trust_score", "compliance_score", "investment_score"]].copy()
        comp_r["label"] = comp_r["locality"] + " (#" + top_r["property_id"].astype(str) + ")"
        comp_r_melt = comp_r.melt(id_vars="label", value_vars=["rent_price_fairness_score", "rent_trust_score", "compliance_score", "investment_score"],
                                    var_name="Dimension", value_name="Score")
        comp_r_melt["Dimension"] = comp_r_melt["Dimension"].map({
            "rent_price_fairness_score": "Price Fairness", "rent_trust_score": "Trust",
            "compliance_score": "Compliance", "investment_score": "Investment Value",
        })
        fig_comp_r = px.bar(
            comp_r_melt, x="label", y="Score", color="Dimension", barmode="stack",
            title="Score composition for the top 10 rental picks",
            color_discrete_sequence=[PRIMARY, PRIMARY_LIGHT, ACCENT, "#166534"],
        )
        fig_comp_r.update_layout(height=520, xaxis_title="", yaxis_title="Stacked score", xaxis_tickangle=-30, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_comp_r, use_container_width=True)

    section_title("📊", "Yield vs. rental trust across matching properties")
    if len(filtered_r) > 0:
        fig_r = px.scatter(
            filtered_r, x="rent_prism_score", y="avg_rental_yield_pct", color="tier",
            size="avg_yoy_appreciation_pct", hover_data=["locality", "city", "property_type"],
            title="Rental PRISM Score vs. yield — bubble size = appreciation",
            color_discrete_sequence=[PRIMARY, PRIMARY_LIGHT, "#93C5FD", ACCENT],
            render_mode="svg",
        )
        fig_r.update_layout(xaxis_title="Rental PRISM Score", yaxis_title="Rental yield (%)", plot_bgcolor="rgba(0,0,0,0)", height=480)
        st.plotly_chart(fig_r, use_container_width=True)

    section_title("🗺️", "Where the top-10 picks are on the map")
    if len(top_r) > 0:
        top_r_geo = add_jittered_coords(top_r)
        st.plotly_chart(
            map_chart(top_r_geo, color_col="rent_prism_score", size_col="avg_rental_yield_pct",
                       hover_name="locality", hover_data={"city": True, "rent_ask_value": ":.0f"},
                       title="Top rental picks — color = PRISM Score, size = yield", height=560, zoom=3.6),
            use_container_width=True,
        )

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
        + filtered_p["compliance_score"] * 0.10
    )

    g1, g2, g3 = st.columns([1, 1, 2])
    with g1:
        st.plotly_chart(gauge_chart(filtered_p["prism_score"].mean() if len(filtered_p) else 300,
                                      "Avg PRISM Score", height=210), use_container_width=True)
    with g2:
        st.metric("Properties matching", f"{len(filtered_p):,}")
        st.metric("Avg appreciation", f"{filtered_p['avg_yoy_appreciation_pct'].mean():.1f}%/yr" if len(filtered_p) else "—")
    with g3:
        flagged_pct = (filtered_p["aml_flag"] == 1).mean() if len(filtered_p) else 0
        st.metric("AML-flagged transactions", f"{flagged_pct:.0%}")
        st.caption(f"**{risk_profile_p.replace('_',' ').title()}** weighting → Yield {weights_p['yield_weight']:.0%} · Appreciation {weights_p['appreciation_weight']:.0%} · Stability {weights_p['stability_weight']:.0%}")

    section_title("🏆", "Top purchase/appreciation properties for your profile")
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
            }).style.format({"Asking Price": "₹{:,.0f}", "Appreciation %/yr": "{:.1f}", "Fit Score": "{:.2f}"})
              .bar(subset=["Fit Score"], color="#93C5FD"),
            use_container_width=True,
        )

        top_pick_p = top_p.iloc[0]
        market_avg_score = graph["prism_score"].mean()
        shortlist_avg_score = top_p["prism_score"].mean()
        score_edge = shortlist_avg_score - market_avg_score
        fastest = top_p.sort_values("avg_yoy_appreciation_pct", ascending=False).iloc[0]
        insight_grid([
            ("Top pick", f"<b>{top_pick_p['locality']}, {top_pick_p['city']}</b> — PRISM Score {top_pick_p['prism_score']:.0f}, appreciating {top_pick_p['avg_yoy_appreciation_pct']:.1f}%/yr.", "primary"),
            ("Shortlist quality edge", f"Your top 10 average <b>{shortlist_avg_score:.0f}</b> vs the market average of <b>{market_avg_score:.0f}</b> — a {score_edge:+.0f} point edge from filtering.", "success" if score_edge > 0 else "warning"),
            ("Fastest appreciator in shortlist", f"<b>{fastest['locality']}, {fastest['city']}</b> at <b>{fastest['avg_yoy_appreciation_pct']:.1f}%/yr</b> — check its price fairness score before assuming it's the best pick.", "primary"),
        ])

        section_title("🧱", "What's driving each top pick's score")
        comp_p = top_p[["locality", "price_fairness_score", "trust_score", "compliance_score", "investment_score"]].copy()
        comp_p["label"] = comp_p["locality"] + " (#" + top_p["property_id"].astype(str) + ")"
        comp_p_melt = comp_p.melt(id_vars="label", value_vars=["price_fairness_score", "trust_score", "compliance_score", "investment_score"],
                                    var_name="Dimension", value_name="Score")
        comp_p_melt["Dimension"] = comp_p_melt["Dimension"].map({
            "price_fairness_score": "Price Fairness", "trust_score": "Trust",
            "compliance_score": "Compliance", "investment_score": "Investment Value",
        })
        fig_comp_p = px.bar(
            comp_p_melt, x="label", y="Score", color="Dimension", barmode="stack",
            title="Score composition for the top 10 purchase picks",
            color_discrete_sequence=[PRIMARY, PRIMARY_LIGHT, ACCENT, "#166534"],
        )
        fig_comp_p.update_layout(height=520, xaxis_title="", yaxis_title="Stacked score", xaxis_tickangle=-30, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_comp_p, use_container_width=True)

    section_title("📊", "PRISM Score vs. appreciation across matching properties")
    if len(filtered_p) > 0:
        fig_p = px.scatter(
            filtered_p, x="prism_score", y="avg_yoy_appreciation_pct", color="tier",
            size="avg_rental_yield_pct", hover_data=["locality", "city", "property_type"],
            title="PRISM Score vs. appreciation — bubble size = yield",
            color_discrete_sequence=[PRIMARY, PRIMARY_LIGHT, "#93C5FD", ACCENT],
            render_mode="svg",
        )
        fig_p.update_layout(xaxis_title="PRISM Score", yaxis_title="YoY appreciation (%)", plot_bgcolor="rgba(0,0,0,0)", height=480)
        st.plotly_chart(fig_p, use_container_width=True)

    section_title("🗺️", "Locality-level rollup, sized by property count")
    if len(filtered_p) > 0:
        rollup = filtered_p.groupby(["city", "locality", "tier"]).agg(
            n_properties=("property_id", "count"),
            avg_prism_score=("prism_score", "mean"),
            avg_appreciation=("avg_yoy_appreciation_pct", "mean"),
        ).reset_index().sort_values("avg_prism_score", ascending=False)

        top_p_geo = add_jittered_coords(top_p) if len(top_p) > 0 else None
        c1, c2 = st.columns([3, 2])
        with c1:
            if top_p_geo is not None:
                st.plotly_chart(
                    map_chart(top_p_geo, color_col="prism_score", size_col="avg_yoy_appreciation_pct",
                               hover_name="locality", hover_data={"city": True, "sale_ask_value": ":.0f"},
                               title="Top-10 picks — color = PRISM Score, size = appreciation", height=520, zoom=3.6),
                    use_container_width=True,
                )
        with c2:
            st.dataframe(rollup.style.format({"avg_prism_score": "{:.0f}", "avg_appreciation": "{:.1f}"})
                          .bar(subset=["avg_prism_score"], color="#86EFAC"),
                         use_container_width=True, height=460)

    section_title("🎯", "Comparing risk profiles side by side")
    st.caption("How each investor persona weighs yield, appreciation, and stability — same market, different lens.")
    radar_cats = ["Yield Weight", "Appreciation Weight", "Stability Weight"]
    radar_series = {
        rp.replace("_", " ").title(): [w["yield_weight"], w["appreciation_weight"], w["stability_weight"]]
        for rp, w in RISK_PROFILES.items()
    }
    st.plotly_chart(multi_radar_chart(radar_cats, radar_series, height=480), use_container_width=True)
