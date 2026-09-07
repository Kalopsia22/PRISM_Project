import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(__file__))
from utils.styling import inject_css, page_header, section_title, insight_grid, PRIMARY, ACCENT
from utils.charts import gauge_chart, donut_chart, treemap_chart, map_chart, PRIMARY_LIGHT
from utils.geo import add_city_coords

st.set_page_config(page_title="PRISM — Property Risk Intelligence", page_icon="🏙️", layout="wide")
inject_css()

page_header(
    "PRISM",
    "Property Risk, Intelligence, Score & Monitoring — a decision-support platform for "
    "buyers and investors across Indian residential real estate, Tier 1 to Tier 3"
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_data():
    props = pd.read_csv(os.path.join(DATA_DIR, "properties.csv"))
    listings = pd.read_csv(os.path.join(DATA_DIR, "listings.csv"))
    locality = pd.read_csv(os.path.join(DATA_DIR, "locality_profile.csv"))
    graph = pd.read_csv(os.path.join(DATA_DIR, "unified_property_graph.csv"))
    trend = pd.read_csv(os.path.join(DATA_DIR, "appreciation_trend.csv"))
    return props, listings, locality, graph, trend

props, listings, locality, graph, trend = load_data()
n_cities = props["city"].nunique()
n_localities = props["locality"].nunique()

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("Properties in graph", f"{len(graph):,}")
with col2:
    st.metric("Cities covered", f"{n_cities}")
with col3:
    st.metric("Micro-markets", f"{n_localities}")
with col4:
    fraud_rate = listings["is_fraud"].mean()
    st.metric("Listing fraud rate", f"{fraud_rate:.1%}")
with col5:
    st.metric("Avg PRISM Score", f"{graph['prism_score'].mean():.0f}/900")
with col6:
    flagged = (graph["prism_band"] == "Needs Review").sum()
    st.metric("Flagged for review", f"{flagged}")

# ---- Computed market insights (every value below is derived live, not hardcoded) ----
section_title("💡", "Today's market insights")

city_scores = graph.groupby("city")["prism_score"].mean().sort_values(ascending=False)
best_city, best_city_score = city_scores.index[0], city_scores.iloc[0]

locality_fraud = listings.groupby("locality")["is_fraud"].mean().sort_values(ascending=False)
fraud_hotspot, fraud_hotspot_rate = locality_fraud.index[0], locality_fraud.iloc[0]

trend_5yr = trend[trend["year_offset"] == 5].sort_values("cumulative_index", ascending=False)
top_appreciation = trend_5yr.iloc[0]
top_appreciation_pct = (top_appreciation["cumulative_index"] - 1) * 100

yield_leader = locality.sort_values("avg_rental_yield_pct", ascending=False).iloc[0]

city_needs_review = graph.groupby("city").apply(
    lambda g: (g["prism_band"] == "Needs Review").mean(), include_groups=False
).sort_values(ascending=False)
riskiest_city, riskiest_pct = city_needs_review.index[0], city_needs_review.iloc[0]

real_val_path = os.path.join(DATA_DIR, "real_validation_results.csv")
real_val_insight = None
if os.path.exists(real_val_path):
    real_val = pd.read_csv(real_val_path)
    real_mape = real_val["abs_pct_error"].mean()
    real_val_insight = ("Validated on real data", f"Price model tested against <b>{len(real_val):,}</b> real scraped listings (7 cities) — <b>{real_mape:.0%}</b> overall MAPE out-of-sample.", "primary")

insights = [
    ("Highest-trust market", f"<b>{best_city}</b> leads with an average PRISM Score of <b>{best_city_score:.0f}</b>/900 across its listings.", "primary"),
    ("5-year appreciation leader", f"<b>{top_appreciation['locality']}, {top_appreciation['city']}</b> has compounded <b>{top_appreciation_pct:.0f}%</b> over 5 years — the fastest in the dataset.", "success"),
    ("Best rental yield", f"<b>{yield_leader['locality']}, {yield_leader['city']}</b> offers <b>{yield_leader['avg_rental_yield_pct']:.2f}%</b> average rental yield.", "success"),
    ("Fraud hotspot to watch", f"<b>{fraud_hotspot}</b> shows the highest listing-fraud rate at <b>{fraud_hotspot_rate:.0%}</b> — worth extra scrutiny.", "danger"),
    ("Highest compliance-review rate", f"<b>{riskiest_city}</b> has the largest share of PRISM-flagged properties at <b>{riskiest_pct:.0%}</b>.", "warning"),
]
if real_val_insight:
    insights.append(real_val_insight)
insight_grid(insights)

section_title("🧭", "How PRISM is structured")
st.markdown("""
Inputs feed a **unified property graph** (geo-indexed by pincode/micro-market), which each
ML module scores independently. Those scores roll up into one **Unified PRISM Score**
(300–900, bureau-style) — surfaced differently depending on who's asking.
""")

m1, m2 = st.columns(2)
with m1:
    st.markdown(
"""**🏠 Buyer App** — Is this specific property fairly priced, trustworthy, and on schedule —
for sale or for rent?

**📊 Investor Dashboard** — Separate views for rental-income investing vs.
purchase/appreciation investing."""
    )
with m2:
    st.markdown(
"""**⚙️ Module deep-dives** (sidebar, below) — the price/rent, AML/compliance, and
fraud models underneath, each explorable on its own."""
    )

st.divider()

section_title("🌐", "Market at a glance")
g1, g2, g3 = st.columns(3)
with g1:
    st.plotly_chart(gauge_chart(graph["prism_score"].mean(), "Avg PRISM Score (market-wide)", height=340), use_container_width=True)
with g2:
    band_counts = graph["prism_band"].value_counts()
    st.plotly_chart(
        donut_chart(band_counts.index.tolist(), band_counts.values.tolist(), "Score band mix",
                     colors=["#15803D", PRIMARY, ACCENT, "#B91C1C"], height=420),
        use_container_width=True,
    )
with g3:
    type_counts = props["property_type"].value_counts()
    st.plotly_chart(
        donut_chart(type_counts.index.tolist(), type_counts.values.tolist(), "Property type mix", height=420),
        use_container_width=True,
    )

section_title("📈", "5-year price appreciation trajectory")
st.caption(
    "Cumulative appreciation index by locality quality tier since year 0 — the compounding gap "
    "between affordable and premium markets is exactly why 'highest price today' and 'best "
    "investment' are different questions."
)
trend_with_tier = trend.merge(props[["city", "locality", "tier"]].drop_duplicates(), on=["city", "locality"], how="left")
tier_trend = trend_with_tier.groupby(["tier", "year_offset"])["cumulative_index"].mean().reset_index()
tier_trend["cumulative_pct"] = (tier_trend["cumulative_index"] - 1) * 100
fig_trend = px.line(
    tier_trend, x="year_offset", y="cumulative_pct", color="tier", markers=True,
    title="Average cumulative appreciation by locality tier",
    color_discrete_sequence=[ACCENT, PRIMARY_LIGHT, PRIMARY, "#166534"],
)
fig_trend.update_layout(xaxis_title="Years from now", yaxis_title="Cumulative appreciation (%)", plot_bgcolor="rgba(0,0,0,0)", height=480)
st.plotly_chart(fig_trend, use_container_width=True)

section_title("🗺️", "PRISM's coverage across India")
city_agg = props.groupby(["city", "city_tier"]).agg(
    n_properties=("property_id", "count"),
).reset_index()
city_score = graph.groupby("city")["prism_score"].mean().reset_index().rename(columns={"prism_score": "avg_prism_score"})
city_agg = city_agg.merge(city_score, on="city", how="left")
city_agg = add_city_coords(city_agg)
st.plotly_chart(
    map_chart(city_agg, color_col="avg_prism_score", size_col="n_properties", hover_name="city",
               hover_data={"city_tier": True, "n_properties": True, "avg_prism_score": ":.0f"},
               title="City coverage — bubble size = property count, color = avg PRISM Score", height=640),
    use_container_width=True,
)
st.caption(f"{n_cities} cities, {n_localities} micro-markets, Tier 1 to Tier 3 — bubble color shows where the average listing trust/quality is highest.")

section_title("🏙️", "Coverage across city tiers")
tier_summary = props.groupby("city_tier")["city"].nunique().reset_index()
tier_summary.columns = ["City Tier", "Cities"]
c1, c2 = st.columns([1, 2])
with c1:
    st.dataframe(tier_summary, use_container_width=True, hide_index=True)
with c2:
    st.caption(
        "Tier 1: Mumbai, Bangalore, Delhi NCR, Chennai, Hyderabad, Pune, Kolkata, Ahmedabad · "
        "Tier 2: Jaipur, Lucknow, Chandigarh, Indore, Kochi, Surat · "
        "Tier 3: Bhubaneswar, Raipur, Ranchi, Dehradun"
    )

section_title("🌳", "City tier → City → Locality, sized by property count")
hierarchy_df = props.groupby(["city_tier", "city", "locality"]).size().reset_index(name="count")
st.plotly_chart(
    treemap_chart(hierarchy_df, ["city_tier", "city", "locality"], "count",
                   title="Where PRISM's inventory lives", color_col="count", height=600),
    use_container_width=True,
)

section_title("🏆", "PRISM Score by city — full ranking")
city_rank = graph.groupby("city").agg(
    avg_score=("prism_score", "mean"), n=("property_id", "count"),
).reset_index().sort_values("avg_score", ascending=True)
city_rank = city_rank.merge(props[["city", "city_tier"]].drop_duplicates(), on="city", how="left")
fig_rank = px.bar(
    city_rank, x="avg_score", y="city", color="city_tier", orientation="h",
    title="Average PRISM Score across all 18 cities",
    color_discrete_sequence=[PRIMARY, PRIMARY_LIGHT, ACCENT],
    text="avg_score",
)
fig_rank.update_traces(texttemplate="%{text:.0f}", textposition="outside")
fig_rank.update_layout(xaxis_title="Avg PRISM Score", yaxis_title="", height=700, plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_rank, use_container_width=True)

section_title("📊", "Micro-market price landscape")
c1, c2 = st.columns(2)
with c1:
    tier_filter = st.selectbox("City tier", options=["All"] + sorted(props["city_tier"].unique().tolist()))
tier_df = props if tier_filter == "All" else props[props["city_tier"] == tier_filter]
with c2:
    city_filter = st.selectbox("City", options=["All"] + sorted(tier_df["city"].unique().tolist()))
plot_df = tier_df if city_filter == "All" else tier_df[tier_df["city"] == city_filter]

fig = px.box(
    plot_df, x="locality", y="price_per_sqft", color="tier",
    title="Price per sqft distribution by micro-market",
    color_discrete_sequence=[PRIMARY, PRIMARY_LIGHT, "#93C5FD", ACCENT],
)
fig.update_layout(xaxis_tickangle=-40, height=550, plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

if city_filter != "All":
    city_avg = plot_df["price_per_sqft"].mean()
    overall_avg = props["price_per_sqft"].mean()
    delta_pct = (city_avg / overall_avg - 1) * 100
    direction = "above" if delta_pct > 0 else "below"
    st.caption(f"💡 **{city_filter}** averages ₹{city_avg:,.0f}/sqft — {abs(delta_pct):.0f}% {direction} the all-city average of ₹{overall_avg:,.0f}/sqft.")

st.caption("Use the sidebar to navigate between PRISM's surfaces and modules →")
