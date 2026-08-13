import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(__file__))
from utils.styling import inject_css, page_header, section_title, PRIMARY, ACCENT
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
    return props, listings, locality, graph

props, listings, locality, graph = load_data()

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("Properties in graph", f"{len(graph):,}")
with col2:
    st.metric("Cities covered", f"{props['city'].nunique()}")
with col3:
    st.metric("Micro-markets", f"{props['locality'].nunique()}")
with col4:
    fraud_rate = listings["is_fraud"].mean()
    st.metric("Listing fraud rate", f"{fraud_rate:.1%}")
with col5:
    st.metric("Avg PRISM Score", f"{graph['prism_score'].mean():.0f}/900")
with col6:
    flagged = (graph["prism_band"] == "Needs Review").sum()
    st.metric("Flagged for review", f"{flagged}")

section_title("🧭", "How PRISM is structured")
st.markdown("""
Inputs feed a **unified property graph** (geo-indexed by pincode/micro-market), which each
ML module scores independently. Those scores roll up into one **Unified PRISM Score**
(300–900, bureau-style) — surfaced differently depending on who's asking.
""")

m1, m2 = st.columns(2)
with m1:
    st.markdown("""
    **🏠 Buyer App** — Is this specific property fairly priced, trustworthy, and on schedule —
    for sale or for rent?

    **📊 Investor Dashboard** — Separate views for rental-income investing vs.
    purchase/appreciation investing.
    """)
with m2:
    st.markdown("""
    **⚙️ Module deep-dives** (sidebar, below) — the price/rent, yield, construction, and
    fraud models underneath, each explorable on its own.
    """)

st.divider()

section_title("🌐", "Market at a glance")
g1, g2, g3 = st.columns(3)
with g1:
    st.plotly_chart(gauge_chart(graph["prism_score"].mean(), "Avg PRISM Score (market-wide)"), use_container_width=True)
with g2:
    band_counts = graph["prism_band"].value_counts()
    st.plotly_chart(
        donut_chart(band_counts.index.tolist(), band_counts.values.tolist(), "Score band mix",
                     colors=["#15803D", PRIMARY, ACCENT, "#B91C1C"]),
        use_container_width=True,
    )
with g3:
    type_counts = props["property_type"].value_counts()
    st.plotly_chart(
        donut_chart(type_counts.index.tolist(), type_counts.values.tolist(), "Property type mix"),
        use_container_width=True,
    )

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
               title="City coverage — bubble size = property count, color = avg PRISM Score"),
    use_container_width=True,
)
st.caption("18 cities, 68 micro-markets, Tier 1 to Tier 3 — bubble color shows where the average listing trust/quality is highest.")

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

st.markdown("#### City tier → City → Locality, sized by property count")
hierarchy_df = props.groupby(["city_tier", "city", "locality"]).size().reset_index(name="count")
st.plotly_chart(
    treemap_chart(hierarchy_df, ["city_tier", "city", "locality"], "count",
                   title="Where PRISM's inventory lives", color_col="count"),
    use_container_width=True,
)

section_title("📈", "Micro-market price landscape")
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
fig.update_layout(xaxis_tickangle=-40, height=450, plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

st.caption("Use the sidebar to navigate between PRISM's surfaces and modules →")
