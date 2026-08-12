import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(__file__))
from utils.styling import inject_css, page_header, PRIMARY, ACCENT

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

st.markdown("### How PRISM is structured")
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

st.markdown("### Coverage across city tiers")
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

st.markdown("### Micro-market price landscape")
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
    color_discrete_sequence=[PRIMARY, "#3B82F6", "#93C5FD", ACCENT],
)
fig.update_layout(xaxis_tickangle=-40, height=450)
st.plotly_chart(fig, use_container_width=True)

st.caption("Use the sidebar to navigate between PRISM's surfaces and modules →")
