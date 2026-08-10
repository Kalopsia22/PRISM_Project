import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(__file__))
from utils.styling import inject_css, page_header, disclosure, PRIMARY, ACCENT

st.set_page_config(page_title="PRISM — Property Risk Intelligence", page_icon="🏙️", layout="wide")
inject_css()

page_header(
    "PRISM",
    "Property Risk, Intelligence, Score & Monitoring — a decision-support platform for lenders, "
    "buyers, and investors in Indian residential real estate"
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

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Properties in graph", f"{len(graph):,}")
with col2:
    st.metric("Micro-markets covered", f"{props['locality'].nunique()}")
with col3:
    fraud_rate = listings["is_fraud"].mean()
    st.metric("Listing fraud rate (synthetic)", f"{fraud_rate:.1%}")
with col4:
    st.metric("Avg PRISM Score", f"{graph['prism_score'].mean():.0f}/900")
with col5:
    flagged = (graph["prism_band"] == "Needs Review").sum()
    st.metric("Flagged for review", f"{flagged}")

st.markdown("### How PRISM is structured")
st.markdown("""
Four inputs feed a **unified property graph** (geo-indexed by pincode/micro-market), which
four ML modules score independently. Those scores roll up into one **Unified PRISM Score**
(300–900, bureau-style) — surfaced differently depending on who's asking.
""")

m1, m2 = st.columns(2)
with m1:
    st.markdown("""
    **🏠 Buyer App** — Is this specific property fairly priced, trustworthy, and on schedule?

    **📊 Investor Dashboard** — Which localities/properties best fit my risk profile and budget?
    """)
with m2:
    st.markdown("""
    **🏦 Lender API** — Should the next construction-linked disbursement tranche be released?

    **⚙️ Module deep-dives** (sidebar, below) — the price, yield, construction, and fraud
    models underneath, each explorable on its own.
    """)

st.divider()

st.markdown("### Micro-market price landscape")
city_filter = st.selectbox("City", options=["All"] + sorted(props["city"].unique().tolist()))
plot_df = props if city_filter == "All" else props[props["city"] == city_filter]

fig = px.box(
    plot_df, x="locality", y="price_per_sqft", color="tier",
    title="Price per sqft distribution by micro-market",
    color_discrete_sequence=[PRIMARY, "#3B82F6", "#93C5FD", ACCENT],
)
fig.update_layout(xaxis_tickangle=-40, height=450)
st.plotly_chart(fig, use_container_width=True)

disclosure(
    "Property and listing data in this demo is synthetically generated but calibrated to real, "
    "publicly available anchors (Ready Reckoner/circle-rate bands, RERA registration patterns, "
    "stamp duty rates by state). Bulk registered-transaction data is not available via a single "
    "public API in India — it is fragmented across state sub-registrar portals. See README for "
    "full methodology and data-sourcing notes."
)

st.caption("Use the sidebar to navigate between the four PRISM modules →")
