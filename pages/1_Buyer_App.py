import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, section_title, insight_grid, tier_pill, PRIMARY, ACCENT, SUCCESS, DANGER
from utils.charts import gauge_chart, radar_chart, property_icon_svg

st.set_page_config(page_title="PRISM — Buyer App", page_icon="🏠", layout="wide")
inject_css()
page_header("🏠 Buyer App", "One score, four checks — is this property fairly priced, trustworthy, and on-schedule?")

BASE = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_graph():
    return pd.read_csv(os.path.join(BASE, "data", "unified_property_graph.csv"))

graph = load_graph()

st.markdown("#### What are you looking for?")
mode = st.radio("Mode", ["Buy", "Rent"], horizontal=True, label_visibility="collapsed")
is_rent = mode == "Rent"

if is_rent:
    candidates = graph[graph["rent_ask_value"].notna()]
else:
    candidates = graph

icon_cols = st.columns(7)
all_types = ["Apartment", "Villa", "Independent House", "Penthouse", "Studio/1RK", "Row House", "Plot/Land"]
for i, ptype in enumerate(all_types):
    with icon_cols[i]:
        st.markdown(
            f'<div style="text-align:center;"><div style="background:#F1F5F9; border-radius:10px; padding:6px; display:inline-block;">{property_icon_svg(ptype, color=PRIMARY, size=32)}</div>'
            f'<div style="font-size:0.68rem; color:#64748B; margin-top:2px;">{ptype.split("/")[0]}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("#### Find a property")
c1, c2, c3, c4 = st.columns(4)
with c1:
    city_tier = st.selectbox("City tier", ["All"] + sorted(candidates["city_tier"].unique().tolist()))
tier_pool = candidates if city_tier == "All" else candidates[candidates["city_tier"] == city_tier]
with c2:
    city = st.selectbox("City", sorted(tier_pool["city"].unique()))
city_pool = tier_pool[tier_pool["city"] == city]
with c3:
    property_type = st.selectbox("Property type", ["All"] + sorted(city_pool["property_type"].unique().tolist()))
type_pool = city_pool if property_type == "All" else city_pool[city_pool["property_type"] == property_type]
with c4:
    localities = sorted(type_pool["locality"].unique())
    locality = st.selectbox("Locality", localities)

subset = type_pool[type_pool["locality"] == locality]
if len(subset) == 0:
    st.warning("No properties match this combination — try a different filter.")
    st.stop()

prop_id = st.selectbox(
    "Property", subset["property_id"].tolist(),
    format_func=lambda pid: (
        f"#{pid} — {subset[subset['property_id']==pid]['property_type'].iloc[0]}, "
        f"{subset[subset['property_id']==pid]['bhk'].iloc[0]}BHK, "
        f"{subset[subset['property_id']==pid]['carpet_area_sqft'].iloc[0]:.0f} sqft"
    )
)

row = graph[graph["property_id"] == prop_id].iloc[0]

st.divider()

if is_rent:
    score = int(row["rent_prism_score"])
    band = row["rent_prism_band"]
    ask_value = row["rent_ask_value"]
    deviation = row["rent_price_deviation_pct"]
    fraud_p = row.get("rent_fraud_probability", 0)
    fraud_type = row.get("rent_fraud_type", "none")
    price_fairness = row["rent_price_fairness_score"]
    trust = row["rent_trust_score"]
    value_label = "Monthly rent asked"
else:
    score = int(row["prism_score"])
    band = row["prism_band"]
    ask_value = row["sale_ask_value"]
    deviation = row["sale_price_deviation_pct"]
    fraud_p = row.get("fraud_probability", 0)
    fraud_type = row.get("fraud_type", "none")
    price_fairness = row["price_fairness_score"]
    trust = row["trust_score"]
    value_label = "Listed sale price"

band_color = {"Excellent": SUCCESS, "Good": PRIMARY, "Fair": ACCENT, "Needs Review": DANGER}[band]

colScore, colDetails = st.columns([1, 2])
with colScore:
    st.plotly_chart(gauge_chart(score, "PRISM Score", height=220), use_container_width=True)
    band_html = f'<div style="text-align:center; margin-top:-1.2rem;"><span style="font-size:1.15rem; font-weight:700; color:{band_color};">{band}</span></div>'
    st.markdown(band_html, unsafe_allow_html=True)

with colDetails:
    icon_col, text_col = st.columns([1, 6])
    with icon_col:
        st.markdown(
            f'<div style="background:#F1F5F9; border-radius:10px; padding:8px; width:fit-content;">{property_icon_svg(row["property_type"], color=band_color, size=44)}</div>',
            unsafe_allow_html=True,
        )
    with text_col:
        st.markdown(f"{tier_pill(row['city_tier'])} **{row['property_type']} · {row['bhk']}BHK · {row['carpet_area_sqft']:.0f} sqft · {row['locality']}, {row['city']}**", unsafe_allow_html=True)
        st.caption(f"Builder: {row['builder']} · RERA: {'✅ Registered' if row['rera_registered'] else '❌ Not registered'} · Age: {row['age_years']} yrs")
    st.metric(value_label, f"₹{ask_value:,.0f}", f"{deviation:+.1f}% vs. fair value" if pd.notna(deviation) else None)

st.divider()
section_title("🔍", "The four checks behind this score")

col_radar, col_detail = st.columns([1, 1])
with col_radar:
    radar_cats = ["Price Fairness", "Trust", "Compliance", "Investment Value"]
    radar_vals = [price_fairness * 100, trust * 100, row["compliance_score"] * 100, row["investment_score"] * 100]
    st.plotly_chart(radar_chart(radar_cats, radar_vals, color=band_color, height=420), use_container_width=True)

with col_detail:
    f1, f2 = st.columns(2)
    with f1:
        st.metric("💰 Price Fairness", f"{price_fairness*100:.0f}/100")
        st.caption(f"How close the {'asking rent' if is_rent else 'asking price'} is to the model-fair value.")
        st.metric("🛂 Compliance", f"{row['compliance_score']*100:.0f}/100")
        if row.get("aml_flag", 0) == 1:
            st.caption(f"⚠️ Transaction flagged: {row['pattern_types']} pattern detected.")
        else:
            st.caption("No AML/structuring pattern detected on this property's transaction history.")
    with f2:
        st.metric("🛡️ Trust", f"{trust*100:.0f}/100")
        if fraud_p is not None and fraud_p > 0.5:
            st.caption(f"⚠️ Flagged: {fraud_type} pattern detected ({fraud_p:.0%} model confidence).")
        else:
            st.caption("No fraud pattern detected on this listing.")
        st.metric("📈 Investment Value", f"{row['investment_score']*100:.0f}/100")
        st.caption(f"{row['avg_rental_yield_pct']:.2f}% yield · {row['avg_yoy_appreciation_pct']:.1f}%/yr appreciation.")

st.divider()
section_title("📐", "How this compares")

score_col = "rent_prism_score" if is_rent else "prism_score"
locality_scores = graph[(graph["locality"] == row["locality"]) & graph[score_col].notna()][score_col]
city_scores_cmp = graph[(graph["city"] == row["city"]) & graph[score_col].notna()][score_col]
locality_pct = (locality_scores < score).mean() * 100 if len(locality_scores) > 1 else 50
city_pct = (city_scores_cmp < score).mean() * 100 if len(city_scores_cmp) > 1 else 50

locality_price_avg = graph[graph["locality"] == row["locality"]]["price_per_sqft"].mean()
this_price_per_sqft = ask_value / row["carpet_area_sqft"] if row["carpet_area_sqft"] else 0
price_vs_locality = (this_price_per_sqft / locality_price_avg - 1) * 100 if locality_price_avg else 0

insight_grid([
    ("Rank within locality", f"Scores better than <b>{locality_pct:.0f}%</b> of properties in {row['locality']} ({len(locality_scores)} compared).",
     "success" if locality_pct >= 60 else ("warning" if locality_pct >= 35 else "danger")),
    ("Rank within city", f"Scores better than <b>{city_pct:.0f}%</b> of properties in {row['city']} ({len(city_scores_cmp)} compared).",
     "success" if city_pct >= 60 else ("warning" if city_pct >= 35 else "danger")),
    ("Price vs. locality average", f"₹{this_price_per_sqft:,.0f}/sqft — <b>{abs(price_vs_locality):.0f}% {'above' if price_vs_locality > 0 else 'below'}</b> the {row['locality']} average of ₹{locality_price_avg:,.0f}/sqft.",
     "warning" if abs(price_vs_locality) > 15 else "primary"),
])

st.divider()
section_title("⚖️", "This property vs. locality and city averages")

locality_avgs = graph[graph["locality"] == row["locality"]][
    ["price_fairness_score", "trust_score", "compliance_score", "investment_score"]
].mean()
city_avgs = graph[graph["city"] == row["city"]][
    ["price_fairness_score", "trust_score", "compliance_score", "investment_score"]
].mean()
compare_df = pd.DataFrame({
    "Dimension": ["Price Fairness", "Trust", "Compliance", "Investment Value"] * 3,
    "Scope": ["This Property"] * 4 + [f"{row['locality']} Avg"] * 4 + [f"{row['city']} Avg"] * 4,
    "Score": (
        [price_fairness * 100, trust * 100, row["compliance_score"] * 100, row["investment_score"] * 100]
        + [locality_avgs["price_fairness_score"] * 100, locality_avgs["trust_score"] * 100,
           locality_avgs["compliance_score"] * 100, locality_avgs["investment_score"] * 100]
        + [city_avgs["price_fairness_score"] * 100, city_avgs["trust_score"] * 100,
           city_avgs["compliance_score"] * 100, city_avgs["investment_score"] * 100]
    ),
})
fig_compare = px.bar(
    compare_df, x="Dimension", y="Score", color="Scope", barmode="group",
    title="Score breakdown vs. locality and city averages",
    color_discrete_sequence=[band_color, PRIMARY, "#94A3B8"],
)
fig_compare.update_layout(height=480, yaxis_title="Score (0-100)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_compare, use_container_width=True)

st.divider()
section_title("📊", "Where this property sits in its local price range")

locality_df = graph[graph["locality"] == row["locality"]].copy()
hist_col = "rent_ask_value" if is_rent else "sale_ask_value"
locality_df["value_per_sqft"] = locality_df[hist_col] / locality_df["carpet_area_sqft"]
fig_hist = px.histogram(
    locality_df, x="value_per_sqft", nbins=30,
    title=f"{'Rent' if is_rent else 'Price'}/sqft distribution across {len(locality_df)} properties in {row['locality']}",
    color_discrete_sequence=[PRIMARY],
)
fig_hist.add_vline(x=this_price_per_sqft, line_width=3, line_dash="dash", line_color=DANGER,
                     annotation_text="This property", annotation_position="top")
fig_hist.update_layout(height=480, xaxis_title=f"{'Rent' if is_rent else 'Price'}/sqft (₹)",
                         yaxis_title="Number of properties", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_hist, use_container_width=True)

if band == "Needs Review":
    st.error("**Our recommendation:** This listing shows one or more risk signals. Verify RERA registration, review the transaction/compliance history, and consider an independent site visit before proceeding.")
elif band == "Fair":
    st.warning("**Our recommendation:** Generally acceptable, but review the price fairness and/or compliance details above before committing.")
else:
    st.success("**Our recommendation:** This property checks out well across price, trust, and compliance signals.")
