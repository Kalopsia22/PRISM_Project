import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, PRIMARY, ACCENT
from models.yield_recommender import score_for_profile, RISK_PROFILES

st.set_page_config(page_title="PRISM — Rental Yield", page_icon="📈", layout="wide")
inject_css()
page_header("📈 Rental Yield & Investment Recommender", "Risk-profile-aware locality recommendations, personalized to investor goals")

BASE = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE, "data", "locality_profile.csv"))

locality = load_data()

st.markdown("#### Tell us your investor profile")
c1, c2, c3 = st.columns(3)
with c1:
    risk_profile = st.selectbox(
        "Risk profile", list(RISK_PROFILES.keys()),
        format_func=lambda x: x.replace("_", " ").title(),
    )
with c2:
    city_pref = st.multiselect("City preference", sorted(locality["city"].unique()), default=list(locality["city"].unique()))
with c3:
    budget_tier = st.multiselect("Tier preference", sorted(locality["tier"].unique()), default=list(locality["tier"].unique()))

weights = RISK_PROFILES[risk_profile]
st.caption(
    f"**{risk_profile.replace('_',' ').title()}** weighting → "
    f"Yield: {weights['yield_weight']:.0%} · Appreciation: {weights['appreciation_weight']:.0%} · "
    f"Stability: {weights['stability_weight']:.0%}"
)

filtered = locality[locality["city"].isin(city_pref) & locality["tier"].isin(budget_tier)]
scored = score_for_profile(filtered, risk_profile)

st.markdown("#### Top recommended micro-markets for your profile")
top5 = scored.head(5)

for _, row in top5.iterrows():
    with st.container():
        cols = st.columns([2, 1, 1, 1, 1])
        cols[0].markdown(f"**{row['locality']}**, {row['city']} · _{row['tier']}_")
        cols[1].metric("Yield", f"{row['avg_rental_yield_pct']:.2f}%")
        cols[2].metric("Appreciation", f"{row['avg_yoy_appreciation_pct']:.1f}%/yr")
        cols[3].metric("Stability", f"{row['stability_score']:.2f}")
        cols[4].metric("Score", f"{row['recommendation_score']:.2f}")

st.divider()

st.markdown("#### Yield vs. appreciation across all micro-markets")
fig = px.scatter(
    scored, x="avg_rental_yield_pct", y="avg_yoy_appreciation_pct",
    size="recommendation_score", color="tier", hover_data=["locality", "city"],
    title="Where localities sit on the yield/growth tradeoff",
    color_discrete_sequence=[PRIMARY, "#3B82F6", "#93C5FD", ACCENT],
    render_mode="svg",
)
fig.update_layout(xaxis_title="Avg rental yield (%)", yaxis_title="Avg YoY appreciation (%)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### How each risk profile re-ranks the same localities")
compare_rows = []
for rp in RISK_PROFILES:
    ranked = score_for_profile(locality, rp).head(3)
    for rank, (_, r) in enumerate(ranked.iterrows(), 1):
        compare_rows.append({"Risk Profile": rp.replace("_", " ").title(), "Rank": rank, "Locality": f"{r['locality']} ({r['city']})"})
compare_df = pd.DataFrame(compare_rows).pivot(index="Rank", columns="Risk Profile", values="Locality")
st.dataframe(compare_df, use_container_width=True)
