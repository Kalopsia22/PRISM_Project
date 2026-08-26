import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, section_title, insight_grid, PRIMARY, ACCENT, SUCCESS
from utils.charts import radar_chart, treemap_chart, donut_chart, map_chart, PRIMARY_LIGHT
from utils.geo import add_jittered_coords
from models.yield_recommender import score_for_profile, RISK_PROFILES

st.set_page_config(page_title="PRISM — Rental Yield", page_icon="📈", layout="wide")
inject_css()
page_header("📈 Rental Yield & Investment Recommender", "Risk-profile-aware locality recommendations, personalized to investor goals")
st.caption(
    "Yield formula calibrated to the **Magicbricks Rental Index** (Jan–Mar 2026 report), which "
    "publishes trailing rental yields for Chennai (4.87%), Kolkata (4.81%), Bengaluru (4.19%), "
    "and Hyderabad (4.06%) — fit against those cities' price levels and applied locality-wide."
)

BASE = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE, "data", "locality_profile.csv"))

locality = load_data()

section_title("🧑‍💼", "Tell us your investor profile")
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

rc1, rc2 = st.columns([1, 2])
with rc1:
    st.plotly_chart(
        radar_chart(["Yield", "Appreciation", "Stability"],
                     [weights["yield_weight"], weights["appreciation_weight"], weights["stability_weight"]],
                     title=f"{risk_profile.replace('_',' ').title()} weighting", color=ACCENT,
                     height=280, range_max=max(weights.values())),
        use_container_width=True,
    )
with rc2:
    st.caption(
        f"**{risk_profile.replace('_',' ').title()}** weighting → "
        f"Yield: {weights['yield_weight']:.0%} · Appreciation: {weights['appreciation_weight']:.0%} · "
        f"Stability: {weights['stability_weight']:.0%}"
    )
    filtered = locality[locality["city"].isin(city_pref) & locality["tier"].isin(budget_tier)]
    scored = score_for_profile(filtered, risk_profile)
    st.metric("Micro-markets matching filters", f"{len(scored)}")
    st.metric("Best-fit locality", f"{scored.iloc[0]['locality']}, {scored.iloc[0]['city']}" if len(scored) else "—")

section_title("🏆", "Top recommended micro-markets for your profile")
top5 = scored.head(5)

for _, row in top5.iterrows():
    with st.container():
        cols = st.columns([2, 1, 1, 1, 1])
        cols[0].markdown(f"**{row['locality']}**, {row['city']} · _{row['tier']}_")
        cols[1].metric("Yield", f"{row['avg_rental_yield_pct']:.2f}%")
        cols[2].metric("Appreciation", f"{row['avg_yoy_appreciation_pct']:.1f}%/yr")
        cols[3].metric("Stability", f"{row['stability_score']:.2f}")
        cols[4].metric("Score", f"{row['recommendation_score']:.2f}")

if len(top5) > 0:
    market_avg_yield = locality["avg_rental_yield_pct"].mean()
    top5_avg_yield = top5["avg_rental_yield_pct"].mean()
    yield_edge = top5_avg_yield - market_avg_yield
    tier_count = top5["tier"].value_counts()
    insight_grid([
        ("Yield edge from personalization", f"Your top 5 average <b>{top5_avg_yield:.2f}%</b> yield vs the market average of <b>{market_avg_yield:.2f}%</b> — a {yield_edge:+.2f}pp edge.", "success" if yield_edge > 0 else "warning"),
        ("Dominant locality tier", f"<b>{tier_count.index[0]}</b> localities make up {tier_count.iloc[0]} of your top 5 picks under this risk profile.", "primary"),
    ])

section_title("🗺️", "Where the top-5 picks are on the map")
if len(top5) > 0:
    top5_geo = add_jittered_coords(top5)
    st.plotly_chart(
        map_chart(top5_geo, color_col="recommendation_score", size_col="avg_rental_yield_pct",
                   hover_name="locality", hover_data={"city": True, "avg_yoy_appreciation_pct": ":.1f"},
                   title="Top-5 picks — color = recommendation score, size = yield", height=380, zoom=3.6),
        use_container_width=True,
    )

st.divider()

section_title("📊", "Yield vs. appreciation across all micro-markets")
fig = px.scatter(
    scored, x="avg_rental_yield_pct", y="avg_yoy_appreciation_pct",
    size="recommendation_score", color="tier", hover_data=["locality", "city"],
    title="Where localities sit on the yield/growth tradeoff",
    color_discrete_sequence=[PRIMARY, PRIMARY_LIGHT, "#93C5FD", ACCENT],
    render_mode="svg",
)
fig.update_layout(xaxis_title="Avg rental yield (%)", yaxis_title="Avg YoY appreciation (%)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    tier_mix = scored["tier"].value_counts()
    st.plotly_chart(donut_chart(tier_mix.index.tolist(), tier_mix.values.tolist(), "Locality quality tier mix",
                                  colors=[SUCCESS, PRIMARY, PRIMARY_LIGHT, ACCENT]), use_container_width=True)
with c2:
    city_mix = scored.groupby("city")["recommendation_score"].mean().sort_values(ascending=False).head(8)
    fig_city = px.bar(
        x=city_mix.values, y=city_mix.index, orientation="h",
        title="Top 8 cities by avg recommendation score",
        color=city_mix.values, color_continuous_scale=[[0, "#DBEAFE"], [1, PRIMARY]],
    )
    fig_city.update_layout(yaxis_title="", xaxis_title="Avg score", coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_city, use_container_width=True)

section_title("🔀", "How each risk profile re-ranks the same localities")
compare_rows = []
for rp in RISK_PROFILES:
    ranked = score_for_profile(locality, rp).head(3)
    for rank, (_, r) in enumerate(ranked.iterrows(), 1):
        compare_rows.append({"Risk Profile": rp.replace("_", " ").title(), "Rank": rank, "Locality": f"{r['locality']} ({r['city']})"})
compare_df = pd.DataFrame(compare_rows).pivot(index="Rank", columns="Risk Profile", values="Locality")
st.dataframe(compare_df, use_container_width=True)
