import streamlit as st
import pandas as pd
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, disclosure, PRIMARY, ACCENT, SUCCESS, DANGER

st.set_page_config(page_title="PRISM — Buyer App", page_icon="🏠", layout="wide")
inject_css()
page_header("🏠 Buyer App", "One score, four checks — is this property fairly priced, trustworthy, and on-schedule?")

BASE = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_graph():
    return pd.read_csv(os.path.join(BASE, "data", "unified_property_graph.csv"))

graph = load_graph()

st.markdown("#### Find a property")
c1, c2, c3 = st.columns(3)
with c1:
    city = st.selectbox("City", sorted(graph["city"].unique()))
with c2:
    localities = sorted(graph[graph["city"] == city]["locality"].unique())
    locality = st.selectbox("Locality", localities)
with c3:
    subset = graph[(graph["city"] == city) & (graph["locality"] == locality)]
    prop_id = st.selectbox(
        "Property", subset["property_id"].tolist(),
        format_func=lambda pid: f"#{pid} — {subset[subset['property_id']==pid]['bhk'].iloc[0]}BHK, "
                                  f"{subset[subset['property_id']==pid]['carpet_area_sqft'].iloc[0]:.0f} sqft"
    )

row = graph[graph["property_id"] == prop_id].iloc[0]

st.divider()

score = int(row["prism_score"])
band = row["prism_band"]
band_color = {"Excellent": SUCCESS, "Good": PRIMARY, "Fair": ACCENT, "Needs Review": DANGER}[band]

colScore, colDetails = st.columns([1, 2])
with colScore:
    st.markdown(f"""
    <div style="text-align:center; padding:1.5rem; background:#F8FAFC; border-radius:14px; border:2px solid {band_color};">
        <div style="font-size:0.9rem; color:#64748B;">PRISM Score</div>
        <div style="font-size:3rem; font-weight:700; color:{band_color};">{score}</div>
        <div style="font-size:1.1rem; font-weight:600; color:{band_color};">{band}</div>
        <div style="font-size:0.78rem; color:#94A3B8; margin-top:4px;">Scale: 300–900</div>
    </div>
    """, unsafe_allow_html=True)

with colDetails:
    st.markdown(f"**{row['bhk']}BHK · {row['carpet_area_sqft']:.0f} sqft · {row['locality']}, {row['city']}**")
    st.caption(f"Builder: {row['builder']} · RERA: {'✅ Registered' if row['rera_registered'] else '❌ Not registered'} · Age: {row['age_years']} yrs")
    listed = row["listed_price"] if pd.notna(row["listed_price"]) else row["fair_value_est"]
    st.metric("Listed / estimated price", f"₹{listed:,.0f}", f"{row['price_deviation_pct']:+.1f}% vs. fair value" if pd.notna(row.get("price_deviation_pct")) else None)

st.divider()
st.markdown("#### The four checks behind this score")

f1, f2, f3, f4 = st.columns(4)
with f1:
    st.metric("💰 Price Fairness", f"{row['price_fairness_score']*100:.0f}/100")
    st.caption("How close the asking price is to the model-fair value for this locality/spec.")
with f2:
    st.metric("🛡️ Trust", f"{row['trust_score']*100:.0f}/100")
    fraud_p = row.get("fraud_probability", 0)
    if fraud_p > 0.5:
        st.caption(f"⚠️ Flagged: {row['fraud_type']} pattern detected ({fraud_p:.0%} model confidence).")
    else:
        st.caption("No fraud pattern detected on this listing.")
with f3:
    st.metric("🏗️ Delivery", f"{row['delivery_score']*100:.0f}/100")
    if row["under_construction"]:
        st.caption(f"Stage {int(row['actual_stage'])}/5 actual vs. stage {int(row['promised_stage'])}/5 promised.")
    else:
        st.caption("Ready to move — no construction risk.")
with f4:
    st.metric("📈 Investment Value", f"{row['investment_score']*100:.0f}/100")
    st.caption(f"{row['avg_rental_yield_pct']:.2f}% yield · {row['avg_yoy_appreciation_pct']:.1f}%/yr appreciation in this locality.")

st.divider()
if band == "Needs Review":
    st.error("**Our recommendation:** This listing shows one or more risk signals. Verify RERA registration, request the builder's payment schedule, and consider an independent site visit before proceeding.")
elif band == "Fair":
    st.warning("**Our recommendation:** Generally acceptable, but review the price fairness and/or delivery details above before committing.")
else:
    st.success("**Our recommendation:** This property checks out well across price, trust, and delivery signals.")

disclosure(
    "The PRISM Score combines price fairness, fraud/trust signals, construction delivery risk, "
    "and locality investment value into one number — all computed on the synthetic, "
    "regulation-calibrated dataset described in the other modules. Treat this as a demonstration "
    "of a unified scoring methodology, not a real property assessment."
)
