import streamlit as st
import pandas as pd
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, PRIMARY, ACCENT, SUCCESS, DANGER

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
    st.markdown(f"""
    <div style="text-align:center; padding:1.5rem; background:#F8FAFC; border-radius:14px; border:2px solid {band_color};">
        <div style="font-size:0.9rem; color:#64748B;">PRISM Score</div>
        <div style="font-size:3rem; font-weight:700; color:{band_color};">{score}</div>
        <div style="font-size:1.1rem; font-weight:600; color:{band_color};">{band}</div>
        <div style="font-size:0.78rem; color:#94A3B8; margin-top:4px;">Scale: 300–900</div>
    </div>
    """, unsafe_allow_html=True)

with colDetails:
    st.markdown(f"**{row['property_type']} · {row['bhk']}BHK · {row['carpet_area_sqft']:.0f} sqft · {row['locality']}, {row['city']}** ({row['city_tier']})")
    st.caption(f"Builder: {row['builder']} · RERA: {'✅ Registered' if row['rera_registered'] else '❌ Not registered'} · Age: {row['age_years']} yrs")
    st.metric(value_label, f"₹{ask_value:,.0f}", f"{deviation:+.1f}% vs. fair value" if pd.notna(deviation) else None)

st.divider()
st.markdown("#### The four checks behind this score")

f1, f2, f3, f4 = st.columns(4)
with f1:
    st.metric("💰 Price Fairness", f"{price_fairness*100:.0f}/100")
    st.caption(f"How close the {'asking rent' if is_rent else 'asking price'} is to the model-fair value.")
with f2:
    st.metric("🛡️ Trust", f"{trust*100:.0f}/100")
    if fraud_p is not None and fraud_p > 0.5:
        st.caption(f"⚠️ Flagged: {fraud_type} pattern detected ({fraud_p:.0%} model confidence).")
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
