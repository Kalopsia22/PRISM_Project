import streamlit as st
import pandas as pd
import json
import sys, os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, disclosure, PRIMARY

st.set_page_config(page_title="PRISM — Lender API", page_icon="🏦", layout="wide")
inject_css()
page_header("🏦 Lender API", "Programmatic disbursement-decision endpoint for construction-linked home loans")

BASE = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_graph():
    return pd.read_csv(os.path.join(BASE, "data", "unified_property_graph.csv"))

graph = load_graph()

st.markdown("""
Banks disbursing construction-linked loans release funds in tranches tied to verified
construction milestones. This endpoint gives a lender's loan-origination system a single
call to check: is this property's construction on schedule, is the listing/builder trustworthy,
and is the price consistent with market value — before releasing the next tranche.
""")

st.markdown("#### Try the endpoint")
c1, c2 = st.columns(2)
with c1:
    city = st.selectbox("City", sorted(graph["city"].unique()), key="lender_city")
with c2:
    localities = sorted(graph[graph["city"] == city]["locality"].unique())
    locality = st.selectbox("Locality", localities, key="lender_locality")

subset = graph[(graph["city"] == city) & (graph["locality"] == locality)]
prop_id = st.selectbox("Property ID", subset["property_id"].tolist(), key="lender_prop")
row = graph[graph["property_id"] == prop_id].iloc[0]

st.code(f"GET /v1/properties/{prop_id}/disbursement-check", language="http")

# Build the disbursement decision
under_construction = bool(row["under_construction"])
schedule_delta = int(row["schedule_delta"]) if pd.notna(row["schedule_delta"]) else 0
fraud_prob = float(row.get("fraud_probability", 0.0))
prism_score = int(row["prism_score"])

if not under_construction:
    decision = "NOT_APPLICABLE"
    reason = "Property is ready-to-move; no construction-linked tranche pending."
elif fraud_prob > 0.5:
    decision = "HOLD"
    reason = f"Listing/builder trust check failed ({row.get('fraud_type', 'unknown')} pattern, {fraud_prob:.0%} confidence). Manual review required before any disbursement."
elif schedule_delta <= -2:
    decision = "HOLD"
    reason = f"Construction materially behind schedule (stage {int(row['actual_stage'])}/5 actual vs. {int(row['promised_stage'])}/5 promised). Recommend site re-inspection."
elif schedule_delta == -1:
    decision = "CONDITIONAL_APPROVE"
    reason = f"Minor schedule lag (stage {int(row['actual_stage'])}/5 vs. {int(row['promised_stage'])}/5 promised). Approve with follow-up inspection in 30 days."
else:
    decision = "APPROVE"
    reason = "Construction on schedule, no trust flags, price consistent with locality fair value."

response_payload = {
    "property_id": int(prop_id),
    "checked_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "prism_score": prism_score,
    "prism_band": row["prism_band"],
    "checks": {
        "price_fairness_score": round(float(row["price_fairness_score"]), 3),
        "trust_score": round(float(row["trust_score"]), 3),
        "fraud_probability": round(fraud_prob, 3),
        "rera_registered": bool(row["rera_registered"]),
        "construction": {
            "under_construction": under_construction,
            "actual_stage": int(row["actual_stage"]) if under_construction else None,
            "promised_stage": int(row["promised_stage"]) if under_construction else None,
            "schedule_delta": schedule_delta if under_construction else None,
        },
    },
    "disbursement_decision": decision,
    "reason": reason,
}

st.markdown("#### Response")
st.json(response_payload)

decision_color = {"APPROVE": "success", "CONDITIONAL_APPROVE": "warning", "HOLD": "error", "NOT_APPLICABLE": "info"}
getattr(st, decision_color[decision])(f"**Decision: {decision}** — {reason}")

st.divider()
st.markdown("#### Portfolio-level exposure view (batch endpoint)")
st.caption("`GET /v1/portfolio/disbursement-summary` — how a lender would monitor exposure across an entire loan book")

uc = graph[graph["under_construction"] == True].copy()
uc["decision"] = uc.apply(
    lambda r: "HOLD" if (r.get("fraud_probability", 0) > 0.5 or r["schedule_delta"] <= -2)
    else ("CONDITIONAL_APPROVE" if r["schedule_delta"] == -1 else "APPROVE"),
    axis=1,
)
summary = uc["decision"].value_counts().reset_index()
summary.columns = ["decision", "count"]
c1, c2, c3 = st.columns(3)
for col, dec in zip([c1, c2, c3], ["APPROVE", "CONDITIONAL_APPROVE", "HOLD"]):
    cnt = summary[summary["decision"] == dec]["count"].sum() if dec in summary["decision"].values else 0
    col.metric(dec.replace("_", " "), f"{cnt}", f"{cnt/max(len(uc),1):.0%} of under-construction book")

disclosure(
    "This is a simulated API response for demonstration — no real lending decision engine is "
    "connected. In production, this endpoint would sit behind loan-origination-system auth, "
    "pull real construction-progress verification (drone/inspector reports), and log every "
    "decision for audit per RBI disbursement-linked lending norms."
)
