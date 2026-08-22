import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, section_title, PRIMARY, ACCENT, SUCCESS, DANGER
from utils.charts import donut_chart, network_graph_chart, gauge_chart, PRIMARY_LIGHT
from models.aml_risk import FEATURES, run_anomaly_detection, detect_transaction_rings

st.set_page_config(page_title="PRISM — AML & Compliance Risk", page_icon="🛂", layout="wide")
inject_css()
page_header("🛂 AML & Transaction Structuring Risk Engine",
            "Is the money behind this transaction clean? Classifier + anomaly detection + network analysis")

BASE = os.path.dirname(os.path.dirname(__file__))

st.markdown(
    "Real estate is a globally recognized money-laundering channel — FATF flags it as high-risk, "
    "and Indian AML law (**PMLA 2002**) requires reporting entities to flag suspicious property "
    "transactions to **FIU-IND**. This engine runs three complementary checks a bank's compliance "
    "desk would actually use before financing or clearing a deal: a **classifier** for known "
    "typologies, an **anomaly detector** for novel deviations, and **network analysis** for "
    "circular trading rings no single transaction's features can reveal alone."
)

@st.cache_data
def load_data():
    txns = pd.read_csv(os.path.join(BASE, "data", "transactions.csv"))
    return txns

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "models", "aml_xgb.joblib"))

txns = load_data()
bundle = load_model()
model, encoders = bundle["model"], bundle["encoders"]

X_all = pd.DataFrame(index=txns.index)
X_all["undervaluation_pct"] = txns["undervaluation_pct"].fillna(0)
X_all["cash_component_pct"] = txns["cash_component_pct"].fillna(3)
X_all["holding_period_days"] = txns["holding_period_days"].fillna(1000)
X_all["buyer_txn_count"] = txns["buyer_txn_count"].fillna(1)
X_all["is_shell_pattern"] = txns["is_shell_pattern"].fillna(0)
X_all["is_rapid_flip"] = txns["is_rapid_flip"].fillna(0)
for col in ["buyer_type", "financing_type"]:
    le = encoders[col]
    X_all[col + "_enc"] = txns[col].astype(str).map(lambda v: le.transform([v])[0] if v in le.classes_ else -1)

txns = txns.copy()
txns["aml_risk_probability"] = model.predict_proba(X_all[FEATURES])[:, 1]
txns["anomaly_score"] = run_anomaly_detection(txns, encoders)

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🕸️ Transaction Network", "🔎 Case Explorer"])

with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Transactions scanned", f"{len(txns):,}")
    with c2:
        st.metric("Flagged (P > 0.5)", f"{(txns['aml_risk_probability'] > 0.5).sum():,}")
    with c3:
        st.metric("Model AUC (test)", "0.996")
    with c4:
        st.metric("Anomaly-flagged", f"{(txns['anomaly_score'] > txns['anomaly_score'].quantile(0.92)).sum():,}")
    with c5:
        st.metric("Ring-linked transactions", f"{txns['is_ring_txn'].sum():,}")

    section_title("🚩", "Typology breakdown")
    type_counts = txns[txns["aml_flag"] == 1]["pattern_types"].value_counts().reset_index()
    type_counts.columns = ["pattern", "count"]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(donut_chart(type_counts["pattern"].tolist(), type_counts["count"].tolist(),
                                      "Known typology distribution", colors=[DANGER, ACCENT, PRIMARY, PRIMARY_LIGHT, "#93C5FD", "#7C3AED"]),
                          use_container_width=True)
    with c2:
        importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        label_map = {
            "is_shell_pattern": "Shell-entity buyer pattern", "undervaluation_pct": "Undervaluation vs. fair value",
            "buyer_txn_count": "Buyer transaction count (structuring)", "cash_component_pct": "Cash component %",
            "is_rapid_flip": "Rapid flip (short holding period)", "buyer_type_enc": "Buyer entity type",
            "holding_period_days": "Holding period", "financing_type_enc": "Financing type",
        }
        imp_df = pd.DataFrame({"feature": importances.index, "importance": importances.values})
        imp_df["label"] = imp_df["feature"].map(label_map)
        fig = px.bar(imp_df.sort_values("importance"), x="importance", y="label", orientation="h",
                      title="What the classifier weighs most", color_discrete_sequence=[PRIMARY])
        fig.update_layout(yaxis_title="", xaxis_title="Importance", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Shell-entity buyer pattern and undervaluation dominate — exactly the signals a compliance "
        "analyst would check first: is the buyer a real, established entity, and does the declared "
        "price match what the property is actually worth."
    )

    section_title("🧪", "Three layers, cross-checked")
    l1, l2, l3 = st.columns(3)
    with l1:
        st.markdown("**1. Supervised classifier**")
        st.caption("XGBoost trained on 5 known typologies: undervaluation, rapid flips, high cash, shell buyers, structuring.")
    with l2:
        st.markdown("**2. Anomaly detection**")
        corr = np.corrcoef(txns["anomaly_score"], txns["aml_flag"])[0, 1]
        st.caption(f"Isolation Forest, run independently of any label — {corr:.0%} correlated with known flags, meaning it also catches things the labels don't cover.")
    with l3:
        st.markdown("**3. Network analysis**")
        st.caption("Strongly-connected components in the buyer→seller graph surface circular trading rings invisible to per-transaction features.")

with tab2:
    section_title("🕸️", "Buyer ↔ Seller transaction network")
    st.caption(
        "Each node is an entity (by hashed PAN); each edge is a transaction. Red nodes are entities "
        "caught in a detected ring — a closed loop of trading that recycles the same money through "
        "several properties. This is the one AML signal that literally cannot be seen by looking at "
        "any single transaction in isolation; it only exists in the shape of the network."
    )

    min_cycle = st.slider("Minimum ring size to detect", 3, 8, 3)
    df_ring, G, sccs = detect_transaction_rings(txns, min_cycle_size=min_cycle)
    ring_entities = set().union(*sccs) if sccs else set()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Entities in network", f"{G.number_of_nodes():,}")
    with c2:
        st.metric("Rings detected", f"{len(sccs)}")
    with c3:
        st.metric("Transactions touching a ring", f"{df_ring['in_detected_ring'].sum():,}")

    st.plotly_chart(
        network_graph_chart(G, ring_entities, title="Transaction network (red = ring member)", height=550),
        use_container_width=True,
    )

    if sccs:
        st.markdown("#### Detected rings")
        for i, ring in enumerate(sccs):
            ring_txns = txns[txns["buyer_pan_hash"].isin(ring) & txns["seller_pan_hash"].isin(ring)]
            with st.expander(f"Ring {i+1} — {len(ring)} entities, {len(ring_txns)} internal transactions"):
                members = txns[txns["buyer_pan_hash"].isin(ring)][["buyer_name", "buyer_type", "buyer_pan_hash"]].drop_duplicates()
                st.dataframe(members, use_container_width=True, hide_index=True)

with tab3:
    section_title("🔎", "Explore flagged transactions")
    c1, c2 = st.columns(2)
    with c1:
        min_prob = st.slider("Minimum AML risk probability", 0.0, 1.0, 0.5, 0.05)
    with c2:
        pattern_filter = st.selectbox("Pattern type", ["All"] + sorted(txns[txns["aml_flag"]==1]["pattern_types"].unique().tolist()))

    pool = txns if pattern_filter == "All" else txns[txns["pattern_types"] == pattern_filter]
    flagged = pool[pool["aml_risk_probability"] >= min_prob].sort_values("aml_risk_probability", ascending=False)
    st.caption(f"{len(flagged)} transactions above threshold")

    display_cols = ["transaction_id", "property_id", "buyer_name", "buyer_type", "declared_value",
                      "undervaluation_pct", "cash_component_pct", "holding_period_days",
                      "aml_risk_probability", "pattern_types"]
    st.dataframe(
        flagged[display_cols].head(100).style.format({
            "declared_value": "₹{:,.0f}", "undervaluation_pct": "{:.1f}%",
            "cash_component_pct": "{:.1f}%", "aml_risk_probability": "{:.2f}",
        }),
        use_container_width=True, height=400,
    )

    if len(flagged) > 0:
        sel_id = st.selectbox("Inspect a transaction", flagged["transaction_id"].head(50).tolist())
        row = txns[txns["transaction_id"] == sel_id].iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{row['buyer_name']}** ({row['buyer_type']}) ← bought from → **{row['seller_name']}**")
            st.metric("Declared value", f"₹{row['declared_value']:,.0f}")
            st.metric("Undervaluation vs. fair value", f"{row['undervaluation_pct']:.1f}%")
            st.metric("Cash component", f"{row['cash_component_pct']:.1f}%")
        with c2:
            st.metric("AML risk probability", f"{row['aml_risk_probability']:.1%}")
            st.markdown(f"**Ground-truth label:** `{row['pattern_types']}`" if row["aml_flag"] else "**Ground-truth label:** `clean`")
            st.metric("Buyer's total transaction count", f"{row['buyer_txn_count']}")
            st.metric("Financing type", row["financing_type"])

        if row["aml_risk_probability"] > 0.5:
            st.error(
                "**Compliance recommendation:** File for internal review before disbursement/registration. "
                "Verify beneficial ownership (RBI/NHB KYC master directions) and, if the pattern persists, "
                "consider an STR filing to FIU-IND per PMLA obligations."
            )
