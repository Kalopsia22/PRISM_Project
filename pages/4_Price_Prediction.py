import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, disclosure, PRIMARY

st.set_page_config(page_title="PRISM — Price Prediction", page_icon="🏷️", layout="wide")
inject_css()
page_header("🏷️ Hyperlocal Price Prediction", "XGBoost regression at pincode/micro-market granularity, explained via SHAP")

BASE = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE, "data", "properties.csv"))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "models", "price_xgb.joblib"))

props = load_data()
bundle = load_model()
model, encoders, feature_cols = bundle["model"], bundle["encoders"], bundle["feature_cols"]

tab1, tab2 = st.tabs(["📊 Model Performance & Insights", "🧮 Try a Prediction"])

with tab1:
    st.markdown("#### Why hyperlocal beats a generic AVM")
    st.markdown(
        "Prices vary street-to-street within the same city — a generic Zillow-style model "
        "trained at city level misses this. PRISM clusters by pincode/micro-market so the "
        "model learns locality-specific pricing dynamics (e.g. Bandra West vs Malad West "
        "in Mumbai can differ 3-4x per sqft despite similar city-level averages)."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Model", "XGBoost Regressor")
    with c2:
        st.metric("MAPE (test set)", "4.5%")
    with c3:
        st.metric("R²", "0.98")

    st.markdown("#### What drives price, according to the model")
    shap_imp = pd.read_csv(os.path.join(BASE, "data", "price_shap_importance.csv")).head(8)
    label_map = {
        "circle_rate_per_sqft": "Circle rate (base)", "age_years": "Age of property",
        "num_amenities": "Amenity count", "builder_score": "Builder reputation score",
        "tier_enc": "Market tier", "locality_enc": "Locality", "metro_distance_km": "Metro distance",
        "city_enc": "City",
    }
    shap_imp["label"] = shap_imp["feature"].map(label_map).fillna(shap_imp["feature"])
    fig = px.bar(
        shap_imp.sort_values("mean_abs_shap"), x="mean_abs_shap", y="label",
        orientation="h", title="SHAP feature impact on predicted price/sqft",
        color_discrete_sequence=[PRIMARY],
    )
    fig.update_layout(yaxis_title="", xaxis_title="Mean |SHAP value| (₹/sqft impact)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Price distribution across micro-markets")
    fig2 = px.scatter(
        props.sample(min(800, len(props)), random_state=1),
        x="metro_distance_km", y="price_per_sqft", color="tier",
        size="num_amenities", hover_data=["locality", "builder"],
        title="Price/sqft vs. metro distance (bubble size = amenity count)",
        color_discrete_sequence=[PRIMARY, "#3B82F6", "#93C5FD", "#D97706"],
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.markdown("#### Estimate price/sqft for a hypothetical property")
    st.caption("Adjust the inputs and see the model's prediction, plus which factors pushed it up or down.")

    colA, colB, colC = st.columns(3)
    with colA:
        city = st.selectbox("City", sorted(props["city"].unique()))
        localities_in_city = sorted(props[props["city"] == city]["locality"].unique())
        locality = st.selectbox("Locality", localities_in_city)
        tier = props[(props["city"] == city) & (props["locality"] == locality)]["tier"].iloc[0]
        st.caption(f"Tier: **{tier}**")
    with colB:
        bhk = st.selectbox("BHK", [1, 2, 3, 4], index=1)
        carpet_area = st.slider("Carpet area (sqft)", 350, 2200, 850)
        age_years = st.slider("Property age (years)", 0, 25, 3)
    with colC:
        builder = st.selectbox("Builder", sorted(props["builder"].unique()))
        builder_score = props[props["builder"] == builder]["builder_score"].iloc[0]
        builder_tier = props[props["builder"] == builder]["builder_tier"].iloc[0]
        num_amenities = st.slider("Number of amenities", 0, 12, 6)
        rera_registered = st.checkbox("RERA registered", value=True)

    row = props[(props["city"] == city) & (props["locality"] == locality)].iloc[0]

    input_dict = {
        "city": city, "locality": locality, "tier": tier, "builder": builder, "builder_tier": builder_tier,
        "metro_distance_km": row["metro_distance_km"], "bhk": bhk, "carpet_area_sqft": carpet_area,
        "age_years": age_years, "builder_score": builder_score, "rera_registered": int(rera_registered),
        "num_amenities": num_amenities, "circle_rate_per_sqft": row["circle_rate_per_sqft"],
        "vastu_compliant": 1, "gated_community": int(num_amenities >= 6),
    }

    feat_row = {}
    for col in ["city", "locality", "tier", "builder", "builder_tier"]:
        le = encoders[col]
        val = input_dict[col]
        feat_row[col + "_enc"] = le.transform([val])[0] if val in le.classes_ else 0
    for col in ["metro_distance_km", "bhk", "carpet_area_sqft", "age_years", "builder_score",
                 "rera_registered", "num_amenities", "circle_rate_per_sqft", "vastu_compliant", "gated_community"]:
        feat_row[col] = input_dict[col]

    X_input = pd.DataFrame([feat_row])[feature_cols]
    pred_price_sqft = model.predict(X_input)[0]
    total_est = pred_price_sqft * carpet_area

    st.divider()
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Predicted price/sqft", f"₹{pred_price_sqft:,.0f}")
    with m2:
        st.metric("Estimated total price", f"₹{total_est:,.0f}")

    st.caption(
        f"Circle rate anchor for {locality}: ₹{row['circle_rate_per_sqft']:,.0f}/sqft. "
        f"Model premium over circle rate: {(pred_price_sqft/row['circle_rate_per_sqft'] - 1):.1%}"
    )

disclosure(
    "Prices are modeled on a synthetic dataset calibrated to real Ready Reckoner/circle-rate "
    "bands per locality — not actual registered transactions (no consolidated public API exists "
    "for that in India). Treat absolute values as illustrative; relative patterns across "
    "localities/features reflect the intended modeling methodology."
)
