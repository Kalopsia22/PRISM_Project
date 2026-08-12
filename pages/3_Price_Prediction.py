import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, PRIMARY, ACCENT

st.set_page_config(page_title="PRISM — Price Prediction", page_icon="🏷️", layout="wide")
inject_css()
page_header("🏷️ Price & Rent Prediction", "XGBoost regression at pincode/micro-market granularity, explained via SHAP")

BASE = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE, "data", "properties.csv"))

@st.cache_resource
def load_models():
    price_bundle = joblib.load(os.path.join(BASE, "models", "price_xgb.joblib"))
    rent_bundle = joblib.load(os.path.join(BASE, "models", "rent_xgb.joblib"))
    return price_bundle, rent_bundle

props = load_data()
price_bundle, rent_bundle = load_models()

mode = st.radio("Predict for:", ["Sale (Possession)", "Rental"], horizontal=True)
is_rent = mode == "Rental"
bundle = rent_bundle if is_rent else price_bundle
model, encoders, feature_cols = bundle["model"], bundle["encoders"], bundle["feature_cols"]
target_label = "rent/sqft" if is_rent else "price/sqft"
shap_file = "rent_shap_importance.csv" if is_rent else "price_shap_importance.csv"

tab1, tab2 = st.tabs(["📊 Model Performance & Insights", "🧮 Try a Prediction"])

with tab1:
    st.markdown(f"#### Why hyperlocal beats a generic AVM ({'rent' if is_rent else 'price'} model)")
    st.markdown(
        "Prices vary street-to-street within the same city, and now also across 18 cities "
        "spanning Tier 1/2/3 India — a generic model trained at city level misses this. PRISM "
        "clusters by pincode/micro-market and property type so the model learns locality- and "
        "category-specific dynamics (a Villa in Gurgaon and a Studio in Dombivli don't move "
        "together)."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Model", "XGBoost Regressor")
    with c2:
        st.metric("MAPE (test set)", "20.7%" if is_rent else "4.9%")
    with c3:
        st.metric("R²", "0.85" if is_rent else "0.99")
    if is_rent:
        st.caption("Rent prediction is inherently noisier than sale price — real rental markets carry more landlord-level idiosyncrasy, which shows up here too.")

    st.markdown(f"#### What drives {target_label}, according to the model")
    shap_path = os.path.join(BASE, "data", shap_file)
    if os.path.exists(shap_path):
        shap_imp = pd.read_csv(shap_path).head(8)
        label_map = {
            "circle_rate_per_sqft": "Circle rate (base)", "age_years": "Age of property",
            "num_amenities": "Amenity count", "builder_score": "Builder reputation score",
            "property_type_enc": "Property type", "tier_enc": "Locality quality tier",
            "city_tier_enc": "City tier", "locality_enc": "Locality", "city_enc": "City",
            "metro_distance_km": "Metro distance", "builder_tier_enc": "Builder tier",
            "carpet_area_sqft": "Carpet area", "bhk": "BHK", "rera_registered": "RERA registered",
            "vastu_compliant": "Vastu compliant", "gated_community": "Gated community",
        }
        shap_imp["label"] = shap_imp["feature"].map(label_map).fillna(shap_imp["feature"])
        fig = px.bar(
            shap_imp.sort_values("mean_abs_shap"), x="mean_abs_shap", y="label",
            orientation="h", title=f"SHAP feature impact on predicted {target_label}",
            color_discrete_sequence=[PRIMARY],
        )
        fig.update_layout(yaxis_title="", xaxis_title=f"Mean |SHAP value| (₹/sqft impact)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Price by property type")
    type_col = "price_per_sqft" if not is_rent else None
    if is_rent:
        rentable = props[props["rentable"] == 1].copy()
        rentable["rent_per_sqft"] = rentable["monthly_rent_est"] / rentable["carpet_area_sqft"]
        fig_type = px.box(
            rentable, x="property_type", y="rent_per_sqft", color="city_tier",
            title="Rent/sqft distribution by property type and city tier",
            color_discrete_sequence=[PRIMARY, "#3B82F6", ACCENT],
        )
        fig_type.update_layout(yaxis_title="₹/sqft/month", xaxis_title="")
    else:
        fig_type = px.box(
            props, x="property_type", y="price_per_sqft", color="city_tier",
            title="Price/sqft distribution by property type and city tier",
            color_discrete_sequence=[PRIMARY, "#3B82F6", ACCENT],
        )
        fig_type.update_layout(yaxis_title="₹/sqft", xaxis_title="")
    fig_type.update_layout(xaxis_tickangle=-20)
    st.plotly_chart(fig_type, use_container_width=True)

    st.markdown("#### Price/rent by city tier")
    tier_agg = props.groupby("city_tier").agg(
        avg_price_per_sqft=("price_per_sqft", "mean"),
    ).reset_index()
    if is_rent:
        rentable2 = props[props["rentable"] == 1].copy()
        rentable2["rent_per_sqft"] = rentable2["monthly_rent_est"] / rentable2["carpet_area_sqft"]
        tier_agg2 = rentable2.groupby("city_tier")["rent_per_sqft"].mean().reset_index()
        fig_tier = px.bar(tier_agg2, x="city_tier", y="rent_per_sqft", color="city_tier",
                            title="Avg rent/sqft by city tier",
                            color_discrete_sequence=[PRIMARY, "#3B82F6", ACCENT])
    else:
        fig_tier = px.bar(tier_agg, x="city_tier", y="avg_price_per_sqft", color="city_tier",
                            title="Avg price/sqft by city tier",
                            color_discrete_sequence=[PRIMARY, "#3B82F6", ACCENT])
    st.plotly_chart(fig_tier, use_container_width=True)

    st.markdown("#### Price/rent vs. metro distance")
    plot_source = rentable if is_rent else props
    y_col = "rent_per_sqft" if is_rent else "price_per_sqft"
    fig2 = px.scatter(
        plot_source.sample(min(800, len(plot_source)), random_state=1),
        x="metro_distance_km", y=y_col, color="tier",
        size="num_amenities", hover_data=["locality", "builder", "property_type"],
        title=f"{target_label} vs. metro distance (bubble size = amenity count)",
        color_discrete_sequence=[PRIMARY, "#3B82F6", "#93C5FD", "#D97706"],
        render_mode="svg",
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.markdown(f"#### Estimate {target_label} for a hypothetical property")
    st.caption("Adjust the inputs and see the model's prediction.")

    colA, colB, colC = st.columns(3)
    with colA:
        city_tier_sel = st.selectbox("City tier", sorted(props["city_tier"].unique()), key="pt_tier")
        city_pool = props[props["city_tier"] == city_tier_sel]
        city = st.selectbox("City", sorted(city_pool["city"].unique()), key="pt_city")
        locality_pool = city_pool[city_pool["city"] == city]
        locality = st.selectbox("Locality", sorted(locality_pool["locality"].unique()), key="pt_locality")
        tier = locality_pool[locality_pool["locality"] == locality]["tier"].iloc[0]
        st.caption(f"Locality quality tier: **{tier}**")
    with colB:
        property_type = st.selectbox("Property type", sorted(props["property_type"].unique()), key="pt_type")
        bhk = st.selectbox("BHK", [0, 1, 2, 3, 4, 5], index=2, key="pt_bhk")
        carpet_area = st.slider("Carpet area (sqft)", 250, 4500, 850, key="pt_area")
        age_years = st.slider("Property age (years)", 0, 25, 3, key="pt_age")
    with colC:
        builder = st.selectbox("Builder", sorted(props["builder"].unique()), key="pt_builder")
        builder_row = props[props["builder"] == builder].iloc[0]
        builder_score = builder_row["builder_score"]
        builder_tier = builder_row["builder_tier"]
        num_amenities = st.slider("Number of amenities", 0, 12, 6, key="pt_amenities")
        rera_registered = st.checkbox("RERA registered", value=True, key="pt_rera")

    row = locality_pool[locality_pool["locality"] == locality].iloc[0]

    input_dict = {
        "city": city, "city_tier": city_tier_sel, "locality": locality, "tier": tier,
        "property_type": property_type, "builder": builder, "builder_tier": builder_tier,
        "metro_distance_km": row["metro_distance_km"], "bhk": bhk, "carpet_area_sqft": carpet_area,
        "age_years": age_years, "builder_score": builder_score, "rera_registered": int(rera_registered),
        "num_amenities": num_amenities, "circle_rate_per_sqft": row["circle_rate_per_sqft"],
        "vastu_compliant": 1, "gated_community": int(num_amenities >= 6),
    }

    feat_row = {}
    for col in ["city", "city_tier", "locality", "tier", "property_type", "builder", "builder_tier"]:
        le = encoders[col]
        val = input_dict[col]
        feat_row[col + "_enc"] = le.transform([val])[0] if val in le.classes_ else 0
    for col in ["metro_distance_km", "bhk", "carpet_area_sqft", "age_years", "builder_score",
                 "rera_registered", "num_amenities", "circle_rate_per_sqft", "vastu_compliant", "gated_community"]:
        feat_row[col] = input_dict[col]

    X_input = pd.DataFrame([feat_row])[feature_cols]
    pred_per_sqft = model.predict(X_input)[0]
    total_est = pred_per_sqft * carpet_area

    st.divider()
    m1, m2 = st.columns(2)
    with m1:
        st.metric(f"Predicted {target_label}", f"₹{pred_per_sqft:,.0f}")
    with m2:
        st.metric("Estimated total " + ("monthly rent" if is_rent else "price"), f"₹{total_est:,.0f}")

    st.caption(
        f"Circle rate anchor for {locality}: ₹{row['circle_rate_per_sqft']:,.0f}/sqft. "
        f"Model premium over circle rate: {(pred_per_sqft/row['circle_rate_per_sqft'] - 1):.1%}"
    )
