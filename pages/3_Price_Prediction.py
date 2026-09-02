import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.styling import inject_css, page_header, section_title, insight_grid, model_metrics_bar, PRIMARY, ACCENT
from utils.charts import gauge_chart, treemap_chart, property_icon_svg, PRIMARY_LIGHT

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
model_metrics = pd.read_csv(os.path.join(BASE, "data", "price_model_metrics.csv")).iloc[0]

mode = st.radio("Predict for:", ["Sale (Possession)", "Rental"], horizontal=True)
is_rent = mode == "Rental"
bundle = rent_bundle if is_rent else price_bundle
model, encoders, feature_cols = bundle["model"], bundle["encoders"], bundle["feature_cols"]
target_label = "rent/sqft" if is_rent else "price/sqft"
shap_file = "rent_shap_importance.csv" if is_rent else "price_shap_importance.csv"

if is_rent:
    rentable = props[props["rentable"] == 1].copy()
    rentable["rent_per_sqft"] = rentable["monthly_rent_est"] / rentable["carpet_area_sqft"]

clf_path = os.path.join(BASE, "data", "price_classification_metrics.csv")
if os.path.exists(clf_path):
    clf_df = pd.read_csv(clf_path)
    target_key = "rent" if is_rent else "sale"
    clf_row = clf_df[clf_df["target"] == target_key]
    if len(clf_row):
        clf_row = clf_row.iloc[0]
        st.markdown("##### Classification framing: is this priced above or below the going rate for its market?")
        st.caption(
            "Accuracy/precision/recall/F1/ROC-AUC are classification metrics and don't natively apply to a "
            "continuous price prediction — so here they evaluate a derived, well-precedented AVM framing "
            "(the same above/below-market call real automated-valuation models are stress-tested on): "
            + ("does the model correctly call whether a **real** listing is priced above or below its city's "
               "real median price/sqft — tested on 13,609 real scraped listings the model never trained on."
               if not is_rent else
               "does the model correctly call whether a held-out **synthetic** test property's rent is above "
               "or below the median for its property type — no real external rent data exists to validate "
               "against, so this is scoped to synthetic-only, unlike the sale-side number.")
        )
        model_metrics_bar(clf_row["accuracy"], clf_row["precision"], clf_row["recall"], clf_row["f1"], clf_row["roc_auc"])

tab1, tab2, tab3 = st.tabs(["📊 Model Performance & Insights", "🧮 Try a Prediction", "✅ Real-World Validation"])

with tab1:
    section_title("🧠", f"Why hyperlocal beats a generic AVM ({'rent' if is_rent else 'price'} model)")
    st.markdown(
        "Prices vary street-to-street within the same city, and now also across 18 cities "
        "spanning Tier 1/2/3 India — a generic model trained at city level misses this. PRISM "
        "clusters by pincode/micro-market and property type so the model learns locality- and "
        "category-specific dynamics (a Villa in Gurgaon and a Studio in Dombivli don't move "
        "together)."
    )

    g1, g2, g3 = st.columns(3)
    with g1:
        st.metric("Model", "XGBoost Regressor")
    with g2:
        mape_val = model_metrics["rent_mape"] if is_rent else model_metrics["sale_mape"]
        st.metric("MAPE (test set)", f"{mape_val:.1%}")
    with g3:
        r2_val = model_metrics["rent_r2"] if is_rent else model_metrics["sale_r2"]
        st.plotly_chart(gauge_chart(r2_val * 100, "R² (fit quality)", min_val=0, max_val=100,
                                      band_edges=(50, 70, 85), height=180), use_container_width=True)
    if is_rent:
        st.caption("Rent prediction is inherently noisier than sale price — real rental markets carry more landlord-level idiosyncrasy, which shows up here too.")

    section_title("🔬", f"What drives {target_label}, according to the model")
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
            color="mean_abs_shap", color_continuous_scale=[[0, "#DBEAFE"], [1, PRIMARY]],
        )
        fig.update_layout(yaxis_title="", xaxis_title="Mean |SHAP value| (₹/sqft impact)",
                            plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        top_driver = shap_imp.sort_values("mean_abs_shap", ascending=False).iloc[0]
        source_df = rentable if is_rent else props
        val_col = "rent_per_sqft" if is_rent else "price_per_sqft"
        tier_gap = source_df.groupby("city_tier")[val_col].mean()
        tier_gap_pct = (tier_gap.max() / tier_gap.min() - 1) * 100
        insight_grid([
            ("Strongest price driver", f"<b>{top_driver['label']}</b> dominates the model's predictions — more than any locality or builder effect.", "primary"),
            ("Tier 1 vs Tier 3 gap", f"Tier 1 cities average <b>{tier_gap_pct:.0f}% higher</b> {target_label} than Tier 3 — the city-tier premium is real and quantifiable.", "warning"),
        ])

    section_title("🗺️", "Price landscape by city tier and property type")
    landscape = (rentable if is_rent else props).groupby(["city_tier", "property_type"]).agg(
        avg_val=("rent_per_sqft" if is_rent else "price_per_sqft", "mean"),
        n=("property_type", "count"),
    ).reset_index()
    st.plotly_chart(
        treemap_chart(landscape, ["city_tier", "property_type"], "n",
                       title=f"Inventory volume by city tier → property type (color = avg {target_label})",
                       color_col="avg_val", height=420),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if is_rent:
            fig_type = px.box(
                rentable, x="property_type", y="rent_per_sqft", color="city_tier",
                title="Rent/sqft distribution by property type and city tier",
                color_discrete_sequence=[PRIMARY, PRIMARY_LIGHT, ACCENT],
            )
            fig_type.update_layout(yaxis_title="₹/sqft/month", xaxis_title="")
        else:
            fig_type = px.box(
                props, x="property_type", y="price_per_sqft", color="city_tier",
                title="Price/sqft distribution by property type and city tier",
                color_discrete_sequence=[PRIMARY, PRIMARY_LIGHT, ACCENT],
            )
            fig_type.update_layout(yaxis_title="₹/sqft", xaxis_title="")
        fig_type.update_layout(xaxis_tickangle=-20, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_type, use_container_width=True)

    with c2:
        if is_rent:
            tier_agg2 = rentable.groupby("city_tier")["rent_per_sqft"].mean().reset_index()
            fig_tier = px.bar(tier_agg2, x="city_tier", y="rent_per_sqft", color="city_tier",
                                title="Avg rent/sqft by city tier",
                                color_discrete_sequence=[PRIMARY, PRIMARY_LIGHT, ACCENT])
        else:
            tier_agg = props.groupby("city_tier")["price_per_sqft"].mean().reset_index()
            fig_tier = px.bar(tier_agg, x="city_tier", y="price_per_sqft", color="city_tier",
                                title="Avg price/sqft by city tier",
                                color_discrete_sequence=[PRIMARY, PRIMARY_LIGHT, ACCENT])
        fig_tier.update_layout(plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_tier, use_container_width=True)

    section_title("📍", f"{target_label} vs. metro distance")
    plot_source = rentable if is_rent else props
    y_col = "rent_per_sqft" if is_rent else "price_per_sqft"
    fig2 = px.scatter(
        plot_source.sample(min(800, len(plot_source)), random_state=1),
        x="metro_distance_km", y=y_col, color="tier",
        size="num_amenities", hover_data=["locality", "builder", "property_type"],
        title=f"{target_label} vs. metro distance (bubble size = amenity count)",
        color_discrete_sequence=[PRIMARY, PRIMARY_LIGHT, "#93C5FD", ACCENT],
        render_mode="svg",
    )
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    section_title("🧮", f"Estimate {target_label} for a hypothetical property")
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
        st.markdown(
            f'<div style="background:#F1F5F9; border-radius:8px; padding:6px; width:fit-content; margin:-4px 0 8px 0;">{property_icon_svg(property_type, color=PRIMARY, size=36)}</div>',
            unsafe_allow_html=True,
        )
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
    premium_pct = (pred_per_sqft / row["circle_rate_per_sqft"] - 1) * 100

    st.divider()
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(f"Predicted {target_label}", f"₹{pred_per_sqft:,.0f}")
    with m2:
        st.metric("Estimated total " + ("monthly rent" if is_rent else "price"), f"₹{total_est:,.0f}")
    with m3:
        st.plotly_chart(
            gauge_chart(premium_pct, "Premium over circle rate (%)", min_val=-20, max_val=100,
                         band_edges=(0, 30, 60), height=180),
            use_container_width=True,
        )

    st.caption(f"Circle rate anchor for {locality}: ₹{row['circle_rate_per_sqft']:,.0f}/sqft.")

    similar = props[(props["locality"] == locality) & (props["property_type"] == property_type)]
    if len(similar) >= 3:
        val_col = "monthly_rent_est" if is_rent else "price_per_sqft"
        if is_rent:
            similar_vals = similar[similar["rentable"] == 1]["monthly_rent_est"] / similar[similar["rentable"] == 1]["carpet_area_sqft"]
        else:
            similar_vals = similar["price_per_sqft"]
        if len(similar_vals) >= 3:
            pctile = (similar_vals < pred_per_sqft).mean() * 100
            insight_grid([
                ("How this compares locally", f"This prediction sits at the <b>{pctile:.0f}th percentile</b> of {len(similar_vals)} similar {property_type} properties in {locality}.", "primary"),
            ])

with tab3:
    section_title("✅", "Validated against real scraped listings")
    st.markdown(
        "This isn't a synthetic-only claim — the sale price model above was run against "
        "**13,609 real, scraped Indian property listings** (uploaded dataset, 7 cities) it never "
        "saw during training. The real data only carries city, locality, property type, BHK, and "
        "area — no builder, amenity, age, or metro-distance fields, so those are filled with "
        "neutral defaults matched to the training data's own averages. That means some of the "
        "residual gap below reflects genuinely missing context, not model failure."
    )

    val_path = os.path.join(BASE, "data", "real_validation_results.csv")
    if os.path.exists(val_path):
        val = pd.read_csv(val_path)
        matched = val[val["locality_matched"]]
        unmatched = val[~val["locality_matched"]]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Listings validated", f"{len(val):,}")
        with c2:
            st.metric("Overall MAPE", f"{val['abs_pct_error'].mean():.1%}")
        with c3:
            st.metric("Matched-locality MAPE", f"{matched['abs_pct_error'].mean():.1%}", help=f"n={len(matched):,} — real listing's locality matches one of PRISM's curated micro-markets by name")
        with c4:
            st.metric("City-fallback MAPE", f"{unmatched['abs_pct_error'].mean():.1%}", help=f"n={len(unmatched):,} — locality not in PRISM's curated list, so the model uses a city-level price anchor instead")

        section_title("📈", "Predicted vs. actual price/sqft")
        sample = val.sample(min(2500, len(val)), random_state=1)
        fig_val = px.scatter(
            sample, x="actual_price_per_sqft", y="predicted_price_per_sqft", color="locality_matched",
            hover_data=["city", "locality", "property_type", "bhk"],
            title="Each point is one real listing — closer to the diagonal is a better prediction",
            color_discrete_map={True: PRIMARY, False: ACCENT},
            labels={"locality_matched": "Locality matched PRISM data"},
            render_mode="svg",
        )
        max_val = max(sample["actual_price_per_sqft"].max(), sample["predicted_price_per_sqft"].max())
        fig_val.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="#94A3B8", dash="dash"))
        fig_val.update_layout(xaxis_title="Actual price/sqft (₹)", yaxis_title="Predicted price/sqft (₹)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_val, use_container_width=True)

        section_title("🏙️", "Accuracy by city")
        city_acc = val.groupby("city").agg(
            n=("abs_pct_error", "count"), mape=("abs_pct_error", "mean"),
            matched_pct=("locality_matched", "mean"),
        ).reset_index().sort_values("mape")
        st.dataframe(
            city_acc.rename(columns={"n": "Listings", "mape": "MAPE", "matched_pct": "% Locality-Matched"})
            .style.format({"MAPE": "{:.1%}", "% Locality-Matched": "{:.0%}"})
            .bar(subset=["MAPE"], color="#FCA5A5"),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Matched-locality accuracy is consistently better than city-fallback accuracy — exactly "
            "what you'd expect, since the model has an actual price anchor for those localities "
            "instead of a city-wide average standing in for one."
        )
    else:
        st.info("Real validation results not found — run `python models/price_model.py` to generate them.")
