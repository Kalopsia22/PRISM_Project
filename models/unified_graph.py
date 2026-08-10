"""
PRISM Unified Property Graph
================================
Joins the outputs of all four modules into a single property-level table,
geo-indexed by pincode/micro-market — the backbone that lets PRISM produce
one Unified Property Score instead of four disconnected model outputs.

Sources joined:
  - properties.csv         (base property attributes, price model input)
  - listings.csv           (fraud model input — one primary listing per property)
  - locality_profile.csv   (yield/appreciation, rental yield recommender)
  - construction_status    (simulated per-property build stage vs promised stage)
"""

import numpy as np
import pandas as pd
import joblib
import os

RNG = np.random.default_rng(99)
BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")


def simulate_construction_status(properties: pd.DataFrame) -> pd.DataFrame:
    """Every property gets a simulated build status. Older properties are
    almost certainly ready-to-move; newer ones may still be under construction
    with a promised-vs-actual stage gap (this is what the Construction
    Monitoring module and Lender API check against)."""
    rows = []
    for _, p in properties.iterrows():
        age = p["age_years"]
        if age >= 3:
            actual_stage, promised_stage = 5, 5
            under_construction = False
        else:
            under_construction = True
            # newer properties: simulate a promised timeline vs actual progress,
            # with most on-track but a meaningful minority materially behind
            promised_stage = RNG.integers(3, 6)
            lag_roll = RNG.random()
            if lag_roll < 0.65:
                actual_stage = promised_stage  # on schedule
            elif lag_roll < 0.88:
                actual_stage = max(1, promised_stage - 1)  # slightly behind
            else:
                actual_stage = max(1, promised_stage - RNG.integers(2, 4))  # materially behind
        rows.append({
            "property_id": p["property_id"], "under_construction": under_construction,
            "actual_stage": int(actual_stage), "promised_stage": int(promised_stage),
            "schedule_delta": int(actual_stage - promised_stage),
        })
    return pd.DataFrame(rows)


def build_unified_graph() -> pd.DataFrame:
    properties = pd.read_csv(os.path.join(DATA_DIR, "properties.csv"))
    listings = pd.read_csv(os.path.join(DATA_DIR, "listings.csv"))
    locality = pd.read_csv(os.path.join(DATA_DIR, "locality_profile.csv"))

    # one primary (lowest listing_id = earliest posted) listing per property
    primary_listings = listings.sort_values("listing_id").groupby("property_id").first().reset_index()

    construction = simulate_construction_status(properties)

    graph = properties.merge(
        primary_listings[["property_id", "listing_id", "listed_price", "fair_value_est",
                            "price_deviation_pct", "broker_phone", "broker_listing_count",
                            "description_reuse_count", "days_on_market", "num_images",
                            "is_fraud", "fraud_type"]],
        on="property_id", how="left",
    )
    graph = graph.merge(
        locality[["city", "locality", "pincode", "avg_rental_yield_pct", "avg_yoy_appreciation_pct",
                   "stability_score", "yield_norm", "appreciation_norm", "stability_norm"]],
        on=["city", "locality", "pincode"], how="left",
    )
    graph = graph.merge(construction, on="property_id", how="left")

    return graph


def score_fraud_layer(graph: pd.DataFrame) -> pd.DataFrame:
    """Attach fraud-model probability to every property via its primary listing."""
    from models.fraud_model import FEATURES

    model = joblib.load(os.path.join(MODELS_DIR, "fraud_xgb.joblib"))
    feat_df = graph.rename(columns={"listed_price": "listed_price"}).copy()
    feat_df["listed_price"] = feat_df["listed_price"].fillna(feat_df["fair_value_est"])
    feat_df["price_deviation_pct"] = feat_df["price_deviation_pct"].fillna(0)
    feat_df["abs_price_deviation_pct"] = feat_df["price_deviation_pct"].abs()
    feat_df["price_per_sqft_listed"] = feat_df["listed_price"] / feat_df["carpet_area_sqft"]
    feat_df["area_per_bhk"] = feat_df["carpet_area_sqft"] / feat_df["bhk"]
    feat_df["days_on_market"] = feat_df["days_on_market"].fillna(0)
    feat_df["num_images"] = feat_df["num_images"].fillna(5)
    feat_df["broker_listing_count"] = feat_df["broker_listing_count"].fillna(1)
    feat_df["description_reuse_count"] = feat_df["description_reuse_count"].fillna(1)

    X = feat_df[FEATURES]
    graph["fraud_probability"] = model.predict_proba(X)[:, 1]
    return graph


def compute_unified_score(graph: pd.DataFrame,
                            weights=None) -> pd.DataFrame:
    """
    Combine four dimensions into one PRISM Score, mapped to a bureau-style
    300-900 band (same pattern as the AA financial health scoring project):

      1. Price Fairness  — how close the listed price sits to model-fair value
      2. Trust            — inverse of fraud probability, boosted by RERA/builder
      3. Delivery Risk    — construction schedule adherence
      4. Investment Value — locality yield + appreciation + stability
    """
    if weights is None:
        weights = {"price_fairness": 0.30, "trust": 0.30, "delivery": 0.15, "investment": 0.25}

    g = graph.copy()

    # 1. Price fairness: penalize large absolute deviation from fair value
    g["price_fairness_score"] = 1 - np.clip(g["price_deviation_pct"].fillna(0).abs() / 40, 0, 1)

    # 2. Trust: inverse fraud probability, small boost for RERA + builder reputation
    g["trust_score"] = (
        (1 - g["fraud_probability"].fillna(0.05)) * 0.75
        + g["rera_registered"] * 0.15
        + g["builder_score"] * 0.10
    )
    g["trust_score"] = g["trust_score"].clip(0, 1)

    # 3. Delivery risk: ready-to-move = perfect score; under construction scored
    #    on schedule adherence
    g["delivery_score"] = np.where(
        g["under_construction"] == False, 1.0,
        np.clip(1 + g["schedule_delta"].fillna(0) * 0.25, 0, 1),
    )

    # 4. Investment value: blended locality yield/appreciation/stability (already normalized)
    g["investment_score"] = (
        g["yield_norm"].fillna(0.5) * 0.4
        + g["appreciation_norm"].fillna(0.5) * 0.4
        + g["stability_norm"].fillna(0.5) * 0.2
    )

    g["prism_score_raw"] = (
        g["price_fairness_score"] * weights["price_fairness"]
        + g["trust_score"] * weights["trust"]
        + g["delivery_score"] * weights["delivery"]
        + g["investment_score"] * weights["investment"]
    )

    # map 0-1 raw score to 300-900 bureau-style band
    g["prism_score"] = (300 + g["prism_score_raw"] * 600).round().astype(int)

    def band(score):
        if score >= 750: return "Excellent"
        if score >= 650: return "Good"
        if score >= 550: return "Fair"
        return "Needs Review"
    g["prism_band"] = g["prism_score"].apply(band)

    return g


if __name__ == "__main__":
    import sys
    sys.path.append(BASE)

    graph = build_unified_graph()
    graph = score_fraud_layer(graph)
    scored = compute_unified_score(graph)

    scored.to_csv(os.path.join(DATA_DIR, "unified_property_graph.csv"), index=False)

    print(f"Unified graph built: {len(scored)} properties")
    print(f"\nPRISM Score distribution:\n{scored['prism_band'].value_counts()}")
    print(f"\nScore stats: mean={scored['prism_score'].mean():.0f}, "
          f"min={scored['prism_score'].min()}, max={scored['prism_score'].max()}")
    print(f"\nSample:")
    print(scored[["property_id", "city", "locality", "prism_score", "prism_band",
                   "price_fairness_score", "trust_score", "delivery_score", "investment_score"]].head(8))
