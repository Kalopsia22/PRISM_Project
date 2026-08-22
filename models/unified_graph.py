"""
PRISM Unified Property Graph
================================
Joins the outputs of all modules into a single property-level table,
geo-indexed by pincode/micro-market — the backbone that lets PRISM produce
one Unified Property Score instead of disconnected model outputs.

Sources joined:
  - properties.csv         (base property attributes, price/rent model input)
  - listings.csv           (fraud model input — one primary listing per property)
  - locality_profile.csv   (yield/appreciation, rental yield recommender)
  - transactions.csv       (AML/transaction-structuring risk — replaces the
                             old construction-monitoring dimension entirely)
"""

import numpy as np
import pandas as pd
import joblib
import os

RNG = np.random.default_rng(99)
BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")


def build_unified_graph() -> pd.DataFrame:
    properties = pd.read_csv(os.path.join(DATA_DIR, "properties.csv"))
    listings = pd.read_csv(os.path.join(DATA_DIR, "listings.csv"))
    locality = pd.read_csv(os.path.join(DATA_DIR, "locality_profile.csv"))
    transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))

    # one primary sale listing and one primary rent listing per property
    sale_listings = listings[listings["listing_type"] == "sale"].sort_values("listing_id").groupby("property_id").first().reset_index()
    rent_listings = listings[listings["listing_type"] == "rent"].sort_values("listing_id").groupby("property_id").first().reset_index()

    graph = properties.merge(
        sale_listings[["property_id", "listing_id", "ask_value", "fair_value", "price_deviation_pct",
                        "broker_phone", "broker_listing_count", "description_reuse_count",
                        "days_on_market", "num_images", "is_fraud", "fraud_type"]]
        .rename(columns={"listing_id": "sale_listing_id", "ask_value": "sale_ask_value",
                          "fair_value": "sale_fair_value", "price_deviation_pct": "sale_price_deviation_pct",
                          "is_fraud": "sale_is_fraud", "fraud_type": "sale_fraud_type"}),
        on="property_id", how="left",
    )
    graph = graph.merge(
        rent_listings[["property_id", "listing_id", "ask_value", "fair_value", "price_deviation_pct",
                        "broker_phone", "broker_listing_count", "description_reuse_count",
                        "days_on_market", "num_images", "is_fraud", "fraud_type"]]
        .rename(columns={"listing_id": "rent_listing_id", "ask_value": "rent_ask_value",
                          "fair_value": "rent_fair_value", "price_deviation_pct": "rent_price_deviation_pct",
                          "broker_phone": "rent_broker_phone", "broker_listing_count": "rent_broker_listing_count",
                          "description_reuse_count": "rent_description_reuse_count",
                          "days_on_market": "rent_days_on_market", "num_images": "rent_num_images",
                          "is_fraud": "rent_is_fraud", "fraud_type": "rent_fraud_type"}),
        on="property_id", how="left",
    )
    graph = graph.merge(
        locality[["city", "locality", "pincode", "avg_rental_yield_pct", "avg_yoy_appreciation_pct",
                   "stability_score", "yield_norm", "appreciation_norm", "stability_norm"]],
        on=["city", "locality", "pincode"], how="left",
    )
    graph = graph.merge(
        transactions[["property_id", "buyer_name", "buyer_type", "buyer_pan_hash", "seller_name",
                       "declared_value", "cash_component_pct", "holding_period_days", "undervaluation_pct",
                       "is_rapid_flip", "is_shell_pattern", "buyer_txn_count", "financing_type",
                       "aml_flag", "pattern_types", "is_ring_txn"]],
        on="property_id", how="left",
    )

    return graph


def _score_fraud_for(graph: pd.DataFrame, prefix: str) -> pd.Series:
    """Run the fraud model against either the sale-side or rent-side listing
    columns on the graph (prefix = 'sale' or 'rent')."""
    from models.fraud_model import FEATURES

    model = joblib.load(os.path.join(MODELS_DIR, "fraud_xgb.joblib"))
    feat_df = pd.DataFrame(index=graph.index)
    feat_df["ask_value"] = graph[f"{prefix}_ask_value"].fillna(graph[f"{prefix}_fair_value"])
    feat_df["carpet_area_sqft"] = graph["carpet_area_sqft"]
    feat_df["bhk"] = graph["bhk"].replace(0, 1)
    feat_df["ask_per_sqft"] = feat_df["ask_value"] / feat_df["carpet_area_sqft"]
    feat_df["area_per_bhk"] = feat_df["carpet_area_sqft"] / feat_df["bhk"]
    dev_col = f"{prefix}_price_deviation_pct"
    feat_df["abs_price_deviation_pct"] = graph[dev_col].fillna(0).abs()
    days_col = f"{prefix}_days_on_market" if prefix == "rent" else "days_on_market"
    feat_df["days_on_market"] = graph[days_col].fillna(0)
    imgs_col = f"{prefix}_num_images" if prefix == "rent" else "num_images"
    feat_df["num_images"] = graph[imgs_col].fillna(5)
    feat_df["rera_registered"] = graph["rera_registered"]
    broker_col = f"{prefix}_broker_listing_count" if prefix == "rent" else "broker_listing_count"
    feat_df["broker_listing_count"] = graph[broker_col].fillna(1)
    desc_col = f"{prefix}_description_reuse_count" if prefix == "rent" else "description_reuse_count"
    feat_df["description_reuse_count"] = graph[desc_col].fillna(1)
    feat_df["is_rent"] = 1 if prefix == "rent" else 0

    X = feat_df[FEATURES]
    return model.predict_proba(X)[:, 1]


def score_fraud_layer(graph: pd.DataFrame) -> pd.DataFrame:
    graph = graph.copy()
    graph["sale_fraud_probability"] = _score_fraud_for(graph, "sale")
    has_rent = graph["rent_ask_value"].notna()
    graph["rent_fraud_probability"] = np.nan
    if has_rent.any():
        graph.loc[has_rent, "rent_fraud_probability"] = _score_fraud_for(graph[has_rent], "rent")
    # primary fraud_probability used by default (sale-side) views
    graph["fraud_probability"] = graph["sale_fraud_probability"]
    graph["fraud_type"] = graph["sale_fraud_type"]
    return graph


def score_aml_layer(graph: pd.DataFrame) -> pd.DataFrame:
    """Attach the AML classifier's transaction-risk probability to every
    property via its associated transaction record."""
    from models.aml_risk import FEATURES as AML_FEATURES

    graph = graph.copy()
    bundle = joblib.load(os.path.join(MODELS_DIR, "aml_xgb.joblib"))
    model, encoders = bundle["model"], bundle["encoders"]

    feat_df = pd.DataFrame(index=graph.index)
    feat_df["undervaluation_pct"] = graph["undervaluation_pct"].fillna(0)
    feat_df["cash_component_pct"] = graph["cash_component_pct"].fillna(3)
    feat_df["holding_period_days"] = graph["holding_period_days"].fillna(1000)
    feat_df["buyer_txn_count"] = graph["buyer_txn_count"].fillna(1)
    feat_df["is_shell_pattern"] = graph["is_shell_pattern"].fillna(0)
    feat_df["is_rapid_flip"] = graph["is_rapid_flip"].fillna(0)
    for col in ["buyer_type", "financing_type"]:
        le = encoders[col]
        feat_df[col + "_enc"] = graph[col].astype(str).map(
            lambda v: le.transform([v])[0] if v in le.classes_ else -1
        )

    graph["aml_risk_probability"] = model.predict_proba(feat_df[AML_FEATURES])[:, 1]
    return graph


def compute_unified_score(graph: pd.DataFrame, weights=None) -> pd.DataFrame:
    """
    Combine four dimensions into one PRISM Score, mapped to a bureau-style
    300-900 band (same pattern as the AA financial health scoring project):

      1. Price Fairness   — how close the ask sits to model-fair value
      2. Trust             — inverse of fraud probability, small RERA/builder boost
      3. Compliance Risk   — inverse of AML/transaction-structuring risk
      4. Investment Value  — locality yield + appreciation + stability

    Calibrated deliberately strict: price fairness zeroes out past a 20%
    deviation (not 40%), and trust leans mostly on the fraud-model output
    rather than flat registration/builder boosts — so the score actually
    discriminates between listings instead of clustering everyone at the top.
    """
    if weights is None:
        weights = {"price_fairness": 0.30, "trust": 0.30, "compliance": 0.15, "investment": 0.25}

    g = graph.copy()

    g["price_fairness_score"] = 1 - np.clip(g["sale_price_deviation_pct"].fillna(15).abs() / 20, 0, 1)

    g["trust_score"] = (
        (1 - g["fraud_probability"].fillna(0.10)) * 0.85
        + g["rera_registered"] * 0.08
        + g["builder_score"] * 0.07
    )
    g["trust_score"] = g["trust_score"].clip(0, 1)

    g["compliance_score"] = (1 - g["aml_risk_probability"].fillna(0.05)).clip(0, 1)

    g["investment_score"] = (
        g["yield_norm"].fillna(0.5) * 0.4
        + g["appreciation_norm"].fillna(0.5) * 0.4
        + g["stability_norm"].fillna(0.5) * 0.2
    )

    g["prism_score_raw"] = (
        g["price_fairness_score"] * weights["price_fairness"]
        + g["trust_score"] * weights["trust"]
        + g["compliance_score"] * weights["compliance"]
        + g["investment_score"] * weights["investment"]
    )

    g["prism_score"] = (300 + g["prism_score_raw"] * 600).round().astype(int)

    def band(score):
        if score >= 750: return "Excellent"
        if score >= 650: return "Good"
        if score >= 550: return "Fair"
        return "Needs Review"
    g["prism_band"] = g["prism_score"].apply(band)

    # ---- Rent-side score (same formula, rent listing's deviation/fraud;
    # compliance risk is a property/transaction-level attribute so it's
    # shared across both sale and rent views of the same property) ----
    has_rent = g["rent_ask_value"].notna()
    g["rent_price_fairness_score"] = np.nan
    g["rent_trust_score"] = np.nan
    g["rent_prism_score"] = np.nan
    g["rent_prism_band"] = None

    rpf = 1 - np.clip(g.loc[has_rent, "rent_price_deviation_pct"].fillna(15).abs() / 20, 0, 1)
    rt = (
        (1 - g.loc[has_rent, "rent_fraud_probability"].fillna(0.10)) * 0.85
        + g.loc[has_rent, "rera_registered"] * 0.08
        + g.loc[has_rent, "builder_score"] * 0.07
    ).clip(0, 1)
    raw = (
        rpf * weights["price_fairness"]
        + rt * weights["trust"]
        + g.loc[has_rent, "compliance_score"] * weights["compliance"]
        + g.loc[has_rent, "investment_score"] * weights["investment"]
    )
    rent_score = (300 + raw * 600).round().astype(int)

    g.loc[has_rent, "rent_price_fairness_score"] = rpf
    g.loc[has_rent, "rent_trust_score"] = rt
    g.loc[has_rent, "rent_prism_score"] = rent_score
    g.loc[has_rent, "rent_prism_band"] = rent_score.apply(band)

    return g


if __name__ == "__main__":
    import sys
    sys.path.append(BASE)

    graph = build_unified_graph()
    graph = score_fraud_layer(graph)
    graph = score_aml_layer(graph)
    scored = compute_unified_score(graph)

    scored.to_csv(os.path.join(DATA_DIR, "unified_property_graph.csv"), index=False)

    print(f"Unified graph built: {len(scored)} properties")
    print(f"\nPRISM Score distribution:\n{scored['prism_band'].value_counts()}")
    print(f"\nScore stats: mean={scored['prism_score'].mean():.0f}, "
          f"min={scored['prism_score'].min()}, max={scored['prism_score'].max()}")
    print(f"\nHas rent listing: {scored['rent_ask_value'].notna().sum()} / {len(scored)}")
    print(f"Avg compliance score: {scored['compliance_score'].mean():.3f}")
