"""
PRISM Module 2 — Rental Yield & Investment Recommendation System
=====================================================================
Combines:
  1. A locality-level yield/appreciation profile (aggregated from the
     synthetic property + trend data)
  2. A risk-profile scoring layer that maps investor preferences to a
     weighted score across yield %, appreciation %, and volatility
  3. Matrix-factorization collaborative filtering over a simulated
     investor-interaction matrix, so recommendations aren't purely
     rule-based — they also pick up "investors who liked X also liked Y"
     signal, the way a real recommender platform would.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

RNG = np.random.default_rng(11)

RISK_PROFILES = {
    "risk_averse": {"yield_weight": 0.6, "appreciation_weight": 0.15, "stability_weight": 0.25},
    "balanced": {"yield_weight": 0.35, "appreciation_weight": 0.35, "stability_weight": 0.30},
    "aggressive": {"yield_weight": 0.15, "appreciation_weight": 0.65, "stability_weight": 0.20},
    "nri_remote": {"yield_weight": 0.45, "appreciation_weight": 0.30, "stability_weight": 0.25},
}


def build_locality_profile(properties: pd.DataFrame, trend: pd.DataFrame) -> pd.DataFrame:
    agg = properties.groupby(["city", "locality", "pincode", "tier"]).agg(
        avg_price_per_sqft=("price_per_sqft", "mean"),
        avg_rental_yield_pct=("rental_yield_pct", "mean"),
        yield_std=("rental_yield_pct", "std"),
        avg_builder_score=("builder_score", "mean"),
        rera_compliance_rate=("rera_registered", "mean"),
        n_properties=("property_id", "count"),
    ).reset_index()

    trend_agg = trend.groupby(["city", "locality", "pincode"]).agg(
        avg_yoy_appreciation_pct=("yoy_appreciation_pct", "mean"),
        appreciation_volatility=("yoy_appreciation_pct", "std"),
        cumulative_5yr_index=("cumulative_index", "last"),
    ).reset_index()

    profile = agg.merge(trend_agg, on=["city", "locality", "pincode"])

    # stability score: inverse of combined volatility (yield + appreciation)
    combined_vol = profile["yield_std"].fillna(0) + profile["appreciation_volatility"].fillna(0)
    profile["stability_score"] = 1 / (1 + combined_vol)

    # normalize components to 0-1 for scoring
    for col, new_col in [
        ("avg_rental_yield_pct", "yield_norm"),
        ("avg_yoy_appreciation_pct", "appreciation_norm"),
        ("stability_score", "stability_norm"),
    ]:
        profile[new_col] = (profile[col] - profile[col].min()) / (profile[col].max() - profile[col].min())

    return profile


def score_for_profile(locality_profile: pd.DataFrame, risk_profile: str) -> pd.DataFrame:
    weights = RISK_PROFILES[risk_profile]
    df = locality_profile.copy()
    df["recommendation_score"] = (
        df["yield_norm"] * weights["yield_weight"]
        + df["appreciation_norm"] * weights["appreciation_weight"]
        + df["stability_norm"] * weights["stability_weight"]
    )
    return df.sort_values("recommendation_score", ascending=False)


def simulate_investor_interactions(locality_profile: pd.DataFrame, n_investors=300):
    """Simulate an investor x locality 'interest score' matrix, biased by
    each simulated investor's assigned risk profile, so we have something
    for collaborative filtering to factorize."""
    n_localities = len(locality_profile)
    profiles = list(RISK_PROFILES.keys())
    investor_profiles = RNG.choice(profiles, size=n_investors)

    matrix = np.zeros((n_investors, n_localities))
    for i, profile in enumerate(investor_profiles):
        scored = score_for_profile(locality_profile, profile)
        scored = scored.set_index(scored.index)
        base_scores = score_for_profile(locality_profile, profile)["recommendation_score"].values
        noise = RNG.normal(0, 0.08, size=n_localities)
        interest = np.clip(base_scores + noise, 0, 1)
        # simulate sparsity: investors only "interact" with a subset
        mask = RNG.random(n_localities) < 0.4
        matrix[i, mask] = interest[mask]

    return matrix, investor_profiles


def collaborative_filter_scores(interaction_matrix: np.ndarray, n_components=8):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent = svd.fit_transform(interaction_matrix)
    reconstructed = latent @ svd.components_
    return reconstructed


def recommend_localities(locality_profile: pd.DataFrame, risk_profile: str,
                           budget_max: float = None, top_n=5,
                           cf_boost: np.ndarray = None, investor_idx: int = None):
    scored = score_for_profile(locality_profile, risk_profile)
    if budget_max is not None:
        scored = scored[scored["avg_price_per_sqft"] * 1000 <= budget_max]  # rough per-unit sanity filter

    if cf_boost is not None and investor_idx is not None:
        cf_row = cf_boost[investor_idx]
        scored = scored.copy()
        scored["cf_score"] = cf_row[: len(scored)]
        scored["final_score"] = 0.7 * scored["recommendation_score"] + 0.3 * scored["cf_score"]
        scored = scored.sort_values("final_score", ascending=False)

    return scored.head(top_n)


if __name__ == "__main__":
    props = pd.read_csv("/home/claude/prism/data/properties.csv")
    trend = pd.read_csv("/home/claude/prism/data/appreciation_trend.csv")

    profile = build_locality_profile(props, trend)
    profile.to_csv("/home/claude/prism/data/locality_profile.csv", index=False)

    print("=== Top 5 recommendations per risk profile ===")
    for rp in RISK_PROFILES:
        top = recommend_localities(profile, rp, top_n=5)
        print(f"\n-- {rp} --")
        print(top[["city", "locality", "avg_rental_yield_pct", "avg_yoy_appreciation_pct", "recommendation_score"]])

    matrix, investor_profiles = simulate_investor_interactions(profile, n_investors=300)
    cf_scores = collaborative_filter_scores(matrix)
    print(f"\nCollaborative filtering matrix shape: {cf_scores.shape}")
