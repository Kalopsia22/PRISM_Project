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


def evaluate_collaborative_filter(interaction_matrix: np.ndarray, test_frac=0.2, n_components=8, seed=7):
    """The rule-based recommendation_score is a deterministic weighted
    formula, not a trained model — there's no genuine held-out ground truth
    to score it against without fabricating one (using its own yield/
    appreciation inputs as "ground truth" would just be checking the
    formula against itself). The collaborative-filtering layer is
    different: it's an actual matrix-factorization model, so it can be
    evaluated the standard recsys way — mask a held-out slice of the
    simulated interaction matrix, fit SVD on what's left, and see whether
    the reconstruction correctly classifies "did this investor show
    interest in this locality" on the interactions it never saw. This is
    the only genuinely non-circular classification evaluation available in
    this module.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    rng = np.random.default_rng(seed)
    nonzero_idx = np.argwhere(interaction_matrix > 0)
    rng.shuffle(nonzero_idx)
    n_test = int(len(nonzero_idx) * test_frac)
    test_idx, train_idx = nonzero_idx[:n_test], nonzero_idx[n_test:]

    masked = interaction_matrix.copy()
    for i, j in test_idx:
        masked[i, j] = 0

    reconstructed = collaborative_filter_scores(masked, n_components=n_components)

    threshold = np.median(interaction_matrix[interaction_matrix > 0])
    y_true, y_score = [], []
    for i, j in test_idx:
        y_true.append(1 if interaction_matrix[i, j] > threshold else 0)
        y_score.append(reconstructed[i, j])
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score > np.median(y_score)).astype(int)

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "n": len(y_true),
    }


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

    print("\n=== Collaborative filter held-out evaluation ===")
    cf_metrics = evaluate_collaborative_filter(matrix)
    if cf_metrics:
        print(f"accuracy={cf_metrics['accuracy']:.2%} precision={cf_metrics['precision']:.2%} "
              f"recall={cf_metrics['recall']:.2%} f1={cf_metrics['f1']:.2%} roc_auc={cf_metrics['roc_auc']:.4f} "
              f"(n={cf_metrics['n']} held-out interactions)")
        pd.DataFrame([cf_metrics]).to_csv("/home/claude/prism/data/yield_classification_metrics.csv", index=False)
    else:
        print("Degenerate split — skipping (try a different seed or larger n_investors)")
