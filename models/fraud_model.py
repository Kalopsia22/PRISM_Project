"""
PRISM Module 4 — Fraud Detection in Listings
================================================
Two-layer approach mirroring how real classifieds trust & safety teams work:

  Layer A: Supervised classifier (XGBoost) on engineered listing features
           to flag price-manipulation, fake/ghost listings, and scam-ring
           broker patterns.
  Layer B: Unsupervised duplicate/near-duplicate detection using text
           similarity (TF-IDF + cosine) on descriptions, independent of
           the label — catching duplicate listings even without training
           signal, the way a real moderation pipeline would run rule-based
           and ML-based checks side by side.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
import joblib

FEATURES = [
    "listed_price", "carpet_area_sqft", "bhk", "price_per_sqft_listed",
    "area_per_bhk", "abs_price_deviation_pct", "days_on_market", "num_images",
    "rera_registered", "broker_listing_count", "description_reuse_count",
]


def train_fraud_classifier(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc", random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    return {
        "model": model,
        "auc": auc,
        "report": report,
        "confusion_matrix": cm,
        "feature_importance": importances,
        "X_test": X_test, "y_test": y_test, "y_pred": y_pred, "y_prob": y_prob,
    }


def detect_near_duplicates(df: pd.DataFrame, threshold=0.92):
    """TF-IDF + cosine similarity duplicate detection, run independently of
    the supervised label so it also catches organic near-duplicates."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    tfidf = vectorizer.fit_transform(df["description"].fillna(""))
    sims = cosine_similarity(tfidf)

    n = len(df)
    flagged_pairs = []
    for i in range(n):
        row_sims = sims[i]
        row_sims[i] = 0  # exclude self
        matches = np.where(row_sims >= threshold)[0]
        for j in matches:
            if j > i:  # avoid double counting
                flagged_pairs.append((
                    df.iloc[i]["listing_id"], df.iloc[j]["listing_id"], round(float(sims[i, j]), 3)
                ))

    dup_df = pd.DataFrame(flagged_pairs, columns=["listing_id_a", "listing_id_b", "similarity"])
    return dup_df


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/prism/data/listings.csv")

    print("=== Training fraud classifier ===")
    results = train_fraud_classifier(df)
    print(f"AUC: {results['auc']:.4f}")
    print(f"\nFeature importance:\n{results['feature_importance']}")
    print(f"\nConfusion matrix:\n{results['confusion_matrix']}")

    joblib.dump(results["model"], "/home/claude/prism/models/fraud_xgb.joblib")

    print("\n=== Duplicate detection layer ===")
    dups = detect_near_duplicates(df, threshold=0.92)
    print(f"Flagged {len(dups)} near-duplicate listing pairs")
    print(dups.head(10))

    dups.to_csv("/home/claude/prism/data/flagged_duplicates.csv", index=False)
