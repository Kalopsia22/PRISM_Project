"""
PRISM Module 1 — Hyperlocal Price Prediction Engine
=======================================================
XGBoost regressor on pincode/micro-market-level features, kept
interpretable via SHAP rather than a black-box DNN — the point of this
module is to explain *why* a price is what it is (circle rate, metro
proximity, builder reputation, amenities), not just predict a number.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import xgboost as xgb
import shap
import joblib

CATEGORICAL = ["city", "locality", "tier", "builder", "builder_tier"]
NUMERIC = [
    "metro_distance_km", "bhk", "carpet_area_sqft", "age_years",
    "builder_score", "rera_registered", "num_amenities",
    "circle_rate_per_sqft", "vastu_compliant", "gated_community",
]
TARGET = "price_per_sqft"


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le
    feature_cols = NUMERIC + [c + "_enc" for c in CATEGORICAL]
    return df, feature_cols, encoders


def train_price_model(df: pd.DataFrame):
    df_enc, feature_cols, encoders = prepare_features(df)
    X = df_enc[feature_cols]
    y = df_enc[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return {
        "model": model, "encoders": encoders, "feature_cols": feature_cols,
        "mape": mape, "r2": r2,
        "X_test": X_test, "y_test": y_test, "y_pred": y_pred,
        "shap_values": shap_values, "explainer": explainer,
    }


def predict_price(model, encoders, feature_cols, input_dict: dict):
    """Predict price/sqft for a single hypothetical property (used by the
    Streamlit 'what would this property sell for' interactive tool)."""
    row = {}
    for col in CATEGORICAL:
        val = input_dict.get(col)
        le = encoders[col]
        if val in le.classes_:
            row[col + "_enc"] = le.transform([val])[0]
        else:
            row[col + "_enc"] = 0  # unseen category fallback
    for col in NUMERIC:
        row[col] = input_dict.get(col, 0)

    X_input = pd.DataFrame([row])[feature_cols]
    pred = model.predict(X_input)[0]
    return pred


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/prism/data/properties.csv")
    results = train_price_model(df)

    print(f"MAPE: {results['mape']:.2%}")
    print(f"R2: {results['r2']:.4f}")

    joblib.dump({
        "model": results["model"], "encoders": results["encoders"],
        "feature_cols": results["feature_cols"],
    }, "/home/claude/prism/models/price_xgb.joblib")

    mean_abs_shap = np.abs(results["shap_values"]).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=results["feature_cols"]).sort_values(ascending=False)
    print("\nTop SHAP feature importances:")
    print(importance.head(8))

    importance.rename_axis("feature").reset_index(name="mean_abs_shap").to_csv(
        "/home/claude/prism/data/price_shap_importance.csv", index=False
    )
