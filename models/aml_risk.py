"""
PRISM Module — AML & Transaction Structuring Risk Engine
============================================================
Three complementary layers, mirroring how a real bank AML/compliance team
actually works — no single technique catches everything:

  1. Supervised classifier (XGBoost) on engineered transaction features,
     trained against known typologies (undervaluation, rapid flips, high
     cash, shell-entity buyers, structuring).
  2. Unsupervised anomaly detection (Isolation Forest) run independently of
     the label — catches transactions that look "off" even if they don't
     match a known labeled pattern, the way real AML systems flag novel
     deviations alongside rule-based typology matches.
  3. Graph-based ring detection (networkx) over the buyer→seller transaction
     network — finds strongly-connected clusters of entities repeatedly
     trading among themselves, a pattern no per-transaction feature can see
     in isolation since it only emerges from the network structure.

Regulatory grounding: PMLA 2002 obligates reporting entities to flag
suspicious transactions to FIU-IND; the Income Tax Act (Sections 269SS/
269ST) effectively caps cash consideration for property deals; RBI/NHB
KYC master directions require beneficial-ownership verification for
non-individual buyers — the typologies here map to checks a bank's AML
function would actually run before financing or clearing a property deal.
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

FEATURES = [
    "undervaluation_pct", "cash_component_pct", "holding_period_days",
    "buyer_txn_count", "is_shell_pattern", "is_rapid_flip", "buyer_type_enc",
    "financing_type_enc",
]


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    encoders = {}
    for col in ["buyer_type", "financing_type"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def train_aml_classifier(df: pd.DataFrame):
    df_enc, encoders = prepare_features(df)
    X = df_enc[FEATURES]
    y = df_enc["aml_flag"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight, eval_metric="auc", random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    return {"model": model, "encoders": encoders, "auc": auc, "confusion_matrix": cm,
             "feature_importance": importances, "X_test": X_test, "y_test": y_test}


def run_anomaly_detection(df: pd.DataFrame, encoders: dict, contamination=0.08):
    """Isolation Forest, run independently of the aml_flag label — flags
    transactions that sit in unusual regions of feature space even when no
    single labeled typology explains why."""
    df_enc = df.copy()
    for col in ["buyer_type", "financing_type"]:
        le = encoders[col]
        df_enc[col + "_enc"] = df_enc[col].astype(str).map(
            lambda v: le.transform([v])[0] if v in le.classes_ else -1
        )
    X = df_enc[FEATURES].fillna(0)
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    iso.fit(X)
    anomaly_score = -iso.score_samples(X)  # higher = more anomalous
    return anomaly_score


def detect_transaction_rings(df: pd.DataFrame, min_cycle_size=3):
    """Build a directed buyer→seller graph from transactions and find
    strongly-connected components of size >= min_cycle_size — entities
    trading in a closed loop, the structural signature of layering that no
    single transaction's features can reveal on its own."""
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row["buyer_pan_hash"], row["seller_pan_hash"], transaction_id=row["transaction_id"])

    sccs = [c for c in nx.strongly_connected_components(G) if len(c) >= min_cycle_size]
    ring_entities = set().union(*sccs) if sccs else set()

    df = df.copy()
    df["in_detected_ring"] = (
        df["buyer_pan_hash"].isin(ring_entities) | df["seller_pan_hash"].isin(ring_entities)
    ).astype(int)

    return df, G, sccs


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/prism/data/transactions.csv")

    print("=== Training AML classifier ===")
    results = train_aml_classifier(df)
    print(f"AUC: {results['auc']:.4f}")
    print(f"\nFeature importance:\n{results['feature_importance']}")
    print(f"\nConfusion matrix:\n{results['confusion_matrix']}")

    joblib.dump({"model": results["model"], "encoders": results["encoders"]},
                "/home/claude/prism/models/aml_xgb.joblib")

    print("\n=== Anomaly detection layer ===")
    anomaly_scores = run_anomaly_detection(df, results["encoders"])
    df["anomaly_score"] = anomaly_scores
    print(f"Anomaly score range: {anomaly_scores.min():.3f} to {anomaly_scores.max():.3f}")
    print(f"Correlation with labeled aml_flag: {np.corrcoef(anomaly_scores, df['aml_flag'])[0,1]:.3f}")

    print("\n=== Ring detection layer ===")
    df_ring, G, sccs = detect_transaction_rings(df)
    print(f"Entities in graph: {G.number_of_nodes()}, transactions (edges): {G.number_of_edges()}")
    print(f"Detected {len(sccs)} rings, sizes: {[len(s) for s in sccs]}")
    print(f"Transactions touching a detected ring: {df_ring['in_detected_ring'].sum()}")

    df_ring.to_csv("/home/claude/prism/data/transactions.csv", index=False)
