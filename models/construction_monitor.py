"""
PRISM Module 3 — Construction Progress Monitoring
=====================================================
IMPORTANT — SCOPE DISCLOSURE:
Real satellite imagery (Sentinel-2, ~10m resolution) is far too coarse to
detect construction stage (floor count, structural progress). Real
construction monitoring needs sub-meter commercial imagery (Planet/Maxar)
or actual drone footage, neither of which is freely obtainable here.

This module instead demonstrates the CV *pipeline* end-to-end using
procedurally generated stage-representative imagery (5 construction
stages, computer-generated rather than claimed-real), with a classical
feature-extraction + classifier approach (no GPU/deep-learning framework
available in this environment). The production path — swapping in a
transfer-learned CNN (ResNet/EfficientNet) on real drone imagery once
available — is noted in the README rather than faked here.

Stages: 1=Excavation/Foundation, 2=Structure/RCC, 3=Brickwork/Walls,
        4=Plastering/Facade, 5=Finishing/Handover-ready
"""

import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

RNG = np.random.default_rng(21)

STAGE_NAMES = {
    1: "Excavation/Foundation", 2: "Structure/RCC", 3: "Brickwork/Walls",
    4: "Plastering/Facade", 5: "Finishing/Handover-ready",
}

# Rough color/texture palette per stage: (base_color, noise_level, edge_density)
STAGE_PALETTE = {
    1: ((120, 100, 80), 35, 0.15),   # bare earth/foundation - brownish, low edges
    2: ((150, 150, 155), 30, 0.55),  # exposed RCC/rebar - grey, high structural edges
    3: ((180, 110, 90), 25, 0.45),   # red brick tones, medium edges
    4: ((200, 195, 185), 20, 0.30),  # plaster grey/beige, smoother
    5: ((210, 200, 190), 12, 0.20),  # painted facade, lowest noise, cleanest
}


def _generate_stage_image(stage: int, size=64) -> np.ndarray:
    base_color, noise_level, edge_density = STAGE_PALETTE[stage]
    img = Image.new("RGB", (size, size), base_color)
    draw = ImageDraw.Draw(img)

    n_lines = int(edge_density * 120)
    for _ in range(n_lines):
        x1, y1 = RNG.integers(0, size, 2)
        length = RNG.integers(4, size // 2)
        angle_choice = RNG.choice(["h", "v"])
        if angle_choice == "h":
            x2, y2 = min(size - 1, x1 + length), y1
        else:
            x2, y2 = x1, min(size - 1, y1 + length)
        shade = tuple(int(np.clip(c + RNG.integers(-40, 40), 0, 255)) for c in base_color)
        draw.line([(x1, y1), (x2, y2)], fill=shade, width=1)

    arr = np.array(img).astype(np.float32)
    noise = RNG.normal(0, noise_level, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return arr


def extract_features(img_arr: np.ndarray) -> dict:
    """Classical CV features standing in for learned CNN features:
    color histogram stats + a simple gradient-magnitude edge-density proxy."""
    gray = img_arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge_density = (gx + gy) / 2

    r, g, b = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
    return {
        "mean_r": r.mean(), "mean_g": g.mean(), "mean_b": b.mean(),
        "std_r": r.std(), "std_g": g.std(), "std_b": b.std(),
        "brightness": gray.mean(), "brightness_std": gray.std(),
        "edge_density": edge_density,
        "color_uniformity": 1.0 / (1.0 + r.std() + g.std() + b.std()),
    }


def build_dataset(n_per_stage=150, size=64):
    records = []
    images = []
    for stage in range(1, 6):
        for _ in range(n_per_stage):
            img = _generate_stage_image(stage, size=size)
            feats = extract_features(img)
            feats["stage"] = stage
            feats["stage_name"] = STAGE_NAMES[stage]
            records.append(feats)
            images.append(img)
    return pd.DataFrame(records), images


def train_stage_classifier(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ("stage", "stage_name")]
    X = df[feature_cols]
    y = df["stage"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[STAGE_NAMES[s] for s in sorted(y.unique())], output_dict=True)

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return {"model": model, "feature_cols": feature_cols, "accuracy": acc, "report": report,
             "importances": importances, "X_test": X_test, "y_test": y_test, "y_pred": y_pred}


def compare_to_promised_timeline(current_stage: int, promised_stage_by_now: int) -> dict:
    """Buyer/lender-facing check: is actual progress behind promised schedule?"""
    delta = current_stage - promised_stage_by_now
    if delta >= 0:
        status = "On schedule or ahead"
    elif delta == -1:
        status = "Slightly behind — monitor next disbursement milestone"
    else:
        status = "Materially behind promised timeline — flag for review"
    return {"current_stage": STAGE_NAMES[current_stage], "promised_stage": STAGE_NAMES[promised_stage_by_now],
            "delta": delta, "status": status}


if __name__ == "__main__":
    df, images = build_dataset(n_per_stage=150)
    results = train_stage_classifier(df)

    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"\nTop features:\n{results['importances'].head(6)}")

    joblib.dump({"model": results["model"], "feature_cols": results["feature_cols"]},
                "/home/claude/prism/models/construction_rf.joblib")
    df.to_csv("/home/claude/prism/data/construction_stage_features.csv", index=False)

    print("\nExample builder-timeline check:")
    print(compare_to_promised_timeline(current_stage=2, promised_stage_by_now=4))
