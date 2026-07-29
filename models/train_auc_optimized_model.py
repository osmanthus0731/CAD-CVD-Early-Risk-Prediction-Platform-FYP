#!/usr/bin/env python3
"""Train an AUC-focused CAD/CVD model for the app.

The app keeps the transparent clinical score for explanations, but this model is
used when we want stronger patient ranking and benchmark AUC.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "combined" / "cad_risk_training.csv"
MODEL_PATH = ROOT / "data" / "combined" / "cad_auc_model.pkl"
METRICS_PATH = ROOT / "data" / "combined" / "cad_auc_model_metrics.json"

MODEL_FIELDS = [
    "age_years",
    "sex",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "hdl",
    "ldl",
    "triglycerides",
    "fasting_glucose",
    "smoking",
    "diabetes",
    "family_history_cvd",
    "physical_activity",
]

NUMERIC_FIELDS = {
    "age_years",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "hdl",
    "ldl",
    "triglycerides",
    "fasting_glucose",
    "physical_activity",
}

BINARY_FIELDS = {"sex", "smoking", "diabetes", "family_history_cvd"}

ENGINEERED_FEATURES = [
    "pulse_pressure",
    "mean_arterial_pressure",
    "non_hdl_cholesterol",
    "cholesterol_hdl_ratio",
    "ldl_hdl_ratio",
    "triglyceride_hdl_ratio",
    "age_x_systolic_bp",
    "age_x_cholesterol",
    "bmi_x_systolic_bp",
    "smoking_x_age",
    "diabetes_x_age",
    "family_history_x_age",
]


def encode_binary(field: str, value):
    # Convert text labels into numbers while keeping unknown values as NaN.
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if not text:
        return np.nan
    if field == "sex":
        if text in {"male", "m", "1", "1.0"}:
            return 1.0
        if text in {"female", "f", "0", "2", "2.0"}:
            return 0.0
        return np.nan
    if text in {"yes", "y", "true", "1", "1.0", "smoker", "high", "positive"}:
        return 1.0
    if text in {"no", "n", "false", "0", "0.0", "low", "negative"}:
        return 0.0
    return pd.to_numeric(value, errors="coerce")


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Prepare the full feature set, including optional lipid/activity columns.
    prepared = {}
    for field in MODEL_FIELDS:
        if field in NUMERIC_FIELDS:
            prepared[field] = pd.to_numeric(df[field], errors="coerce")
        elif field in BINARY_FIELDS:
            prepared[field] = df[field].map(lambda v, f=field: encode_binary(f, v))
    return pd.DataFrame(prepared)


def stratified_split(y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    # Keep the same class ratio in train and test data.
    rng = np.random.default_rng(seed)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n0_test = max(1, int(round(len(idx0) * test_size)))
    n1_test = max(1, int(round(len(idx1) * test_size)))
    test_idx = np.concatenate([idx0[:n0_test], idx1[:n1_test]])
    train_idx = np.concatenate([idx0[n0_test:], idx1[n1_test:]])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def make_engineered_features(x: np.ndarray) -> np.ndarray:
    # Add ratios and interaction terms that help tree models rank risk.
    age = x[:, 0]
    bmi = x[:, 2]
    systolic_bp = x[:, 3]
    diastolic_bp = x[:, 4]
    cholesterol = x[:, 5]
    hdl = np.maximum(x[:, 6], 1.0)
    ldl = x[:, 7]
    triglycerides = x[:, 8]
    smoking = x[:, 10]
    diabetes = x[:, 11]
    family_history = x[:, 12]

    return np.column_stack(
        [
            systolic_bp - diastolic_bp,
            (systolic_bp + (2.0 * diastolic_bp)) / 3.0,
            cholesterol - hdl,
            cholesterol / hdl,
            ldl / hdl,
            triglycerides / hdl,
            age * systolic_bp,
            age * cholesterol,
            bmi * systolic_bp,
            smoking * age,
            diabetes * age,
            family_history * age,
        ]
    )


def fit_preprocessor(x_train: np.ndarray):
    # Store training medians so future app predictions fill missing values the same way.
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    x_filled = np.where(np.isnan(x_train), medians, x_train)
    x_full = np.concatenate([x_filled, make_engineered_features(x_filled), np.isnan(x_train).astype(float)], axis=1)
    return medians, x_full


def transform(x_raw: np.ndarray, medians: np.ndarray):
    # Build the final model matrix: filled values + engineered values + missing flags.
    x_filled = np.where(np.isnan(x_raw), medians, x_raw)
    return np.concatenate([x_filled, make_engineered_features(x_filled), np.isnan(x_raw).astype(float)], axis=1)


def main() -> None:
    # Train several AUC-focused models and save the best one as a pickle.
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    x_raw = build_feature_frame(df).to_numpy(dtype=float)
    y = df["target_cad_risk"].astype(int).to_numpy()

    train_idx, test_idx = stratified_split(y)
    x_train_raw, x_test_raw = x_raw[train_idx], x_raw[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    medians, x_train = fit_preprocessor(x_train_raw)
    x_test = transform(x_test_raw, medians)

    candidates = [
        (
            "ExtraTrees AUC optimized",
            ExtraTreesClassifier(
                n_estimators=800,
                max_depth=18,
                min_samples_leaf=8,
                min_samples_split=16,
                max_features=0.75,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "RandomForest AUC optimized",
            RandomForestClassifier(
                n_estimators=700,
                max_depth=16,
                min_samples_leaf=10,
                min_samples_split=20,
                max_features=0.75,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "HistGradientBoosting AUC optimized",
            HistGradientBoostingClassifier(
                max_iter=700,
                learning_rate=0.028,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                min_samples_leaf=30,
                random_state=42,
            ),
        ),
    ]

    results = []
    best_name = ""
    best_model = None
    best_auc = -1.0
    for name, model in candidates:
        # Pick the model that gives the best holdout ROC-AUC.
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_test)[:, 1]
        auc = float(roc_auc_score(y_test, probs))
        results.append({"model": name, "holdout_roc_auc": round(auc, 4)})
        if auc > best_auc:
            best_name = name
            best_model = model
            best_auc = auc

    feature_names = MODEL_FIELDS + ENGINEERED_FEATURES + [f"{field}_missing" for field in MODEL_FIELDS]
    payload = {
        "model_name": best_name,
        "model": best_model,
        "fields": MODEL_FIELDS,
        "numeric_fields": sorted(NUMERIC_FIELDS),
        "binary_fields": sorted(BINARY_FIELDS),
        "engineered_features": ENGINEERED_FEATURES,
        "feature_names": feature_names,
        "medians": medians,
    }
    with MODEL_PATH.open("wb") as f:
        pickle.dump(payload, f)

    metrics = {
        "model_path": str(MODEL_PATH),
        "rows_total": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "best_model": {"model": best_name, "holdout_roc_auc": round(best_auc, 4)},
        "results": sorted(results, key=lambda row: row["holdout_roc_auc"], reverse=True),
        "note": "AUC-optimized model includes optional HDL, LDL, triglycerides, and physical activity when available.",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
