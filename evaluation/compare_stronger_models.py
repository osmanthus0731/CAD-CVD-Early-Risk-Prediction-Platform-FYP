#!/usr/bin/env python3
"""Compare stronger CAD/CVD risk models on the same holdout split."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "combined" / "cad_risk_training.csv"
OUTPUT_JSON = ROOT / "data" / "combined" / "strong_model_comparison.json"

MODEL_FIELDS = [
    "age_years",
    "sex",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "fasting_glucose",
    "smoking",
    "diabetes",
    "family_history_cvd",
]

NUMERIC_FIELDS = [
    "age_years",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "fasting_glucose",
]

BINARY_FIELDS = [
    "sex",
    "smoking",
    "diabetes",
    "family_history_cvd",
]

ENGINEERED_FEATURES = [
    "pulse_pressure",
    "mean_arterial_pressure",
    "age_x_systolic_bp",
    "age_x_cholesterol",
    "bmi_x_systolic_bp",
    "cholesterol_x_glucose",
    "smoking_x_age",
    "diabetes_x_age",
    "family_history_x_age",
]


def encode_binary(field: str, value):
    # Convert text categories into numeric values for sklearn models.
    if value is None or (isinstance(value, float) and np.isnan(value)):
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
    # Select only the questionnaire-style model columns.
    prepared: dict[str, pd.Series] = {}
    for field in MODEL_FIELDS:
        if field in NUMERIC_FIELDS:
            prepared[field] = pd.to_numeric(df[field], errors="coerce")
        elif field in BINARY_FIELDS:
            prepared[field] = df[field].map(lambda v, f=field: encode_binary(f, v))
    return pd.DataFrame(prepared)


def stratified_train_test_split(y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    # Split while preserving the low/high risk class balance.
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


def make_engineered_features(x_filled: np.ndarray) -> np.ndarray:
    # Add simple clinical interaction features for stronger models.
    age = x_filled[:, 0]
    bmi = x_filled[:, 2]
    systolic_bp = x_filled[:, 3]
    diastolic_bp = x_filled[:, 4]
    cholesterol = x_filled[:, 5]
    fasting_glucose = x_filled[:, 6]
    smoking = x_filled[:, 7]
    diabetes = x_filled[:, 8]
    family_history = x_filled[:, 9]

    return np.column_stack(
        [
            systolic_bp - diastolic_bp,
            (systolic_bp + (2.0 * diastolic_bp)) / 3.0,
            age * systolic_bp,
            age * cholesterol,
            bmi * systolic_bp,
            cholesterol * fasting_glucose,
            smoking * age,
            diabetes * age,
            family_history * age,
        ]
    )


def fit_preprocessor(x_train: np.ndarray):
    # Learn medians and scaling values from the training split only.
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    x_filled = np.where(np.isnan(x_train), medians, x_train)
    x_full = np.concatenate([x_filled, make_engineered_features(x_filled), np.isnan(x_train).astype(float)], axis=1)
    means = x_full.mean(axis=0)
    stds = x_full.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    return medians, means, stds


def transform_features(x_raw: np.ndarray, medians: np.ndarray, means: np.ndarray, stds: np.ndarray):
    # Apply imputation, engineered features, missing flags, and scaling.
    x_filled = np.where(np.isnan(x_raw), medians, x_raw)
    x_full = np.concatenate([x_filled, make_engineered_features(x_filled), np.isnan(x_raw).astype(float)], axis=1)
    return (x_full - means) / stds, x_full


def metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5):
    # Convert probabilities into the usual classification metric set.
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    brier = brier_score_loss(y_true, probs)
    return {
        "threshold": round(float(threshold), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_sensitivity": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
        "brier_score": round(float(brier), 4),
        "rmse": round(float(math.sqrt(brier)), 4),
        "log_loss": round(float(log_loss(y_true, np.clip(probs, 1e-7, 1 - 1e-7))), 4),
    }


def best_threshold_by_f1(y_true: np.ndarray, probs: np.ndarray):
    # Try simple thresholds and keep the one with the best F1 score.
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.25, 0.76, 0.01):
        f1 = f1_score(y_true, probs >= threshold, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def optional_models():
    # Add optional libraries only if they are installed on this machine.
    models = []

    try:
        from xgboost import XGBClassifier

        models.append(
            (
                "XGBoost",
                XGBClassifier(
                    n_estimators=500,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=2.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
                "full",
            )
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        models.append(
            (
                "LightGBM",
                LGBMClassifier(
                    n_estimators=600,
                    learning_rate=0.035,
                    num_leaves=24,
                    max_depth=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=2.0,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
                "full",
            )
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        models.append(
            (
                "CatBoost",
                CatBoostClassifier(
                    iterations=500,
                    depth=4,
                    learning_rate=0.04,
                    l2_leaf_reg=5.0,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    random_seed=42,
                    verbose=False,
                    thread_count=-1,
                ),
                "full",
            )
        )
    except Exception:
        pass

    return models


def main() -> None:
    # Train many candidate models and save a ranked comparison.
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    x_raw = build_feature_frame(df).to_numpy(dtype=float)
    y = df["target_cad_risk"].astype(int).to_numpy()

    train_idx, test_idx = stratified_train_test_split(y, test_size=0.2, seed=42)
    x_train_raw = x_raw[train_idx]
    x_test_raw = x_raw[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    medians, means, stds = fit_preprocessor(x_train_raw)
    x_train_scaled, x_train_full = transform_features(x_train_raw, medians, means, stds)
    x_test_scaled, x_test_full = transform_features(x_test_raw, medians, means, stds)

    models = [
        (
            "LogisticRegression sklearn",
            LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0, solver="lbfgs", random_state=42),
            "scaled",
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=350,
                max_depth=12,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            "full",
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier(
                n_estimators=450,
                max_depth=14,
                min_samples_leaf=12,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "full",
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingClassifier(
                max_iter=450,
                learning_rate=0.035,
                max_leaf_nodes=24,
                l2_regularization=0.08,
                random_state=42,
            ),
            "full",
        ),
        (
            "MLP neural network",
            MLPClassifier(
                hidden_layer_sizes=(64, 24),
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=120,
                early_stopping=True,
                random_state=42,
            ),
            "scaled",
        ),
    ]
    models.extend(optional_models())

    results = []
    for name, model, feature_set in models:
        # Each model chooses either scaled features or raw full features.
        start = time.time()
        x_train = x_train_scaled if feature_set == "scaled" else x_train_full
        x_test = x_test_scaled if feature_set == "scaled" else x_test_full
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_test)[:, 1]
        threshold = best_threshold_by_f1(y_test, probs)
        result = {
            "model": name,
            "feature_set": feature_set,
            "seconds": round(time.time() - start, 2),
            "metrics_at_0_5": metrics_from_probs(y_test, probs, threshold=0.5),
            "metrics_at_best_f1_threshold": metrics_from_probs(y_test, probs, threshold=threshold),
        }
        results.append(result)
        print(json.dumps({"model": name, **result["metrics_at_0_5"]}, indent=2))

    results.sort(
        key=lambda row: (
            row["metrics_at_0_5"]["roc_auc"],
            row["metrics_at_0_5"]["balanced_accuracy"],
            row["metrics_at_0_5"]["f1"],
            -row["metrics_at_0_5"]["brier_score"],
        ),
        reverse=True,
    )

    output = {
        "input_csv": str(INPUT_CSV),
        "rows_total": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "feature_names": MODEL_FIELDS + ENGINEERED_FEATURES + [f"{field}_missing" for field in MODEL_FIELDS],
        "selection_note": "Sorted by holdout ROC-AUC, then balanced accuracy, F1, and Brier score.",
        "best_model": results[0],
        "results": results,
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nBest model:")
    print(json.dumps(output["best_model"], indent=2))
    print(f"\nSaved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
