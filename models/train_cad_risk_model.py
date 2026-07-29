#!/usr/bin/env python3
"""
Train a more stable CAD/CVD risk model from the combined dataset.

Why this version exists:
- Keeps the lightweight NumPy-only deployment path.
- Uses cleaner source inclusion from the combined dataset builder.
- Adds engineered risk features and repeated stratified CV for steadier tuning.
- Tunes the decision threshold on out-of-fold predictions for more consistent
  classification behavior.

Outputs:
- data/combined/cad_risk_model.json
- data/combined/cad_risk_model_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "combined" / "cad_risk_training.csv"
MODEL_JSON = ROOT / "data" / "combined" / "cad_risk_model.json"
METRICS_JSON = ROOT / "data" / "combined" / "cad_risk_model_metrics.json"

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

FEATURE_DESCRIPTIONS = {
    "age_years": "Age (years)",
    "sex": "Sex",
    "bmi": "BMI",
    "systolic_bp": "Systolic blood pressure",
    "diastolic_bp": "Diastolic blood pressure",
    "cholesterol": "Total cholesterol",
    "fasting_glucose": "Fasting glucose",
    "smoking": "Smoking",
    "diabetes": "Diabetes",
    "family_history_cvd": "Family history of CVD/CAD",
    "pulse_pressure": "Pulse pressure",
    "mean_arterial_pressure": "Mean arterial pressure",
    "age_x_systolic_bp": "Age x systolic blood pressure",
    "age_x_cholesterol": "Age x cholesterol",
    "bmi_x_systolic_bp": "BMI x systolic blood pressure",
    "cholesterol_x_glucose": "Cholesterol x fasting glucose",
    "smoking_x_age": "Smoking x age",
    "diabetes_x_age": "Diabetes x age",
    "family_history_x_age": "Family history x age",
}


def to_float(value):
    # Convert raw into floats for model 
    if value is None:
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def encode_binary(field: str, value):
    # Convert sex and yes/no-style values into 0/1.
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
    return to_float(value)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only the app questionnaire fields and encode them consistently.
    prepared: dict[str, pd.Series] = {}
    for field in MODEL_FIELDS:
        if field in NUMERIC_FIELDS:
            prepared[field] = pd.to_numeric(df[field], errors="coerce")
        else:
            prepared[field] = df[field].map(lambda v, f=field: encode_binary(f, v))
    return pd.DataFrame(prepared)


def repeated_stratified_kfold_indices(
    y: np.ndarray,
    n_splits: int = 5,
    seeds: tuple[int, ...] = (42, 2024),
):
    # Yield repeated class-balanced folds for steadier validation.
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx0 = np.flatnonzero(y == 0)
        idx1 = np.flatnonzero(y == 1)
        rng.shuffle(idx0)
        rng.shuffle(idx1)
        fold0 = np.array_split(idx0, n_splits)
        fold1 = np.array_split(idx1, n_splits)

        for fold_idx in range(n_splits):
            valid_idx = np.concatenate([fold0[fold_idx], fold1[fold_idx]])
            train_parts = [x for i, x in enumerate(fold0) if i != fold_idx] + [x for i, x in enumerate(fold1) if i != fold_idx]
            train_idx = np.concatenate(train_parts)
            rng.shuffle(train_idx)
            rng.shuffle(valid_idx)
            yield seed, fold_idx, train_idx, valid_idx


def stratified_train_test_split(y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    # Hold out a balanced test set for final reporting.
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


def stratified_train_calibration_split(y: np.ndarray, calibration_size: float = 0.15, seed: int = 123):
    # Split training data again so calibration is not fitted on the same rows.
    rng = np.random.default_rng(seed)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n0_cal = max(1, int(round(len(idx0) * calibration_size)))
    n1_cal = max(1, int(round(len(idx1) * calibration_size)))
    calib_idx = np.concatenate([idx0[:n0_cal], idx1[:n1_cal]])
    fit_idx = np.concatenate([idx0[n0_cal:], idx1[n1_cal:]])
    rng.shuffle(fit_idx)
    rng.shuffle(calib_idx)
    return fit_idx, calib_idx


def make_engineered_features(x_filled: np.ndarray) -> np.ndarray:
    # Add simple clinical combinations that can improve model signal.
    age = x_filled[:, 0]
    bmi = x_filled[:, 2]
    systolic_bp = x_filled[:, 3]
    diastolic_bp = x_filled[:, 4]
    cholesterol = x_filled[:, 5]
    fasting_glucose = x_filled[:, 6]
    smoking = x_filled[:, 7]
    diabetes = x_filled[:, 8]
    family_history = x_filled[:, 9]

    pulse_pressure = systolic_bp - diastolic_bp
    mean_arterial_pressure = (systolic_bp + (2.0 * diastolic_bp)) / 3.0
    age_x_systolic_bp = age * systolic_bp
    age_x_cholesterol = age * cholesterol
    bmi_x_systolic_bp = bmi * systolic_bp
    cholesterol_x_glucose = cholesterol * fasting_glucose
    smoking_x_age = smoking * age
    diabetes_x_age = diabetes * age
    family_history_x_age = family_history * age

    return np.column_stack(
        [
            pulse_pressure,
            mean_arterial_pressure,
            age_x_systolic_bp,
            age_x_cholesterol,
            bmi_x_systolic_bp,
            cholesterol_x_glucose,
            smoking_x_age,
            diabetes_x_age,
            family_history_x_age,
        ]
    )


def fit_preprocessor(x_train: np.ndarray):
    # Learn medians, means, and standard deviations from training data only.
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    missing_rates = np.isnan(x_train).mean(axis=0)
    x_filled = np.where(np.isnan(x_train), medians, x_train)
    engineered = make_engineered_features(x_filled)
    x_full = np.concatenate([x_filled, engineered, np.isnan(x_train).astype(float)], axis=1)
    means = x_full.mean(axis=0)
    stds = x_full.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    return medians, means, stds, missing_rates


def transform_features(x_raw: np.ndarray, medians: np.ndarray, means: np.ndarray, stds: np.ndarray):
    # Apply the saved preprocessing steps to any raw feature matrix.
    missing_flags = np.isnan(x_raw).astype(float)
    x_filled = np.where(np.isnan(x_raw), medians, x_raw)
    engineered = make_engineered_features(x_filled)
    x_full = np.concatenate([x_filled, engineered, missing_flags], axis=1)
    x_scaled = (x_full - means) / stds
    return x_scaled, x_filled, x_full


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Clip first so very large values do not overflow exp().
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    l2_penalty: float,
    learning_rate: float = 0.06,
    epochs: int = 500,
):
    # Small NumPy logistic regression trainer with class weighting and L2 penalty.
    n_samples, n_features = x.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0

    class_counts = np.bincount(y.astype(int), minlength=2).astype(float)
    total = class_counts.sum()
    class_weights = {
        0: total / (2.0 * max(class_counts[0], 1.0)),
        1: total / (2.0 * max(class_counts[1], 1.0)),
    }
    sample_weights = np.where(y == 1, class_weights[1], class_weights[0])

    for _ in range(epochs):
        logits = x @ weights + bias
        probs = sigmoid(logits)
        errors = (probs - y) * sample_weights
        grad_w = (x.T @ errors) / n_samples + (l2_penalty * weights)
        grad_b = errors.mean()
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return weights, bias


def predict_proba(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return sigmoid(x @ weights + bias)


def predict_logits(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return x @ weights + bias


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5):
    # Convert probabilities into confusion-matrix based metrics.
    y_pred = (probs >= threshold).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "threshold": round(float(threshold), 4),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall_sensitivity": round(recall, 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "f1": round(f1, 4),
    }


def roc_auc_score_manual(y_true: np.ndarray, probs: np.ndarray) -> float:
    # Manual AUC calculation for the lightweight NumPy training path.
    y_true = y_true.astype(int)
    pos = probs[y_true == 1]
    neg = probs[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    comparisons = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    auc = (comparisons + 0.5 * ties) / (len(pos) * len(neg))
    return float(auc)


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs - y_true) ** 2))


def log_loss_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return float(-(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs)).mean())


def fit_platt_scaler(logits: np.ndarray, y: np.ndarray, learning_rate: float = 0.05, epochs: int = 800):
    # Learn a simple scale and offset to calibrate raw logits.
    scale = 1.0
    offset = 0.0
    n = len(y)

    for _ in range(epochs):
        calibrated_logits = (scale * logits) + offset
        probs = sigmoid(calibrated_logits)
        errors = probs - y
        grad_scale = float(np.dot(errors, logits) / n)
        grad_offset = float(errors.mean())
        scale -= learning_rate * grad_scale
        offset -= learning_rate * grad_offset

    return float(scale), float(offset)


def apply_platt_scaler(logits: np.ndarray, scale: float, offset: float) -> np.ndarray:
    return sigmoid((scale * logits) + offset)


def fit_isotonic_calibrator(scores: np.ndarray, y: np.ndarray):
    # Build a monotonic calibration curve from validation scores.
    order = np.argsort(scores)
    sorted_scores = np.asarray(scores, dtype=float)[order]
    sorted_y = np.asarray(y, dtype=float)[order]

    values = sorted_y.tolist()
    weights = [1.0] * len(sorted_y)
    starts = list(range(len(sorted_y)))
    ends = list(range(len(sorted_y)))

    idx = 0
    while idx < len(values) - 1:
        if values[idx] > values[idx + 1]:
            total_weight = weights[idx] + weights[idx + 1]
            pooled_value = ((values[idx] * weights[idx]) + (values[idx + 1] * weights[idx + 1])) / total_weight
            values[idx] = pooled_value
            weights[idx] = total_weight
            ends[idx] = ends[idx + 1]
            del values[idx + 1]
            del weights[idx + 1]
            del starts[idx + 1]
            del ends[idx + 1]
            if idx > 0:
                idx -= 1
        else:
            idx += 1

    thresholds = [float(sorted_scores[end]) for end in ends]
    predictions = [float(np.clip(value, 1e-7, 1.0 - 1e-7)) for value in values]
    return {
        "thresholds": thresholds,
        "predictions": predictions,
        "min_score": float(sorted_scores[0]),
        "max_score": float(sorted_scores[-1]),
    }


def apply_isotonic_calibrator(scores: np.ndarray, calibrator: dict) -> np.ndarray:
    thresholds = np.asarray(calibrator["thresholds"], dtype=float)
    predictions = np.asarray(calibrator["predictions"], dtype=float)
    min_score = float(calibrator["min_score"])
    max_score = float(calibrator["max_score"])
    scores = np.asarray(scores, dtype=float)

    idx = np.searchsorted(thresholds, scores, side="left")
    idx = np.clip(idx, 0, len(predictions) - 1)
    calibrated = predictions[idx]
    calibrated = np.where(scores < min_score, predictions[0], calibrated)
    calibrated = np.where(scores > max_score, predictions[-1], calibrated)
    return np.clip(calibrated, 1e-7, 1.0 - 1e-7)


def summarize_prob_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float):
    # Keep probability and classification metrics together.
    metrics = binary_metrics(y_true, probs, threshold=threshold)
    metrics["roc_auc"] = round(roc_auc_score_manual(y_true, probs), 4)
    metrics["brier_score"] = round(brier_score(y_true, probs), 4)
    metrics["log_loss"] = round(log_loss_score(y_true, probs), 4)
    return metrics


def evaluate_grid(x_raw: np.ndarray, y: np.ndarray):
    # Try a small hyperparameter grid and rank settings by validation metrics.
    penalties = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05]
    learning_rates = [0.04, 0.06, 0.08]
    results = []

    for l2_penalty in penalties:
        for learning_rate in learning_rates:
            fold_scores = []
            for _, _, train_idx, valid_idx in repeated_stratified_kfold_indices(y, n_splits=5, seeds=(42, 2024)):
                x_train_raw = x_raw[train_idx]
                x_valid_raw = x_raw[valid_idx]
                y_train = y[train_idx]
                y_valid = y[valid_idx]

                medians, means, stds, _ = fit_preprocessor(x_train_raw)
                x_train, _, _ = transform_features(x_train_raw, medians, means, stds)
                x_valid, _, _ = transform_features(x_valid_raw, medians, means, stds)

                weights, bias = fit_logistic_regression(
                    x_train,
                    y_train,
                    l2_penalty=l2_penalty,
                    learning_rate=learning_rate,
                )
                probs = predict_proba(x_valid, weights, bias)
                fold_scores.append(
                    {
                        "balanced_accuracy": binary_metrics(y_valid, probs, threshold=0.5)["balanced_accuracy"],
                        "f1": binary_metrics(y_valid, probs, threshold=0.5)["f1"],
                        "roc_auc": round(roc_auc_score_manual(y_valid, probs), 4),
                        "brier_score": round(brier_score(y_valid, probs), 4),
                    }
                )

            results.append(
                {
                    "l2_penalty": l2_penalty,
                    "learning_rate": learning_rate,
                    "mean_balanced_accuracy": round(float(np.mean([x["balanced_accuracy"] for x in fold_scores])), 4),
                    "mean_f1": round(float(np.mean([x["f1"] for x in fold_scores])), 4),
                    "mean_roc_auc": round(float(np.mean([x["roc_auc"] for x in fold_scores])), 4),
                    "mean_brier_score": round(float(np.mean([x["brier_score"] for x in fold_scores])), 4),
                }
            )

    results.sort(
        key=lambda row: (
            row["mean_roc_auc"],
            row["mean_balanced_accuracy"],
            row["mean_f1"],
            -row["mean_brier_score"],
        ),
        reverse=True,
    )
    return results


def sample_for_cv(x_raw: np.ndarray, y: np.ndarray, max_rows: int = 20000, seed: int = 42):
    # Speed up CV by using a balanced sample when the dataset is large.
    if len(y) <= max_rows:
        return x_raw, y

    rng = np.random.default_rng(seed)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    n0 = min(len(idx0), max_rows // 2)
    n1 = min(len(idx1), max_rows - n0)
    chosen = np.concatenate(
        [
            rng.choice(idx0, size=n0, replace=False),
            rng.choice(idx1, size=n1, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return x_raw[chosen], y[chosen]


def tune_threshold(x_raw: np.ndarray, y: np.ndarray, l2_penalty: float, learning_rate: float):
    # Use out-of-fold predictions to choose a practical decision threshold.
    oof_probs = np.zeros(len(y), dtype=float)
    seen = np.zeros(len(y), dtype=int)

    for _, _, train_idx, valid_idx in repeated_stratified_kfold_indices(y, n_splits=5, seeds=(42,)):
        x_train_raw = x_raw[train_idx]
        x_valid_raw = x_raw[valid_idx]
        y_train = y[train_idx]

        medians, means, stds, _ = fit_preprocessor(x_train_raw)
        x_train, _, _ = transform_features(x_train_raw, medians, means, stds)
        x_valid, _, _ = transform_features(x_valid_raw, medians, means, stds)

        weights, bias = fit_logistic_regression(
            x_train,
            y_train,
            l2_penalty=l2_penalty,
            learning_rate=learning_rate,
        )
        oof_probs[valid_idx] += predict_proba(x_valid, weights, bias)
        seen[valid_idx] += 1

    seen = np.where(seen == 0, 1, seen)
    oof_probs = oof_probs / seen

    best_threshold = 0.5
    best_metrics = summarize_prob_metrics(y, oof_probs, threshold=best_threshold)
    for threshold in np.arange(0.35, 0.66, 0.01):
        metrics = summarize_prob_metrics(y, oof_probs, threshold=float(threshold))
        if (
            metrics["balanced_accuracy"],
            metrics["f1"],
            metrics["roc_auc"],
        ) > (
            best_metrics["balanced_accuracy"],
            best_metrics["f1"],
            best_metrics["roc_auc"],
        ):
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def top_coefficients(feature_names: list[str], weights: np.ndarray, limit: int = 10):
    # Save the biggest coefficients so the model can be explained later.
    pairs = sorted(zip(feature_names, weights), key=lambda item: abs(item[1]), reverse=True)
    output = []
    for name, value in pairs[:limit]:
        output.append(
            {
                "feature": name,
                "display_name": FEATURE_DESCRIPTIONS.get(name.replace("_missing", ""), name),
                "coefficient": round(float(value), 5),
            }
        )
    return output


def main():
    # Full training workflow: split, tune, calibrate, evaluate, and save.
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    x_df = build_feature_frame(df)
    y = df["target_cad_risk"].astype(int).to_numpy()
    x_raw = x_df.to_numpy(dtype=float)

    train_idx, test_idx = stratified_train_test_split(y, test_size=0.2, seed=42)
    x_train_raw = x_raw[train_idx]
    x_test_raw = x_raw[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    fit_idx, calib_idx = stratified_train_calibration_split(y_train, calibration_size=0.15, seed=123)
    x_fit_raw = x_train_raw[fit_idx]
    y_fit = y_train[fit_idx]
    x_calib_raw = x_train_raw[calib_idx]
    y_calib = y_train[calib_idx]

    x_cv_raw, y_cv = sample_for_cv(x_fit_raw, y_fit, max_rows=20000, seed=42)
    grid_results = evaluate_grid(x_cv_raw, y_cv)
    best = grid_results[0]

    tuned_threshold, threshold_metrics = tune_threshold(
        x_fit_raw,
        y_fit,
        l2_penalty=float(best["l2_penalty"]),
        learning_rate=float(best["learning_rate"]),
    )

    medians, means, stds, missing_rates = fit_preprocessor(x_fit_raw)
    x_fit, _, _ = transform_features(x_fit_raw, medians, means, stds)
    x_calib, _, _ = transform_features(x_calib_raw, medians, means, stds)
    x_test, _, _ = transform_features(x_test_raw, medians, means, stds)

    weights, bias = fit_logistic_regression(
        x_fit,
        y_fit,
        l2_penalty=float(best["l2_penalty"]),
        learning_rate=float(best["learning_rate"]),
    )

    calib_logits = predict_logits(x_calib, weights, bias)
    platt_scale, platt_offset = fit_platt_scaler(calib_logits, y_calib)
    isotonic_calibrator = fit_isotonic_calibrator(calib_logits, y_calib)
    calib_probs_uncalibrated = predict_proba(x_calib, weights, bias)
    calib_probs_platt = apply_platt_scaler(calib_logits, platt_scale, platt_offset)
    calib_probs = apply_isotonic_calibrator(calib_logits, isotonic_calibrator)
    test_logits = predict_logits(x_test, weights, bias)
    test_probs_uncalibrated = predict_proba(x_test, weights, bias)
    test_probs_platt = apply_platt_scaler(test_logits, platt_scale, platt_offset)
    test_probs = apply_isotonic_calibrator(test_logits, isotonic_calibrator)
    calibration_metrics = {
        "uncalibrated": summarize_prob_metrics(y_calib, calib_probs_uncalibrated, threshold=tuned_threshold),
        "platt_scaling": summarize_prob_metrics(y_calib, calib_probs_platt, threshold=tuned_threshold),
        "isotonic": summarize_prob_metrics(y_calib, calib_probs, threshold=tuned_threshold),
    }
    test_metrics = summarize_prob_metrics(y_test, test_probs, threshold=tuned_threshold)
    test_metrics_platt = summarize_prob_metrics(y_test, test_probs_platt, threshold=tuned_threshold)
    test_metrics_uncalibrated = summarize_prob_metrics(y_test, test_probs_uncalibrated, threshold=tuned_threshold)

    transformed_feature_names = MODEL_FIELDS + ENGINEERED_FEATURES + [f"{field}_missing" for field in MODEL_FIELDS]

    model_payload = {
        "model_name": "CAD/CVD Risk Logistic Regression v2 (NumPy)",
        "input_csv": str(INPUT_CSV),
        "question_fields": MODEL_FIELDS,
        "numeric_fields": NUMERIC_FIELDS,
        "binary_fields": BINARY_FIELDS,
        "engineered_feature_names": ENGINEERED_FEATURES,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "feature_names_after_transform": transformed_feature_names,
        "medians": medians.tolist(),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "probability_calibration": {
            "method": "isotonic",
            "platt_reference": {
                "scale": round(float(platt_scale), 6),
                "offset": round(float(platt_offset), 6),
            },
            "isotonic": isotonic_calibrator,
            "calibration_rows": int(len(calib_idx)),
        },
        "threshold": round(float(tuned_threshold), 4),
        "training_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "missing_rate_by_field": {
            field: round(float(rate), 4) for field, rate in zip(MODEL_FIELDS, missing_rates)
        },
        "best_hyperparameters": best,
        "threshold_tuning": {
            "selected_threshold": round(float(tuned_threshold), 4),
            "oof_training_metrics": threshold_metrics,
        },
        "top_coefficients": top_coefficients(transformed_feature_names, weights),
    }

    metrics_payload = {
        "model": model_payload["model_name"],
        "rows_total": int(len(df)),
        "class_distribution": {
            "0": int((y == 0).sum()),
            "1": int((y == 1).sum()),
        },
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "fit_rows_before_calibration": int(len(fit_idx)),
        "calibration_rows": int(len(calib_idx)),
        "grid_search": {
            "search_space": {
                "l2_penalty": [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05],
                "learning_rate": [0.04, 0.06, 0.08],
            },
            "cross_validation": "Repeated 5-fold stratified cross-validation (2 seeds)",
            "results": grid_results,
            "best": best,
        },
        "threshold_tuning": {
            "procedure": "5-fold out-of-fold tuning on training split",
            "selected_threshold": round(float(tuned_threshold), 4),
            "training_oof_metrics": threshold_metrics,
        },
        "probability_calibration": {
            "method": "isotonic",
            "calibration_set_metrics": calibration_metrics,
        },
        "holdout_test_metrics_uncalibrated": test_metrics_uncalibrated,
        "holdout_test_metrics_platt": test_metrics_platt,
        "holdout_test_metrics": test_metrics,
    }

    MODEL_JSON.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")
    METRICS_JSON.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
