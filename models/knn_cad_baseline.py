
# Simple KNN baseline for the combined CAD-risk dataset without scikit-learn.

# What it does:
#Loads data/combined/cad_risk_training.csv
# Uses a compact feature set with broad coverage across datasets
# Imputes missing values with training medians
# Standardizes features using training mean/std
# Runs a stratified train/test split
# Evaluates a NumPy-based KNN classifier

#Outputs:
# data/combined/knn_baseline_metrics.json


from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "combined" / "cad_risk_training.csv"
OUTPUT_JSON = ROOT / "data" / "combined" / "knn_baseline_metrics.json"

# Chosen for a first baseline: broadly available clinical features.
FEATURES = [
    "age_years",
    "sex",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "fasting_glucose",
    "smoking",
]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    # Convert the chosen columns into numeric values that KNN can compare.
    x = df.copy()
    x["sex"] = x["sex"].astype("string").str.strip().str.lower().map({"female": 0.0, "male": 1.0})
    for col in FEATURES:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x[FEATURES]


def stratified_split(y: np.ndarray, test_size: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    # Keep both classes represented in train and test sets.
    rng = np.random.default_rng(seed)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n0_test = max(1, int(round(len(idx0) * test_size)))
    n1_test = max(1, int(round(len(idx1) * test_size)))
    test_idx = np.concatenate([idx0[:n0_test], idx1[:n1_test]])
    train_idx = np.concatenate([idx0[n0_test:], idx1[n1_test:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return train_idx, test_idx


def fit_imputer_scaler(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Learn missing-value medians and scaling values from training data only.
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    x_train_filled = np.where(np.isnan(x_train), medians, x_train)
    means = x_train_filled.mean(axis=0)
    stds = x_train_filled.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    return medians, means, stds


def transform(x: np.ndarray, medians: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    # Apply the same imputation and scaling to train/test data.
    x = np.where(np.isnan(x), medians, x)
    return (x - means) / stds


def knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    k: int = 11,
    chunk_size: int = 128,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    for start in range(0, len(x_test), chunk_size):
        chunk = x_test[start : start + chunk_size]
        # Euclidean distance using broadcasting over a small test chunk.
        dists = np.sqrt(((chunk[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2))
        nn_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        votes = y_train[nn_idx]
        preds.append((votes.mean(axis=1) >= 0.5).astype(int))
    return np.concatenate(preds)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    # Basic classification metrics without depending on scikit-learn.
    # t = true f= false p = positive n= negative
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    def safe_div(a: float, b: float) -> float:
        return a / b if b else 0.0

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall_sensitivity": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
    }


def main() -> None:
    # Run the baseline from data loading to saved metric JSON.
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    # Keep runtime reasonable for a baseline while preserving class balance.
    max_rows = 12000
    if len(df) > max_rows:
        df0 = df[df["target_cad_risk"] == 0].sample(n=max_rows // 2, random_state=42)
        df1 = df[df["target_cad_risk"] == 1].sample(n=max_rows // 2, random_state=42)
        df = pd.concat([df0, df1], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    x_df = encode_features(df)
    y = df["target_cad_risk"].astype(int).to_numpy()
    x = x_df.to_numpy(dtype=float)

    train_idx, test_idx = stratified_split(y, test_size=0.2, seed=42)
    x_train_raw, x_test_raw = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    medians, means, stds = fit_imputer_scaler(x_train_raw)
    x_train = transform(x_train_raw, medians, means, stds)
    x_test = transform(x_test_raw, medians, means, stds)

# used 11 for tie-prevention purpose and then test to predict.
    k = 11
    y_pred = knn_predict(x_train, y_train, x_test, k=k)
    metrics = calc_metrics(y_test, y_pred)

    result = {
        "model": "KNN baseline (NumPy implementation)",
        "input_csv": str(INPUT_CSV),
        "features_used": FEATURES,
        "rows_used": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "k": k,
        "preprocessing_for_model": {
            "numeric_imputation": "median",
            "categorical_encoding": {"sex": {"female": 0, "male": 1}},
            "feature_scaling": "z-score standardization using training mean/std",
        },
        "metrics": metrics,
    }

    OUTPUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
