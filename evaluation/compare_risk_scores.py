#!/usr/bin/env python3
"""
Benchmark the project risk model against Framingham and SCORE2.

This comparison is intentionally limited to the subset with the variables needed
for each published score and documents the assumptions required by the local
datasets.

Outputs:
- data/combined/risk_score_comparison.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "combined" / "cad_risk_training.csv"
MODEL_JSON = ROOT / "data" / "combined" / "cad_risk_model.json"
OUTPUT_JSON = ROOT / "data" / "combined" / "risk_score_comparison.json"

MMOL_PER_MGDL_CHOLESTEROL = 1.0 / 38.67

# Published SCORE2 and Framingham coefficients used for benchmark calculations.
SCORE2_COEFFICIENTS = {
    "male": {
        "age": 0.3742,
        "smoking": 0.6012,
        "sbp": 0.2777,
        "diabetes": 0.6457,
        "tchol": 0.1458,
        "hdl": -0.2698,
        "age_smoking": -0.0755,
        "age_sbp": -0.0255,
        "age_tchol": -0.0281,
        "age_hdl": 0.0426,
        "age_diabetes": -0.0983,
        "baseline_survival": 0.9605,
        "scale_1": {
            "low": -0.5699,
            "moderate": -0.1565,
            "high": 0.3207,
            "very_high": 0.5836,
        },
        "scale_2": {
            "low": 0.7476,
            "moderate": 0.8009,
            "high": 0.9360,
            "very_high": 0.8294,
        },
    },
    "female": {
        "age": 0.4648,
        "smoking": 0.7744,
        "sbp": 0.3131,
        "diabetes": 0.8096,
        "tchol": 0.1002,
        "hdl": -0.2606,
        "age_smoking": -0.1088,
        "age_sbp": -0.0277,
        "age_tchol": -0.0226,
        "age_hdl": 0.0613,
        "age_diabetes": -0.1272,
        "baseline_survival": 0.9776,
        "scale_1": {
            "low": -0.7380,
            "moderate": -0.3143,
            "high": 0.5710,
            "very_high": 0.9412,
        },
        "scale_2": {
            "low": 0.7019,
            "moderate": 0.7701,
            "high": 0.9369,
            "very_high": 0.8329,
        },
    },
}

FRAMINGHAM_LIPIDS = {
    "male": {
        "age": 3.06117,
        "tchol": 1.12370,
        "hdl": -0.93263,
        "sbp_untreated": 1.93303,
        "smoking": 0.65451,
        "diabetes": 0.57367,
        "baseline_survival": 0.88936,
        "mean_sum": 23.9802,
    },
    "female": {
        "age": 2.32888,
        "tchol": 1.20904,
        "hdl": -0.70833,
        "sbp_untreated": 2.76157,
        "smoking": 0.52873,
        "diabetes": 0.69154,
        "baseline_survival": 0.95012,
        "mean_sum": 26.1931,
    },
}


def encode_binary(field: str, value):
    # Convert sex and yes/no fields into 0/1 values for modelling.
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
    try:
        return float(text)
    except ValueError:
        return np.nan


def build_feature_frame(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    # Build the same model input columns used during project-model training.
    prepared: dict[str, pd.Series] = {}
    for field in fields:
        if field == "sex" or field in {"smoking", "diabetes", "family_history_cvd"}:
            prepared[field] = df[field].map(lambda v, f=field: encode_binary(f, v))
        else:
            prepared[field] = pd.to_numeric(df[field], errors="coerce")
    return pd.DataFrame(prepared)


def make_engineered_features(x_filled: np.ndarray) -> np.ndarray:
    # Recreate the engineered features so benchmark predictions match training.
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


def model_predict_proba(df: pd.DataFrame, model: dict) -> np.ndarray:
    # Run the saved lightweight logistic model from the JSON file.
    x_raw = build_feature_frame(df, model["question_fields"]).to_numpy(dtype=float)
    medians = np.array(model["medians"], dtype=float)
    means = np.array(model["means"], dtype=float)
    stds = np.array(model["stds"], dtype=float)
    missing_flags = np.isnan(x_raw).astype(float)
    x_filled = np.where(np.isnan(x_raw), medians, x_raw)
    engineered = make_engineered_features(x_filled)
    x_full = np.concatenate([x_filled, engineered, missing_flags], axis=1)
    x_scaled = (x_full - means) / stds
    weights = np.array(model["weights"], dtype=float)
    bias = float(model["bias"])
    logits = x_scaled @ weights + bias
    calibration = model.get("probability_calibration", {})
    if calibration.get("method") == "platt_scaling":
        # Platt scaling adjusts raw logits into better probabilities.
        logits = (float(calibration.get("scale", 1.0)) * logits) + float(calibration.get("offset", 0.0))
        logits = np.clip(logits, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-logits))
    if calibration.get("method") == "isotonic":
        # Isotonic calibration uses saved score thresholds.
        isotonic = calibration.get("isotonic", {})
        thresholds = np.asarray(isotonic.get("thresholds", []), dtype=float)
        predictions = np.asarray(isotonic.get("predictions", []), dtype=float)
        if len(thresholds) and len(predictions):
            idx = np.searchsorted(thresholds, logits, side="left")
            idx = np.clip(idx, 0, len(predictions) - 1)
            return np.clip(predictions[idx], 1e-7, 1.0 - 1e-7)
    logits = np.clip(logits, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-logits))


def roc_auc_score_manual(y_true: np.ndarray, probs: np.ndarray) -> float:
    # Manual AUC so this script can still work without sklearn.
    y_true = y_true.astype(int)
    pos = probs[y_true == 1]
    neg = probs[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    comparisons = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((comparisons + (0.5 * ties)) / (len(pos) * len(neg)))


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs - y_true) ** 2))


def log_loss_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return float(-(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs)).mean())


def summarize_probs(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float | int]:
    # Put all benchmark metrics into one compact dictionary.
    return {
        "rows": int(len(y_true)),
        "event_rate": round(float(np.mean(y_true)), 4),
        "mean_predicted_risk": round(float(np.mean(probs)), 4),
        "roc_auc": round(roc_auc_score_manual(y_true, probs), 4),
        "brier_score": round(brier_score(y_true, probs), 4),
        "log_loss": round(log_loss_score(y_true, probs), 4),
    }


def framingham_risk(row: pd.Series) -> float:
    # Lipid-based Framingham general CVD 10-year risk.
    sex = "male" if str(row["sex"]).strip().lower() == "male" else "female"
    coeffs = FRAMINGHAM_LIPIDS[sex]
    lp = (
        coeffs["age"] * np.log(row["age_years"])
        + coeffs["tchol"] * np.log(row["cholesterol"])
        + coeffs["hdl"] * np.log(row["hdl"])
        + coeffs["sbp_untreated"] * np.log(row["systolic_bp"])
        + coeffs["smoking"] * float(row["smoking"])
        + coeffs["diabetes"] * float(row["diabetes"])
    )
    return float(1.0 - (coeffs["baseline_survival"] ** np.exp(lp - coeffs["mean_sum"])))


def score2_risk(row: pd.Series, region: str = "low") -> float:
    # SCORE2 10-year CVD risk using the chosen European risk-region calibration.
    sex = "male" if str(row["sex"]).strip().lower() == "male" else "female"
    coeffs = SCORE2_COEFFICIENTS[sex]
    cage = (float(row["age_years"]) - 60.0) / 5.0
    csbp = (float(row["systolic_bp"]) - 120.0) / 20.0
    tchol = float(row["cholesterol"]) * MMOL_PER_MGDL_CHOLESTEROL
    hdl = float(row["hdl"]) * MMOL_PER_MGDL_CHOLESTEROL
    ctchol = tchol - 6.0
    chdl = (hdl - 1.3) / 0.5
    smoking = float(row["smoking"])
    diabetes = float(row["diabetes"])

    linear_predictor = (
        coeffs["age"] * cage
        + coeffs["smoking"] * smoking
        + coeffs["sbp"] * csbp
        + coeffs["diabetes"] * diabetes
        + coeffs["tchol"] * ctchol
        + coeffs["hdl"] * chdl
        + coeffs["age_smoking"] * cage * smoking
        + coeffs["age_sbp"] * cage * csbp
        + coeffs["age_tchol"] * cage * ctchol
        + coeffs["age_hdl"] * cage * chdl
        + coeffs["age_diabetes"] * cage * diabetes
    )

    uncalibrated = 1.0 - (coeffs["baseline_survival"] ** np.exp(linear_predictor))
    uncalibrated = min(max(float(uncalibrated), 1e-8), 1.0 - 1e-8)
    return float(
        1.0
        - np.exp(
            -np.exp(
                coeffs["scale_1"][region]
                + coeffs["scale_2"][region] * np.log(-np.log(1.0 - uncalibrated))
            )
        )
    )


def main():
    # Build the project-vs-published benchmark JSON file.
    model = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    # Use the clinically structured CVD dataset for published score comparisons.
    comparison_df = df[df["source_dataset"].astype("string").str.contains("CVD Dataset_processed.csv", case=False, na=False)].copy()
    comparison_df["sex"] = comparison_df["sex"].astype("string").str.strip().str.lower()
    comparison_df["smoking"] = pd.to_numeric(comparison_df["smoking"], errors="coerce")
    comparison_df["diabetes"] = pd.to_numeric(comparison_df["diabetes"], errors="coerce")
    comparison_df["age_years"] = pd.to_numeric(comparison_df["age_years"], errors="coerce")
    comparison_df["cholesterol"] = pd.to_numeric(comparison_df["cholesterol"], errors="coerce")
    comparison_df["hdl"] = pd.to_numeric(comparison_df["hdl"], errors="coerce")
    comparison_df["systolic_bp"] = pd.to_numeric(comparison_df["systolic_bp"], errors="coerce")

    framingham_df = comparison_df[
        comparison_df["age_years"].between(30, 74, inclusive="both")
        & comparison_df["sex"].isin(["male", "female"])
        & comparison_df["cholesterol"].gt(0)
        & comparison_df["hdl"].gt(0)
        & comparison_df["systolic_bp"].gt(0)
        & comparison_df["smoking"].isin([0, 1])
        & comparison_df["diabetes"].isin([0, 1])
    ].copy()

    score2_df = comparison_df[
        comparison_df["age_years"].between(40, 69, inclusive="both")
        & comparison_df["sex"].isin(["male", "female"])
        & comparison_df["cholesterol"].gt(0)
        & comparison_df["hdl"].gt(0)
        & comparison_df["systolic_bp"].gt(0)
        & comparison_df["smoking"].isin([0, 1])
        & comparison_df["diabetes"].isin([0, 1])
    ].copy()

    framingham_df["project_model_risk"] = model_predict_proba(framingham_df, model)
    framingham_df["framingham_risk"] = framingham_df.apply(framingham_risk, axis=1)

    score2_df["project_model_risk"] = model_predict_proba(score2_df, model)
    score2_df["score2_low_risk_region"] = score2_df.apply(lambda row: score2_risk(row, region="low"), axis=1)

    framingham_y = framingham_df["target_cad_risk"].astype(int).to_numpy()
    score2_y = score2_df["target_cad_risk"].astype(int).to_numpy()

    result = {
        "comparison_date": pd.Timestamp.now("UTC").isoformat(),
        "project_model": model["model_name"],
        "dataset_used_for_published_scores": "processed/CVD Dataset_processed.csv",
        "clinical_alignment_notes": [
            "Framingham and SCORE2 were evaluated only on rows with the variables those scores require.",
            "The local CVD comparison dataset uses a risk-level target rather than adjudicated 10-year events, so this is a directional benchmark rather than a formal external validation.",
            "Framingham was run with the lipid-based general CVD equation and assumes untreated systolic blood pressure because treatment status is unavailable.",
            "SCORE2 was run with the published low-risk-region recalibration and should be interpreted cautiously because the dataset is not a European cohort.",
        ],
        "framingham_comparison": {
            "eligibility": "Age 30-74, sex known, total cholesterol/HDL/SBP present, smoking and diabetes present",
            "project_model": summarize_probs(framingham_y, framingham_df["project_model_risk"].to_numpy(dtype=float)),
            "framingham_lipids": summarize_probs(framingham_y, framingham_df["framingham_risk"].to_numpy(dtype=float)),
        },
        "score2_comparison": {
            "eligibility": "Age 40-69, sex known, total cholesterol/HDL/SBP present, smoking and diabetes present",
            "project_model": summarize_probs(score2_y, score2_df["project_model_risk"].to_numpy(dtype=float)),
            "score2_low_risk_region": summarize_probs(score2_y, score2_df["score2_low_risk_region"].to_numpy(dtype=float)),
        },
        "source_references": {
            "framingham": "https://www.framinghamheartstudy.org/fhs-risk-functions/cardiovascular-disease-10-year-risk/",
            "score2": "https://academic.oup.com/eurheartj/article/43/3/241/6433491",
        },
    }

    OUTPUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
