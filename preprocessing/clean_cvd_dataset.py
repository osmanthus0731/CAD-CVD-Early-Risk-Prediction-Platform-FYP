"""
Clean and standardize the CVD dataset.

Usage:
  python preprocessing/clean_cvd_dataset.py --input "data/CVD Dataset.csv" --output "data/clean/CVD Dataset_clean.csv"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


RENAME_MAP = {
    "Sex": "sex",
    "Age": "age",
    "Weight (kg)": "weight_kg",
    "Height (m)": "height_m",
    "BMI": "bmi",
    "Abdominal Circumference (cm)": "abdominal_circumference_cm",
    "Blood Pressure (mmHg)": "blood_pressure_mmhg",
    "Total Cholesterol (mg/dL)": "total_cholesterol_mg_dl",
    "HDL (mg/dL)": "hdl_mg_dl",
    "Fasting Blood Sugar (mg/dL)": "fasting_blood_sugar_mg_dl",
    "Smoking Status": "smoking_status",
    "Diabetes Status": "diabetes_status",
    "Physical Activity Level": "physical_activity_level",
    "Family History of CVD": "family_history_cvd",
    "CVD Risk Level": "cvd_risk_level",
    "Height (cm)": "height_cm",
    "Waist-to-Height Ratio": "waist_to_height_ratio",
    "Systolic BP": "systolic_bp",
    "Diastolic BP": "diastolic_bp",
    "Blood Pressure Category": "blood_pressure_category",
    "Estimated LDL (mg/dL)": "estimated_ldl_mg_dl",
    "CVD Risk Score": "cvd_risk_score",
}

NUMERIC_COLS = [
    "age",
    "weight_kg",
    "height_m",
    "bmi",
    "abdominal_circumference_cm",
    "total_cholesterol_mg_dl",
    "hdl_mg_dl",
    "fasting_blood_sugar_mg_dl",
    "height_cm",
    "waist_to_height_ratio",
    "systolic_bp",
    "diastolic_bp",
    "estimated_ldl_mg_dl",
    "cvd_risk_score",
]

CATEGORICAL_COLS = [
    "sex",
    "smoking_status",
    "diabetes_status",
    "physical_activity_level",
    "family_history_cvd",
    "cvd_risk_level",
    "blood_pressure_category",
]


def parse_args() -> argparse.Namespace:
    # Read file paths from the command line.
    parser = argparse.ArgumentParser(description="Clean CVD dataset")
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--output", required=True, help="Path to cleaned CSV")
    parser.add_argument(
        "--report",
        default="cleaning_report.json",
        help="Where to save cleaning report JSON",
    )
    return parser.parse_args()


def normalize_yes_no(series: pd.Series) -> pd.Series:
    # Make all yes/no variants look the same.
    mapping = {
        "Y": "yes",
        "N": "no",
        "YES": "yes",
        "NO": "no",
        "1": "yes",
        "0": "no",
        "TRUE": "yes",
        "FALSE": "no",
    }
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace(mapping)
        .str.lower()
        .replace({"<NA>": np.nan})
    )


def normalize_sex(series: pd.Series) -> pd.Series:
    # Convert M/F/male/female variants into one consistent label.
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace({"M": "male", "F": "female", "MALE": "male", "FEMALE": "female"})
        .str.lower()
        .replace({"<NA>": np.nan})
    )


def normalize_text_label(series: pd.Series) -> pd.Series:
    # Trim labels and collapse double spaces so categories group correctly.
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
        .replace({"<NA>": np.nan})
    )


def parse_bp_pair(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    # Extract systolic and diastolic numbers from strings like "120/80".
    bp = series.astype("string").str.strip()
    extracted = bp.str.extract(r"^\s*(\d{2,3})\s*/\s*(\d{2,3})\s*$")
    systolic = pd.to_numeric(extracted[0], errors="coerce")
    diastolic = pd.to_numeric(extracted[1], errors="coerce")
    return systolic, diastolic


def derive_bp_category(systolic: pd.Series, diastolic: pd.Series) -> pd.Series:
    # Create an easy-to-read BP category from the two BP numbers.
    cond_normal = (systolic < 120) & (diastolic < 80)
    cond_elevated = (systolic.between(120, 129, inclusive="both")) & (diastolic < 80)
    cond_stage1 = (systolic.between(130, 139, inclusive="both")) | (
        diastolic.between(80, 89, inclusive="both")
    )
    cond_stage2 = (systolic >= 140) | (diastolic >= 90)

    result = pd.Series(np.nan, index=systolic.index, dtype="object")
    result[cond_normal] = "normal"
    result[cond_elevated] = "elevated"
    result[cond_stage1] = "hypertension stage 1"
    result[cond_stage2] = "hypertension stage 2"
    return result

def iqr_clip(df: pd.DataFrame, numeric_cols: list[str], factor: float = 1.5) -> dict[str, int]:
    # Cap numeric outliers and count how many values were adjusted.
    clipped_counts: dict[str, int] = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = df[col]
        if series.dropna().empty:
            clipped_counts[col] = 0
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            clipped_counts[col] = 0
            continue
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        before = series.copy()
        df[col] = series.clip(lower=lower, upper=upper)
        clipped_counts[col] = int((before != df[col]).fillna(False).sum())
    return clipped_counts


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Track before/after quality metrics so you can prove preprocessing impact.
    report: dict = {}
    report["rows_before"] = int(df.shape[0])
    report["cols_before"] = int(df.shape[1])
    report["missing_before"] = df.isna().sum().to_dict()

    # 1) Standardize column names.
    # Useful because model pipelines break when headers are inconsistent
    # across files (spaces, units, case differences, special symbols).
    df = df.rename(columns=RENAME_MAP).copy()
    df.columns = [re.sub(r"[^a-z0-9_]+", "_", c.strip().lower()) for c in df.columns]

    # 2) Coerce expected numeric fields to numeric dtype.
    # Useful because invalid numeric text should become NaN early, so
    # missing-value handling is deterministic.
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3) Parse blood-pressure pair text ("SYS/DIA") into numeric columns.
    # Useful because models need separate systolic/diastolic features.
    if "blood_pressure_mmhg" in df.columns:
        bp_s, bp_d = parse_bp_pair(df["blood_pressure_mmhg"])
        if "systolic_bp" in df.columns:
            df["systolic_bp"] = df["systolic_bp"].fillna(bp_s)
        else:
            df["systolic_bp"] = bp_s
        if "diastolic_bp" in df.columns:
            df["diastolic_bp"] = df["diastolic_bp"].fillna(bp_d)
        else:
            df["diastolic_bp"] = bp_d

    # 4) Reconcile duplicate representations and derived health features.
    # Useful because mixed units (cm vs m) and missing derived values can
    # create contradictory records and weaken model signal.
    if "height_cm" in df.columns and "height_m" in df.columns:
        # Fill either side where possible, then force consistency.
        df["height_cm"] = df["height_cm"].fillna(df["height_m"] * 100.0)
        df["height_m"] = df["height_m"].fillna(df["height_cm"] / 100.0)
        df["height_m"] = df["height_cm"] / 100.0

    if {"weight_kg", "height_m"}.issubset(df.columns):
        valid = (df["weight_kg"] > 0) & (df["height_m"] > 0)
        bmi_from_formula = df["weight_kg"] / (df["height_m"] ** 2)
        if "bmi" in df.columns:
            df.loc[valid, "bmi"] = df.loc[valid, "bmi"].fillna(bmi_from_formula[valid])
            # Replace implausible BMI entries with computed value.
            bad_bmi = (df["bmi"] < 10) | (df["bmi"] > 70)
            df.loc[bad_bmi & valid, "bmi"] = bmi_from_formula[bad_bmi & valid]
        else:
            df["bmi"] = bmi_from_formula

    if {"abdominal_circumference_cm", "height_cm"}.issubset(df.columns):
        ratio = df["abdominal_circumference_cm"] / df["height_cm"]
        if "waist_to_height_ratio" in df.columns:
            df["waist_to_height_ratio"] = df["waist_to_height_ratio"].fillna(ratio)
        else:
            df["waist_to_height_ratio"] = ratio

    # 5) Normalize categorical labels.
    # Useful because "Y/Yes/1" should be treated as one class, not three.
    if "sex" in df.columns:
        df["sex"] = normalize_sex(df["sex"])
    for col in ("smoking_status", "diabetes_status", "family_history_cvd"):
        if col in df.columns:
            df[col] = normalize_yes_no(df[col])
    for col in ("physical_activity_level", "cvd_risk_level", "blood_pressure_category"):
        if col in df.columns:
            df[col] = normalize_text_label(df[col])

    # 6) Backfill BP risk category from systolic/diastolic values.
    # Useful when category labels are missing but raw BP measurements exist.
    if {"systolic_bp", "diastolic_bp"}.issubset(df.columns):
        derived = derive_bp_category(df["systolic_bp"], df["diastolic_bp"])
        if "blood_pressure_category" in df.columns:
            df["blood_pressure_category"] = df["blood_pressure_category"].fillna(derived)
        else:
            df["blood_pressure_category"] = derived

    # 7) Mark impossible values as missing.
    # Useful because negative LDL or impossible ages should not be treated as
    # valid training signal.
    if "estimated_ldl_mg_dl" in df.columns:
        df.loc[df["estimated_ldl_mg_dl"] < 0, "estimated_ldl_mg_dl"] = np.nan
    if "age" in df.columns:
        df.loc[(df["age"] < 0) | (df["age"] > 120), "age"] = np.nan

    # 8) Impute missing values.
    # Useful because many ML algorithms cannot train with NaN.
    # Median is robust for numeric skew; mode keeps categorical consistency.
    for col in NUMERIC_COLS:
        if col in df.columns:
            med = df[col].median(skipna=True)
            df[col] = df[col].fillna(med)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            mode = df[col].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "unknown"
            df[col] = df[col].fillna(fill)

    # 9) Clip extreme numeric outliers using IQR bounds.
    # Useful to reduce domination of extreme values without deleting rows.
    clipped_counts = iqr_clip(df, [c for c in NUMERIC_COLS if c in df.columns], factor=1.5)

    # 10) Remove exact duplicate rows.
    # Useful to prevent repeated records from biasing model learning.
    rows_before_dedup = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)

    report["rows_after"] = int(df.shape[0])
    report["cols_after"] = int(df.shape[1])
    report["duplicates_removed"] = int(rows_before_dedup - df.shape[0])
    report["missing_after"] = df.isna().sum().to_dict()
    report["iqr_clipped_counts"] = clipped_counts

    return df, report


def main() -> None:
    # Load raw data, clean it, save the cleaned CSV and a JSON report.
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    cleaned_df, report = clean_dataset(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Cleaned dataset saved to: {output_path}")
    print(f"Cleaning report saved to: {report_path}")
    print(f"Rows: {report['rows_before']} -> {report['rows_after']}")


if __name__ == "__main__":
    main()
