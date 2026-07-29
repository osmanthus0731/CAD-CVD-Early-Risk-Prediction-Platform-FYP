#!/usr/bin/env python3
"""
Generate 3 documentation files for CVD preprocessing:
1) original profile
2) preprocessing steps + justification + outcomes
3) result profile with explicit original-vs-result differences
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "CVD Dataset.csv"
CLEAN_PATH = ROOT / "data" / "clean" / "CVD Dataset_clean.csv"
OUT_DIR = ROOT / "data" / "docs"


def read_csv(path: Path):
    # Read CSV data using utf-8-sig to handle files saved from Excel. (utf - 8 sig is used to handle byte order mark from excel to identify 0XEF, OxBB, 0xBF most common www byte)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        cols = reader.fieldnames or []
    return cols, rows


def is_number(v: str) -> bool:
    # Check whether a string can safely be treated as numeric.
    if v is None:
        return False
    s = str(v).strip()
    if s == "":
        return False
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def profile(cols, rows):
    # Summarize a dataset before/after cleaning for documentation.
    missing = {c: 0 for c in cols}
    numeric_cols = []
    categorical_cols = []
    numeric_stats = {}
    categorical_top = {}

    for c in cols:
        vals = [(r.get(c) or "").strip() for r in rows]
        for v in vals:
            if v == "":
                missing[c] += 1

        non_blank = [v for v in vals if v != ""]
        numeric_count = sum(1 for v in non_blank if is_number(v))
        if non_blank and numeric_count / len(non_blank) >= 0.9:
            numeric_cols.append(c)
            nums = [float(v.replace(",", "")) for v in non_blank if is_number(v)]
            nums_sorted = sorted(nums)
            numeric_stats[c] = {
                "count": len(nums),
                "min": nums_sorted[0] if nums_sorted else None,
                "median": median(nums) if nums else None,
                "max": nums_sorted[-1] if nums_sorted else None,
            }
        else:
            categorical_cols.append(c)
            top = Counter(non_blank).most_common(5)
            categorical_top[c] = top

    duplicate_rows = 0
    seen = set()
    for r in rows:
        k = tuple(r.get(c, "") for c in cols)
        if k in seen:
            duplicate_rows += 1
        else:
            seen.add(k)

    return {
        "row_count": len(rows),
        "column_count": len(cols),
        "columns": cols,
        "missing_cells_total": sum(missing.values()),
        "missing_per_column": missing,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "numeric_stats": numeric_stats,
        "categorical_top_values": categorical_top,
        "duplicate_rows_exact": duplicate_rows,
    }


def build_preprocess_markdown(raw_prof, clean_prof):
    # Write a short, readable explanation of what preprocessing changed.
    before_missing = raw_prof["missing_cells_total"]
    after_missing = clean_prof["missing_cells_total"]
    before_rows = raw_prof["row_count"]
    after_rows = clean_prof["row_count"]
    before_cols = raw_prof["column_count"]
    after_cols = clean_prof["column_count"]

    lines = []
    lines.append("# CVD Dataset Preprocessing Steps")
    lines.append("")
    lines.append("## 1) Column Standardization")
    lines.append("- Reasoning: ML code is safer and more consistent with machine-friendly column names.")
    lines.append("- Action: Renamed headers to snake_case, removed spaces/special characters.")
    lines.append(f"- Outcome: columns remained {before_cols} -> {after_cols}, but names are standardized.")
    lines.append("")
    lines.append("## 2) Missing Value Handling")
    lines.append("- Reasoning: Most ML algorithms cannot train with missing values.")
    lines.append("- Action: Numeric columns imputed with median; categorical columns imputed with mode.")
    lines.append(f"- Outcome: missing cells reduced {before_missing} -> {after_missing}.")
    lines.append("")
    lines.append("## 3) Duplicate Handling")
    lines.append("- Reasoning: Duplicates can bias model training and metrics.")
    lines.append("- Action: Dropped exact duplicate rows.")
    lines.append(f"- Outcome: rows changed {before_rows} -> {after_rows}.")
    lines.append("")
    lines.append("## 4) Outlier Treatment")
    lines.append("- Reasoning: Extreme values can distort model training.")
    lines.append("- Action: Applied IQR clipping to numeric columns.")
    lines.append("- Outcome: extreme values are capped to robust bounds.")
    lines.append("")
    lines.append("## 5) Original vs Result Summary")
    lines.append(f"- Rows: {before_rows} -> {after_rows}")
    lines.append(f"- Columns: {before_cols} -> {after_cols}")
    lines.append(f"- Missing cells: {before_missing} -> {after_missing}")
    lines.append(f"- Numeric columns: {len(raw_prof['numeric_columns'])} -> {len(clean_prof['numeric_columns'])}")
    lines.append(f"- Categorical columns: {len(raw_prof['categorical_columns'])} -> {len(clean_prof['categorical_columns'])}")
    return "\n".join(lines) + "\n"


def main():
    # Create the three documentation files used in the project report.
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw file: {RAW_PATH}")
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing cleaned file: {CLEAN_PATH}")

    raw_cols, raw_rows = read_csv(RAW_PATH)
    clean_cols, clean_rows = read_csv(CLEAN_PATH)

    raw_profile = profile(raw_cols, raw_rows)
    clean_profile = profile(clean_cols, clean_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    original_profile_path = OUT_DIR / "cvd_01_original_profile.json"
    preprocess_steps_path = OUT_DIR / "cvd_02_preprocess_steps.md"
    result_profile_path = OUT_DIR / "cvd_03_result_profile.json"

    original_profile_path.write_text(json.dumps(raw_profile, indent=2), encoding="utf-8")
    preprocess_steps_path.write_text(
        build_preprocess_markdown(raw_profile, clean_profile),
        encoding="utf-8",
    )

    result_payload = {
        "result_profile": clean_profile,
        "difference_from_original": {
            "rows_before": raw_profile["row_count"],
            "rows_after": clean_profile["row_count"],
            "columns_before": raw_profile["column_count"],
            "columns_after": clean_profile["column_count"],
            "missing_before": raw_profile["missing_cells_total"],
            "missing_after": clean_profile["missing_cells_total"],
            "duplicates_before": raw_profile["duplicate_rows_exact"],
            "duplicates_after": clean_profile["duplicate_rows_exact"],
        },
    }
    result_profile_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    print(f"Created: {original_profile_path}")
    print(f"Created: {preprocess_steps_path}")
    print(f"Created: {result_profile_path}")


if __name__ == "__main__":
    main()
