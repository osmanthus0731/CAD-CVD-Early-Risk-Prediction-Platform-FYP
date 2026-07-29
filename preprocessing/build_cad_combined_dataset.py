#!/usr/bin/env python3
"""
Build a combined CAD-risk dataset from cleaned CSV files.

Outputs:
- data/original/... (copies of original source CSVs)
- data/combined/cad_risk_training.csv
- data/combined/cad_risk_summary.json
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_GLOB_EXCLUDES = {"data", "__pycache__"}


def read_csv(path: Path):
    # Load a CSV as dictionaries so each source schema can be mapped by name.
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        rows = [dict(x) for x in r]
    return cols, rows


def to_float(x):
    # Convert a cell to float; unknown or invalid values become None.
    try:
        return float(str(x).strip())
    except Exception:
        return None


def to_int_bool(x):
    # Normalize many yes/no or low/high labels into 0/1 target values.
    s = str(x).strip().lower()
    if s in {"1", "yes", "y", "true", "high", "intermediary"}:
        return 1
    if s in {"0", "no", "n", "false", "low"}:
        return 0
    try:
        n = float(s)
        if n == 1.0:
            return 1
        if n == 0.0:
            return 0
    except Exception:
        pass
    return None


def parse_bp(bp_text):
    # Split a blood pressure string like "120/80" into two numbers.
    s = str(bp_text).strip()
    if "/" not in s:
        return None, None
    a, b = s.split("/", 1)
    return to_float(a), to_float(b)


def norm_sex(v):
    # Keep sex coding consistent across datasets.
    s = str(v).strip().lower()
    if s in {"m", "male", "1", "1.0"}:
        return "male"
    if s in {"f", "female", "0", "2", "2.0"}:
        return "female"
    return None


CANON_COLS = [
    "source_dataset",
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
    "target_cad_risk",
]


def map_rows(clean_csv: Path):
    # Convert one cleaned source file into the shared CAD training schema.
    cols, rows = read_csv(clean_csv)
    colset = set(cols)
    mapped = []
    target_rows = 0
    source_name = str(clean_csv.parent.name + "/" + clean_csv.name)

    # Exclude non-CVD targets from the combined risk model to keep labels clinically aligned.
    if "diabetes classification" in source_name.lower():
        return mapped, len(rows), target_rows

    for r in rows:
        m = {k: None for k in CANON_COLS}
        m["source_dataset"] = source_name

        if {"diagnosis", "tg", "hdl", "ldl"}.issubset(colset):
            # Blood-test style dataset.
            m["age_years"] = to_float(r.get("age"))
            m["sex"] = norm_sex(r.get("gender"))
            m["bmi"] = to_float(r.get("bmi"))
            m["cholesterol"] = to_float(r.get("chol"))
            m["hdl"] = to_float(r.get("hdl"))
            m["ldl"] = to_float(r.get("ldl"))
            m["triglycerides"] = to_float(r.get("tg"))
            m["target_cad_risk"] = to_int_bool(r.get("diagnosis"))

        elif "cvd_risk_level" in colset:
            # Main CVD dataset with risk level already included.
            m["age_years"] = to_float(r.get("age"))
            m["sex"] = norm_sex(r.get("sex"))
            m["bmi"] = to_float(r.get("bmi"))
            m["systolic_bp"] = to_float(r.get("systolic_bp"))
            m["diastolic_bp"] = to_float(r.get("diastolic_bp"))
            m["cholesterol"] = to_float(r.get("total_cholesterol_mg_dl"))
            m["hdl"] = to_float(r.get("hdl_mg_dl"))
            m["ldl"] = to_float(r.get("estimated_ldl_mg_dl"))
            m["fasting_glucose"] = to_float(r.get("fasting_blood_sugar_mg_dl"))
            m["smoking"] = to_int_bool(r.get("smoking_status"))
            m["diabetes"] = to_int_bool(r.get("diabetes_status"))
            m["family_history_cvd"] = to_int_bool(r.get("family_history_of_cvd"))
            m["physical_activity"] = r.get("physical_activity_level")
            m["target_cad_risk"] = to_int_bool(r.get("cvd_risk_level"))

        elif "heart_attack_risk" in colset:
            # Heart attack risk dataset with BP stored as one text field.
            sbp, dbp = parse_bp(r.get("blood_pressure"))
            m["age_years"] = to_float(r.get("age"))
            m["sex"] = norm_sex(r.get("sex"))
            m["bmi"] = to_float(r.get("bmi"))
            m["systolic_bp"] = sbp
            m["diastolic_bp"] = dbp
            m["cholesterol"] = to_float(r.get("cholesterol"))
            m["triglycerides"] = to_float(r.get("triglycerides"))
            m["smoking"] = to_int_bool(r.get("smoking"))
            m["diabetes"] = to_int_bool(r.get("diabetes"))
            m["family_history_cvd"] = to_int_bool(r.get("family_history"))
            m["physical_activity"] = r.get("exercise_hours_per_week")
            m["target_cad_risk"] = to_int_bool(r.get("heart_attack_risk"))

        elif {"target", "trestbps", "chol"}.issubset(colset):
            # Common heart disease dataset format.
            m["age_years"] = to_float(r.get("age"))
            m["sex"] = norm_sex(r.get("sex"))
            m["systolic_bp"] = to_float(r.get("trestbps"))
            m["cholesterol"] = to_float(r.get("chol"))
            m["diabetes"] = to_int_bool(r.get("fbs"))
            m["physical_activity"] = r.get("thalach")
            m["target_cad_risk"] = to_int_bool(r.get("target"))

        elif {"cardio", "ap_hi", "ap_lo"}.issubset(colset):
            # Cardiovascular disease dataset with age in days and ordinal cholesterol/glucose.
            age_days = to_float(r.get("age"))
            m["age_years"] = (age_days / 365.25) if age_days is not None else None
            m["sex"] = norm_sex(r.get("gender"))
            m["bmi"] = None
            h = to_float(r.get("height"))
            w = to_float(r.get("weight"))
            if h and h > 0 and w is not None:
                hm = h / 100.0
                m["bmi"] = w / (hm * hm)
            m["systolic_bp"] = to_float(r.get("ap_hi"))
            m["diastolic_bp"] = to_float(r.get("ap_lo"))
            m["cholesterol"] = to_float(r.get("cholesterol"))  # ordinal in this dataset
            m["fasting_glucose"] = to_float(r.get("gluc"))     # ordinal in this dataset
            m["smoking"] = to_int_bool(r.get("smoke"))
            m["physical_activity"] = r.get("active")
            m["target_cad_risk"] = to_int_bool(r.get("cardio"))

        else:
            # Unknown schema or no CAD target; skip for supervised CAD training.
            continue

        if m["target_cad_risk"] is not None:
            target_rows += 1
        mapped.append(m)

    return mapped, len(rows), target_rows


def copy_original_csvs():
    # Preserve a copy of raw source files before generated folders are created.
    out_root = ROOT / "data" / "original"
    out_root.mkdir(parents=True, exist_ok=True)

    copied = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in RAW_GLOB_EXCLUDES for part in p.parts):
            continue
        if p.suffix.lower() not in {".csv", ".xlsx", ".xlsm", ".xls"}:
            continue
        rel = p.relative_to(ROOT)
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        copied.append((str(p), str(dest)))
    return copied


def main():
    # Build the final supervised training CSV used by the model scripts.
    copied = copy_original_csvs()

    processed_root = ROOT / "data" / "processed"
    clean_root = ROOT / "data" / "clean"
    if processed_root.exists():
        clean_files = sorted(processed_root.rglob("*_processed.csv"))
    else:
        clean_files = sorted(clean_root.rglob("*_clean.csv"))
    all_rows = []
    source_stats = {}

    for f in clean_files:
        mapped_rows, source_total, source_with_target = map_rows(f)
        all_rows.extend(mapped_rows)
        source_stats[str(f)] = {
            "rows_in_source_clean_file": source_total,
            "rows_mapped": len(mapped_rows),
            "rows_with_target": source_with_target,
        }

    # Keep only rows with known target for CAD training.
    train_rows = [r for r in all_rows if r["target_cad_risk"] in {0, 1}]

    combined_dir = ROOT / "data" / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    out_csv = combined_dir / "cad_risk_training.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CANON_COLS)
        w.writeheader()
        w.writerows(train_rows)

    summary = {
        "original_files_copied": len(copied),
        "clean_files_seen": len(clean_files),
        "combined_rows_total_mapped": len(all_rows),
        "combined_rows_with_target": len(train_rows),
        "target_distribution": {
            "0": sum(1 for r in train_rows if r["target_cad_risk"] == 0),
            "1": sum(1 for r in train_rows if r["target_cad_risk"] == 1),
        },
        "source_stats": source_stats,
        "output_csv": str(out_csv),
    }
    (combined_dir / "cad_risk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Original files copied to: {ROOT / 'data' / 'original'}")
    print(f"Combined CAD dataset: {out_csv}")
    print(f"Summary: {combined_dir / 'cad_risk_summary.json'}")


if __name__ == "__main__":
    main()
