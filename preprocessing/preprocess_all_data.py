#!/usr/bin/env python3
"""
Unified preprocessing for CSV and XLSX datasets.

Outputs:
- data/processed/... processed CSV files
- data/reports_all/... per-file JSON reports + summary
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {"data", "__pycache__", ".git"}


def snake_case(name: str) -> str:
    # Turn messy spreadsheet headers into predictable Python-friendly names.
    x = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", x).strip("_") or "col"


def normalize_blank_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    # Make blank-looking text values become real missing values.
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = (
                df[c]
                .astype("string")
                .str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA})
            )
    return df


def likely_numeric(series: pd.Series, threshold: float = 0.9) -> bool:
    # Some CSVs store numbers as text, so detect numeric columns by parsing.
    s = series.dropna().astype("string").str.replace(",", "", regex=False)
    if s.empty:
        return False
    parsed = pd.to_numeric(s, errors="coerce")
    return (parsed.notna().mean() >= threshold)


def find_unnecessary_columns(df: pd.DataFrame) -> list[str]:
    # Remove export/index columns and ID columns that do not help prediction.
    drop_cols: list[str] = []
    n = len(df)
    if n == 0:
        return drop_cols

    for c in df.columns:
        lc = c.lower()
        # Explicit noise/index columns from exports.
        if lc in {"unnamed_0", "unnamed", "index", "col"} or lc.startswith("unnamed"):
            drop_cols.append(c)
            continue

        # Drop ID-like columns with near-unique values.
        if any(k in lc for k in ["id", "patient_id", "record_id"]):
            nunique = df[c].nunique(dropna=True)
            if nunique >= int(0.98 * max(1, n)):
                drop_cols.append(c)

    return sorted(list(dict.fromkeys(drop_cols)))


def iqr_clip(df: pd.DataFrame, numeric_cols: list[str], factor: float = 1.5) -> dict[str, int]:
    # Limit extreme values using IQR bounds and record how many were changed.
    changed: dict[str, int] = {}
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce").astype("float64")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            changed[c] = 0
            continue
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        before = s.copy()
        s = s.clip(lower=lo, upper=hi)
        df[c] = s
        changed[c] = int((before != s).fillna(False).sum())
    return changed


def preprocess_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Main cleaning pipeline used for both CSV and Excel sheets.
    rep: dict = {}
    rep["rows_before"] = int(df.shape[0])
    rep["cols_before"] = int(df.shape[1])

    # Standardize headers and normalize blank values.
    df = df.copy()
    df.columns = [snake_case(c) for c in df.columns]
    df = normalize_blank_to_nan(df)

    # Drop fully empty columns first.
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    # Drop unnecessary columns explicitly.
    dropped_cols = find_unnecessary_columns(df)
    if dropped_cols:
        df = df.drop(columns=dropped_cols)

    # Detect and convert numeric-like columns.
    numeric_cols: list[str] = []
    for c in df.columns:
        if likely_numeric(df[c]):
            df[c] = pd.to_numeric(df[c].astype("string").str.replace(",", "", regex=False), errors="coerce")
            numeric_cols.append(c)

    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # Remove exact duplicates.
    before_dups = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed_dups = before_dups - len(df)

    # Impute missing.
    for c in numeric_cols:
        med = df[c].median(skipna=True)
        if pd.isna(med):
            med = 0.0
        df[c] = df[c].fillna(med)
    for c in categorical_cols:
        mode = df[c].mode(dropna=True)
        fill = mode.iloc[0] if not mode.empty else "unknown"
        df[c] = df[c].fillna(fill)

    # Clip outliers on numeric columns.
    clipped = iqr_clip(df, numeric_cols, factor=1.5)

    rep["rows_after"] = int(df.shape[0])
    rep["cols_after"] = int(df.shape[1])
    rep["duplicates_removed"] = int(removed_dups)
    rep["dropped_empty_columns"] = empty_cols
    rep["dropped_unnecessary_columns"] = dropped_cols
    rep["numeric_columns"] = numeric_cols
    rep["categorical_columns"] = categorical_cols
    rep["missing_after"] = int(df.isna().sum().sum())
    rep["iqr_clipped_counts"] = clipped

    return df, rep


def read_csv_any_delim(path: Path) -> pd.DataFrame:
    # sep=None lets pandas guess commas, semicolons, tabs, etc.
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")


def process_csv(path: Path, root: Path, out_root: Path, rep_root: Path) -> dict:
    # Process one CSV and save both the cleaned data and its report.
    df = read_csv_any_delim(path)
    cleaned, rep = preprocess_df(df)

    rel = path.relative_to(root)
    out = out_root / rel.parent / f"{path.stem}_processed.csv"
    rpt = rep_root / rel.parent / f"{path.stem}_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rpt.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out, index=False)
    rep["source_file"] = str(path)
    rep["output_file"] = str(out)
    rep["source_type"] = "csv"
    rpt.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


def process_xlsx(path: Path, root: Path, out_root: Path, rep_root: Path) -> list[dict]:
    # Process every sheet in an Excel workbook as its own cleaned CSV.
    reports: list[dict] = []
    book = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    rel = path.relative_to(root)

    for sheet_name, df in book.items():
        cleaned, rep = preprocess_df(df)
        safe_sheet = snake_case(sheet_name)
        out = out_root / rel.parent / f"{path.stem}__{safe_sheet}_processed.csv"
        rpt = rep_root / rel.parent / f"{path.stem}__{safe_sheet}_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        rpt.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(out, index=False)
        rep["source_file"] = str(path)
        rep["source_sheet"] = sheet_name
        rep["output_file"] = str(out)
        rep["source_type"] = "xlsx"
        rpt.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        reports.append(rep)
    return reports


def discover_files(root: Path) -> tuple[list[Path], list[Path]]:
    # Find raw data files while skipping generated data folders.
    csvs: list[Path] = []
    xlsxs: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.suffix.lower() == ".csv":
            csvs.append(p)
        elif p.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
            xlsxs.append(p)
    return sorted(csvs), sorted(xlsxs)


def main() -> None:
    # Run the full preprocessing workflow from the command line.
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default=str(ROOT))
    ap.add_argument("--output-root", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--reports-root", default=str(ROOT / "data" / "reports_all"))
    args = ap.parse_args()

    root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    rep_root = Path(args.reports_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rep_root.mkdir(parents=True, exist_ok=True)

    csvs, xlsxs = discover_files(root)
    all_reports: list[dict] = []

    for p in csvs:
        rep = process_csv(p, root, out_root, rep_root)
        all_reports.append(rep)
        print(f"[OK][CSV] {p}")

    for p in xlsxs:
        reps = process_xlsx(p, root, out_root, rep_root)
        all_reports.extend(reps)
        print(f"[OK][XLSX] {p} (sheets={len(reps)})")

    summary = {
        "files_csv_processed": len(csvs),
        "files_xlsx_processed": len(xlsxs),
        "outputs_generated": len(all_reports),
        "reports": all_reports,
    }
    (rep_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {rep_root / 'summary.json'}")


if __name__ == "__main__":
    main()
