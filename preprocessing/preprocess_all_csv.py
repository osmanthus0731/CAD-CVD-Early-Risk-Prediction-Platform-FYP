#!/usr/bin/env python3
#
#Dependency-free CSV preprocessing for multiple datasets.

#What it does:
# Finds all CSV files under a root folder
# Standardizes column names to snake_case
# Trims whitespace, converts blank strings to missing
# Detects numeric columns and imputes missing values with median
# Imputes categorical missing values with mode
# Removes exact duplicate rows
# Saves cleaned CSVs + JSON reports


from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def snake_case(name: str) -> str:
    # Make every header safe to use in code, for example "Blood Pressure" -> "blood_pressure".
    n = name.strip().lower()
    n = re.sub(r"[^a-z0-9]+", "_", n)
    n = re.sub(r"_+", "_", n).strip("_")
    return n or "col"


def parse_num(value: str) -> float | None:
    # Convert text into a number only when it is really numeric.
    if value is None:
        return None
    v = str(value).strip()
    if v == "":
        return None
    v = v.replace(",", "")
    try:
        num = float(v)
    except ValueError:
        return None
    if math.isfinite(num):
        return num
    return None


def qtile(sorted_vals: list[float], q: float) -> float:
    # Small helper for quartiles without needing pandas or numpy.
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def detect_numeric(rows: list[dict[str, str]], cols: list[str], threshold: float = 0.9) -> set[str]:
    # Treat a column as numeric when most non-empty values can be parsed as numbers.
    numeric_cols: set[str] = set()
    for c in cols:
        non_blank = 0
        numeric = 0
        for r in rows:
            val = (r.get(c) or "").strip()
            if val == "":
                continue
            non_blank += 1
            if parse_num(val) is not None:
                numeric += 1
        if non_blank > 0 and (numeric / non_blank) >= threshold:
            numeric_cols.add(c)
    return numeric_cols


def clean_one_csv(src: Path, out_csv: Path, out_report: Path) -> dict[str, Any]:
    # Read one CSV, clean it, save the cleaned file, and write a report beside it.
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delim = dialect.delimiter
        except csv.Error:
            delim = ","
        reader = csv.DictReader(f, delimiter=delim)
        original_cols = reader.fieldnames or []
        rows = [dict(r) for r in reader]

    col_map = {c: snake_case(c) for c in original_cols}
    used = set()
    for old, new in list(col_map.items()):
        # If two messy headers become the same snake_case name, add _2, _3, etc.
        base = new
        i = 2
        while new in used:
            new = f"{base}_{i}"
            i += 1
        col_map[old] = new
        used.add(new)
    cols = [col_map[c] for c in original_cols]

    normalized_rows: list[dict[str, str]] = []
    for r in rows:
        # Trim all cell values first so missing checks are consistent later.
        nr: dict[str, str] = {}
        for old in original_cols:
            val = r.get(old, "")
            val = "" if val is None else str(val).strip()
            nr[col_map[old]] = val
        normalized_rows.append(nr)

    missing_before = {c: 0 for c in cols}
    for r in normalized_rows:
        for c in cols:
            if r[c] == "":
                missing_before[c] += 1

    numeric_cols = detect_numeric(normalized_rows, cols, threshold=0.9)
    categorical_cols = [c for c in cols if c not in numeric_cols]

    # Numeric blanks get medians; text blanks get the most common value.
    numeric_medians: dict[str, float] = {}
    for c in numeric_cols:
        vals = [parse_num(r[c]) for r in normalized_rows]
        nn = [v for v in vals if v is not None]
        numeric_medians[c] = median(nn) if nn else 0.0

    categorical_modes: dict[str, str] = {}
    for c in categorical_cols:
        items = [r[c] for r in normalized_rows if r[c] != ""]
        if items:
            mode_val = Counter(items).most_common(1)[0][0]
        else:
            mode_val = "unknown"
        categorical_modes[c] = mode_val

    for r in normalized_rows:
        for c in cols:
            if r[c] == "":
                if c in numeric_cols:
                    mv = numeric_medians[c]
                    r[c] = str(int(mv)) if float(mv).is_integer() else f"{mv:.6g}"
                else:
                    r[c] = categorical_modes[c]

    clipped_counts: dict[str, int] = {}
    for c in numeric_cols:
        # Cap extreme numeric values using IQR, instead of deleting the whole row.
        vals = [parse_num(r[c]) for r in normalized_rows]
        nn = sorted([v for v in vals if v is not None])
        if len(nn) < 4:
            clipped_counts[c] = 0
            continue
        q1 = qtile(nn, 0.25)
        q3 = qtile(nn, 0.75)
        iqr = q3 - q1
        if iqr <= 0:
            clipped_counts[c] = 0
            continue
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        changed = 0
        for r in normalized_rows:
            v = parse_num(r[c])
            if v is None:
                continue
            nv = min(max(v, lo), hi)
            if nv != v:
                changed += 1
                r[c] = str(int(nv)) if float(nv).is_integer() else f"{nv:.6g}"
        clipped_counts[c] = changed

    seen = set()
    dedup_rows: list[dict[str, str]] = []
    for r in normalized_rows:
        # Remove exact duplicate rows after cleaning.
        key = tuple(r[c] for c in cols)
        if key in seen:
            continue
        seen.add(key)
        dedup_rows.append(r)

    missing_after = {c: 0 for c in cols}
    for r in dedup_rows:
        for c in cols:
            if r[c] == "":
                missing_after[c] += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(dedup_rows)

    report = {
        "source_file": str(src),
        "output_file": str(out_csv),
        "delimiter_detected": delim,
        "rows_before": len(rows),
        "rows_after": len(dedup_rows),
        "duplicates_removed": len(normalized_rows) - len(dedup_rows),
        "columns_before": original_cols,
        "columns_after": cols,
        "numeric_columns": sorted(list(numeric_cols)),
        "categorical_columns": sorted(categorical_cols),
        "missing_before": missing_before,
        "missing_after": missing_after,
        "iqr_clipped_counts": clipped_counts,
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    # Command-line entry point so the script can be reused on any folder.
    parser = argparse.ArgumentParser(description="Preprocess all CSV files under a root folder")
    parser.add_argument("--input-root", default=str(ROOT), help="Root folder to search for CSV files")
    parser.add_argument("--output-root", default=str(ROOT / "data" / "clean"), help="Output folder for cleaned files")
    parser.add_argument("--reports-root", default=str(ROOT / "data" / "reports"), help="Output folder for reports")
    args = parser.parse_args()

    root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    reports_root = Path(args.reports_root).resolve()

    csv_paths = sorted([p for p in root.rglob("*.csv") if output_root not in p.parents])
    all_reports: list[dict[str, Any]] = []

    for src in csv_paths:
        # Keep the same folder structure under data/clean and data/reports.
        rel = src.relative_to(root)
        cleaned_name = f"{src.stem}_clean.csv"
        report_name = f"{src.stem}_report.json"
        out_csv = output_root / rel.parent / cleaned_name
        out_report = reports_root / rel.parent / report_name
        report = clean_one_csv(src, out_csv, out_report)
        all_reports.append(report)
        print(f"[OK] {src} -> {out_csv}")

    summary = {
        "processed_files": len(all_reports),
        "files": all_reports,
    }
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary saved to: {reports_root / 'summary.json'}")


if __name__ == "__main__":
    main()
