"""
feature_engineering.py
Builds 5 risk features from extracted invoice fields, each normalized to
[0, 1] so they can be used as quantum rotation angles (angle embedding)
as well as classical model inputs.

Features:
  1. amount_zscore_norm   - how unusual the amount is vs. the vendor population
  2. threshold_proximity  - how close the amount sits to the $10,000 reporting threshold
  3. duplicate_score      - graded similarity to the closest other invoice on file
  4. weekend_flag         - was this invoice posted on a Sat/Sun
  5. days_to_due_norm     - normalized days between posting and due date
"""

import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from duplicate_matching import score_dataframe

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
GT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ground_truth.csv")
EXTRACTED_PATH = os.path.join(PROCESSED_DIR, "extracted_fields.csv")
OUT_PATH = os.path.join(PROCESSED_DIR, "features.csv")


def min_max_norm(series):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return series * 0
    return (series - lo) / (hi - lo)


OUT_MATCHES_PATH = os.path.join(PROCESSED_DIR, "duplicate_matches.csv")


def main():
    extracted = pd.read_csv(EXTRACTED_PATH)
    gt = pd.read_csv(GT_PATH)

    # merge extracted fields with ground-truth label (label used only for
    # evaluation later, never as a model input)
    df = extracted.merge(gt[["invoice_id", "label"]], on="invoice_id", how="left")

    df["posting_date"] = pd.to_datetime(df["posting_date"])
    df["due_date"] = pd.to_datetime(df["due_date"])

    # 1. Amount z-score (per-vendor), normalized to [0,1]
    df["vendor_mean"] = df.groupby("vendor")["amount"].transform("mean")
    df["vendor_std"] = df.groupby("vendor")["amount"].transform("std").replace(0, 1).fillna(1)
    df["amount_zscore"] = (df["amount"] - df["vendor_mean"]).abs() / df["vendor_std"]
    df["amount_zscore_norm"] = min_max_norm(df["amount_zscore"])

    # 2. Threshold proximity: closeness to $10,000 structuring threshold
    df["threshold_proximity"] = 1 - (10000 - df["amount"]).abs().clip(lower=0, upper=10000) / 10000
    df["threshold_proximity"] = df["threshold_proximity"].clip(0, 1)

    # 3. Duplicate score: graded similarity to the closest other invoice on file.
    #    Scored on the original date strings, before the datetime coercion above
    #    would matter, so the batch path and the live single-invoice path in
    #    pipeline_utils compute the same number for the same pair.
    df["duplicate_score"], df["duplicate_match_id"] = score_dataframe(df)

    # 4. Weekend posting flag
    df["weekend_flag"] = (df["posting_date"].dt.weekday >= 5).astype(float)

    # 5. Days to due, normalized
    df["days_to_due"] = (df["due_date"] - df["posting_date"]).dt.days
    df["days_to_due_norm"] = min_max_norm(df["days_to_due"])

    feature_cols = [
        "amount_zscore_norm",
        "threshold_proximity",
        "duplicate_score",
        "weekend_flag",
        "days_to_due_norm",
    ]

    out = df[["invoice_id"] + feature_cols + ["label"]]
    out.to_csv(OUT_PATH, index=False)

    # The closest match per invoice is kept alongside the features so a flagged
    # duplicate can be traced back to what it matched against.
    matches = df[["invoice_id", "duplicate_match_id", "duplicate_score"]].sort_values(
        "duplicate_score", ascending=False
    )
    matches.to_csv(OUT_MATCHES_PATH, index=False)

    print(f"Wrote {len(out)} feature rows -> {OUT_PATH}")
    print(out[feature_cols].describe().T[["min", "max", "mean"]])
    print(f"\nClass balance: {out['label'].value_counts().to_dict()}")

    flagged = matches[matches["duplicate_score"] >= 0.80]
    print(f"\nNear-duplicate pairs (score >= 0.80): {len(flagged)}")
    print(flagged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
