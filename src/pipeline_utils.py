"""
pipeline_utils.py
Reusable functions for extracting fields and computing risk features for a
SINGLE new invoice (as opposed to the batch scripts, which process the full
training set at once). Used by the Streamlit app for live inference.
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import pdfplumber

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from duplicate_matching import find_closest_match

PATTERNS = {
    "invoice_id": r"Invoice Number:\s*(INV-\d+)",
    "vendor": r"Vendor:\s*(.+)",
    "posting_date": r"Posting Date:\s*([\d-]+)",
    "due_date": r"Due Date:\s*([\d-]+)",
    "amount": r"Amount Due:\s*\$([\d,]+\.\d{2})",
}

FEATURE_COLS = [
    "amount_zscore_norm", "threshold_proximity",
    "duplicate_score", "weekend_flag", "days_to_due_norm",
]


def extract_fields_from_pdf(file_obj):
    """Extract fields from a single PDF (file path or file-like object)."""
    with pdfplumber.open(file_obj) as pdf:
        text = pdf.pages[0].extract_text()

    record = {}
    for field, pattern in PATTERNS.items():
        match = re.search(pattern, text)
        record[field] = match.group(1).strip() if match else None

    if record.get("amount"):
        record["amount"] = float(record["amount"].replace(",", ""))

    return record


def compute_features_for_new_invoice(new_record, training_df):
    """
    Compute the 5 normalized risk features for ONE new invoice, using
    statistics (vendor mean/std, global min/max) derived from the training
    set.
    """
    vendor = new_record["vendor"]
    amount = new_record["amount"]
    posting_date = pd.to_datetime(new_record["posting_date"])
    due_date = pd.to_datetime(new_record["due_date"])

    vendor_rows = training_df[training_df["vendor"] == vendor]
    if len(vendor_rows) >= 2:
        v_mean, v_std = vendor_rows["amount"].mean(), vendor_rows["amount"].std()
    else:
        v_mean, v_std = training_df["amount"].mean(), training_df["amount"].std()
    v_std = v_std if v_std and v_std > 0 else 1.0
    raw_zscore = abs(amount - v_mean) / v_std

    train_zscores = training_df.groupby("vendor")["amount"].transform(
        lambda s: (s - s.mean()).abs() / (s.std() if s.std() > 0 else 1.0)
    )
    z_min, z_max = train_zscores.min(), max(train_zscores.max(), raw_zscore)
    amount_zscore_norm = (raw_zscore - z_min) / (z_max - z_min) if z_max > z_min else 0.0

    threshold_proximity = 1 - min(abs(10000 - amount), 10000) / 10000
    threshold_proximity = max(0.0, min(1.0, threshold_proximity))

    duplicate_match = find_closest_match(
        new_record, training_df, exclude_invoice_id=new_record.get("invoice_id")
    )
    duplicate_score = duplicate_match.score

    weekend_flag = 1.0 if posting_date.weekday() >= 5 else 0.0

    days_to_due = (due_date - posting_date).days
    train_days = (pd.to_datetime(training_df["due_date"]) - pd.to_datetime(training_df["posting_date"])).dt.days
    d_min, d_max = train_days.min(), max(train_days.max(), days_to_due)
    days_to_due_norm = (days_to_due - d_min) / (d_max - d_min) if d_max > d_min else 0.0

    return {
        "amount_zscore_norm": round(amount_zscore_norm, 4),
        "threshold_proximity": round(threshold_proximity, 4),
        "duplicate_score": round(duplicate_score, 4),
        "weekend_flag": weekend_flag,
        "days_to_due_norm": round(days_to_due_norm, 4),
    }


def assess_duplicate(new_record, training_df):
    """
    The closest invoice on file to this one, with its component similarities.

    Returns the same match `compute_features_for_new_invoice` scored, kept as a
    separate call so the feature vector stays five plain numbers while the
    reviewer-facing detail (what it matched, and how closely) is still available.
    """
    return find_closest_match(
        new_record, training_df, exclude_invoice_id=new_record.get("invoice_id")
    )
