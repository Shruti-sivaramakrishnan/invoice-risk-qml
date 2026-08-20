"""
dashboard.py
KPI computation for the approval workflow: touchless processing rate, the
flagged-vs-cleared trend across batch runs, false-positive rate tracking,
and per-vendor risk profiling.

Pure functions over DataFrames already loaded by the caller (app.py) or a
list of per-invoice result dicts — no Streamlit and no direct database
access, so the numbers here can be tested without a UI or a live ledger, the
same separation approval_rules.py and confidence.py already keep.
"""

import pandas as pd

from approval_rules import NEAR_DUPLICATE
from audit_store import STATUS_APPROVED, STATUS_REJECTED, vendor_risk_summary

DECIDED_STATUSES = (STATUS_APPROVED, STATUS_REJECTED)


def summarize_results(results):
    """
    Reduce one batch's per-invoice results — the dicts app.py builds while
    scoring a batch — to the counts both the batch-tab metrics and a
    run-history row need, so the two stay in sync by construction rather
    than by two copies of the same arithmetic.
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0, "auto_approved": 0, "routed": 0, "failed": 0,
            "high_band": 0, "split": 0, "duplicates": 0, "touchless_rate": None,
        }

    auto_approved = sum(1 for r in results if r.get("route") == "auto_approve")
    return {
        "total": total,
        "auto_approved": auto_approved,
        "routed": total - auto_approved,
        "failed": sum(1 for r in results if r.get("failed")),
        "high_band": sum(1 for r in results if r.get("band") == "high"),
        "split": sum(1 for r in results if r.get("split")),
        "duplicates": sum(
            1 for r in results if (r.get("duplicate_score") or 0) >= NEAR_DUPLICATE
        ),
        "touchless_rate": auto_approved / total,
    }


def touchless_rate(queue_df):
    """Cumulative auto-approval rate across every invoice ever submitted."""
    if queue_df is None or queue_df.empty:
        return None
    return float((queue_df["route"] == "auto_approve").mean())


def annotate_run_history(runs_df):
    """
    Add per-run and cumulative touchless rates to a batch_runs frame, in
    submission order, for the trend view. Cumulative rate is what actually
    answers "is automation improving" — a single run's rate is one data
    point, this is the line through them.
    """
    if runs_df is None or runs_df.empty:
        return runs_df

    df = runs_df.sort_values("started_at").reset_index(drop=True)
    df["flagged_count"] = df["invoice_count"] - df["auto_approved_count"]
    df["touchless_rate"] = df["auto_approved_count"] / df["invoice_count"].replace(0, pd.NA)

    cum_total = df["invoice_count"].cumsum()
    cum_auto = df["auto_approved_count"].cumsum()
    df["cumulative_touchless_rate"] = cum_auto / cum_total.replace(0, pd.NA)
    df["run_label"] = [f"Run {i + 1}" for i in range(len(df))]
    return df


def false_positive_summary(queue_df):
    """
    Operational false-positive rate: of the invoices the policy sent for
    human review that have since been decided, how many did the reviewer
    approve rather than reject — i.e. how often the flag turned out to be
    unnecessary.

    Ground truth isn't used here even though the demo dataset happens to have
    it. A production invoice never carries a label; the reviewer's own
    decision is the only signal a real deployment has, and that's what this
    tracks. `coverage` is carried alongside the rate because a rate built on
    a handful of decisions out of a large flagged backlog isn't trustworthy
    yet, and shouldn't be read as if it were.
    """
    if queue_df is None or queue_df.empty:
        return {"flagged": 0, "decided": 0, "false_positives": 0,
                "false_positive_rate": None, "coverage": None}

    flagged = queue_df[queue_df["route"] != "auto_approve"]
    decided = flagged[flagged["status"].isin(DECIDED_STATUSES)]
    false_positives = decided[decided["status"] == STATUS_APPROVED]

    return {
        "flagged": len(flagged),
        "decided": len(decided),
        "false_positives": len(false_positives),
        "false_positive_rate": (len(false_positives) / len(decided)) if len(decided) else None,
        "coverage": (len(decided) / len(flagged)) if len(flagged) else None,
    }


def false_positive_trend(queue_df):
    """
    False-positive rate grouped by decision date. Small-sample noise is real
    here — a handful of decisions on one day can swing the rate sharply — so
    each row carries its own decision count rather than hiding it behind a
    single trend line.
    """
    columns = ["date", "decided", "false_positives", "false_positive_rate"]
    if queue_df is None or queue_df.empty:
        return pd.DataFrame(columns=columns)

    flagged = queue_df[queue_df["route"] != "auto_approve"]
    decided = flagged[flagged["status"].isin(DECIDED_STATUSES)].copy()
    if decided.empty:
        return pd.DataFrame(columns=columns)

    decided["date"] = pd.to_datetime(decided["decided_at"]).dt.date.astype(str)
    grouped = decided.groupby("date").agg(
        decided=("status", "count"),
        false_positives=("status", lambda s: int((s == STATUS_APPROVED).sum())),
    ).reset_index()
    grouped["false_positive_rate"] = grouped["false_positives"] / grouped["decided"]
    return grouped.sort_values("date").reset_index(drop=True)
