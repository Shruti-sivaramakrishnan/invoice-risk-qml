"""
test_reporting.py
Tests for the KPI dashboard's computations (dashboard.py) and the simulated
approver notification preview (notifications.py).

Run with:  python src/test_reporting.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import approval_rules as rules
import confidence
import dashboard
import notifications as notif
from audit_store import (
    STATUS_APPROVED, STATUS_AUTO_APPROVED, STATUS_PENDING, STATUS_REJECTED,
)

PASSED, FAILED = 0, 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# dashboard.summarize_results
# ---------------------------------------------------------------------------

def test_summarize_results():
    print("\nSUMMARIZE_RESULTS")

    check("empty batch summarizes to zeros",
          dashboard.summarize_results([]) == {
              "total": 0, "auto_approved": 0, "routed": 0, "failed": 0,
              "high_band": 0, "split": 0, "duplicates": 0, "touchless_rate": None,
          })

    results = [
        {"route": "auto_approve", "band": "low", "split": False, "duplicate_score": 0.1, "failed": False},
        {"route": "auto_approve", "band": "cleared", "split": False, "duplicate_score": 0.0, "failed": False},
        {"route": "manager_review", "band": "elevated", "split": True, "duplicate_score": 0.85, "failed": False},
        {"route": "controller_review", "band": "high", "split": False, "duplicate_score": 0.97, "failed": False},
        {"route": "manager_review", "band": "elevated", "split": False, "duplicate_score": 0.0, "failed": True},
    ]
    s = dashboard.summarize_results(results)
    check("total counts every result", s["total"] == 5, s)
    check("auto_approved counts the auto route only", s["auto_approved"] == 2, s)
    check("routed is everything else", s["routed"] == 3, s)
    check("failed is counted independently of route", s["failed"] == 1, s)
    check("high_band counts the high band", s["high_band"] == 1, s)
    check("split counts split decisions", s["split"] == 1, s)
    check("duplicates counts scores at/above the near-duplicate threshold",
          s["duplicates"] == 2, s)
    check("touchless_rate is auto_approved / total", abs(s["touchless_rate"] - 0.4) < 1e-9, s)

    missing_keys = [{"route": "auto_approve"}]
    check("missing optional keys default safely rather than raising",
          dashboard.summarize_results(missing_keys)["duplicates"] == 0)


# ---------------------------------------------------------------------------
# dashboard.touchless_rate / annotate_run_history
# ---------------------------------------------------------------------------

def test_touchless_and_run_history():
    print("\nTOUCHLESS RATE + RUN HISTORY")

    check("empty queue has no touchless rate", dashboard.touchless_rate(pd.DataFrame()) is None)
    check("empty queue has no touchless rate (None)", dashboard.touchless_rate(None) is None)

    queue = pd.DataFrame({"route": ["auto_approve", "auto_approve", "manager_review", "dual_control"]})
    rate = dashboard.touchless_rate(queue)
    check("touchless rate is auto-approved fraction of the whole queue",
          abs(rate - 0.5) < 1e-9, rate)

    check("empty run history returns as-is", dashboard.annotate_run_history(pd.DataFrame()).empty)
    check("None run history returns as-is", dashboard.annotate_run_history(None) is None)

    runs = pd.DataFrame([
        {"started_at": "2026-01-01T10:00:00", "invoice_count": 10, "auto_approved_count": 8},
        {"started_at": "2026-01-02T10:00:00", "invoice_count": 10, "auto_approved_count": 2},
    ])
    annotated = dashboard.annotate_run_history(runs)
    check("flagged_count is the complement of auto_approved", list(annotated["flagged_count"]) == [2, 8])
    check("per-run touchless rate is computed", list(annotated["touchless_rate"]) == [0.8, 0.2])
    check("cumulative rate blends both runs",
          abs(annotated["cumulative_touchless_rate"].iloc[-1] - 0.5) < 1e-9,
          annotated["cumulative_touchless_rate"].iloc[-1])
    check("runs are labeled in chronological order",
          list(annotated["run_label"]) == ["Run 1", "Run 2"])

    reversed_runs = pd.DataFrame([
        {"started_at": "2026-01-02T10:00:00", "invoice_count": 5, "auto_approved_count": 5},
        {"started_at": "2026-01-01T10:00:00", "invoice_count": 5, "auto_approved_count": 0},
    ])
    reannotated = dashboard.annotate_run_history(reversed_runs)
    check("history is sorted by start time regardless of input order",
          list(reannotated["auto_approved_count"]) == [0, 5])


# ---------------------------------------------------------------------------
# dashboard.false_positive_summary / false_positive_trend
# ---------------------------------------------------------------------------

def _queue_row(route, status, decided_at=None):
    return {"route": route, "status": status, "decided_at": decided_at}


def test_false_positive():
    print("\nFALSE-POSITIVE RATE")

    empty = dashboard.false_positive_summary(pd.DataFrame())
    check("empty queue reports zero with no rate", empty == {
        "flagged": 0, "decided": 0, "false_positives": 0,
        "false_positive_rate": None, "coverage": None,
    })

    queue = pd.DataFrame([
        _queue_row("auto_approve", STATUS_AUTO_APPROVED),                      # not flagged
        _queue_row("manager_review", STATUS_PENDING),                          # flagged, undecided
        _queue_row("manager_review", STATUS_APPROVED, "2026-01-01T09:00:00"),  # flagged, false positive
        _queue_row("controller_review", STATUS_APPROVED, "2026-01-01T11:00:00"),  # flagged, false positive
        _queue_row("controller_review", STATUS_REJECTED, "2026-01-02T09:00:00"),  # flagged, true positive
        _queue_row("dual_control", STATUS_REJECTED, "2026-01-02T10:00:00"),       # flagged, true positive
    ])
    summary = dashboard.false_positive_summary(queue)
    check("auto-approved invoices are excluded from the flagged count",
          summary["flagged"] == 5, summary)
    check("only decided invoices count toward the rate", summary["decided"] == 4, summary)
    check("false positives are the flagged-and-approved subset",
          summary["false_positives"] == 2, summary)
    check("rate is false positives over decided", summary["false_positive_rate"] == 0.5, summary)
    check("coverage is decided over flagged", summary["coverage"] == 0.8, summary)

    trend = dashboard.false_positive_trend(queue)
    check("trend has one row per decision date", len(trend) == 2, trend)
    check("trend is sorted by date", list(trend["date"]) == ["2026-01-01", "2026-01-02"])
    check("day 1 rate reflects both being false positives",
          trend.loc[trend["date"] == "2026-01-01", "false_positive_rate"].iloc[0] == 1.0, trend)
    check("day 2 rate reflects neither being a false positive",
          trend.loc[trend["date"] == "2026-01-02", "false_positive_rate"].iloc[0] == 0.0, trend)

    no_decisions = pd.DataFrame([_queue_row("manager_review", STATUS_PENDING)])
    check("no decisions yet yields an empty trend, not an error",
          dashboard.false_positive_trend(no_decisions).empty)
    check("no decisions yet yields no rate", dashboard.false_positive_summary(no_decisions)["decided"] == 0)


# ---------------------------------------------------------------------------
# notifications.build_notification
# ---------------------------------------------------------------------------

CLEAN_FEATURES = {
    "amount_zscore_norm": 0.10, "threshold_proximity": 0.05,
    "duplicate_score": 0.0, "weekend_flag": 0.0, "days_to_due_norm": 0.5,
}


def test_notification_auto_approve():
    print("\nNOTIFICATION — AUTO-APPROVE")

    clear = confidence.assess({"LogReg": 0.05, "RBF-SVM": 0.05, "Quantum": 0.05})
    decision = rules.evaluate(500.0, CLEAN_FEATURES, clear)
    check("a clean invoice actually auto-approves", decision.is_auto, decision.route)

    n = notif.build_notification("INV-1001", "Acme Ltd", 500.0, decision)
    check("auto-approved invoice is not actionable", not n.is_actionable)
    check("auto-approved invoice has no recipients", n.recipients == [])
    check("auto-approved invoice states no notification is sent",
          "No notification" in n.subject, n.subject)
    check("auto-approved invoice still names the invoice", "INV-1001" in n.summary_line)


def test_notification_routed():
    print("\nNOTIFICATION — ROUTED")

    elevated = confidence.assess({"LogReg": 0.55, "RBF-SVM": 0.60, "Quantum": 0.58})
    decision = rules.evaluate(15_000.0, CLEAN_FEATURES, elevated)
    check("a mid-risk large invoice actually routes", not decision.is_auto, decision.route)

    n = notif.build_notification("INV-1002", "Bright Logistics", 15_000.0, decision,
                                  submitted_at="2026-03-01T09:00:00+00:00")
    check("routed invoice is actionable", n.is_actionable)
    check("routed invoice has at least one recipient", len(n.recipients) >= 1, n.recipients)
    check("recipient address is on the documentation domain, not a real one",
          all(addr.endswith("@yourcompany.example") for addr in n.recipients), n.recipients)
    check("subject names the invoice and the amount", "INV-1002" in n.subject and "15,000" in n.subject,
          n.subject)
    check("subject carries a priority tag", n.subject.startswith("["), n.subject)
    check("reasons mirror the routing decision's rules",
          [r["code"] for r in n.reasons] == [h.code for h in decision.rules], n.reasons)
    check("an absolute SLA deadline is computed when a submission time is given",
          "2026-03-02" in n.sla_line, n.sla_line)

    n_no_time = notif.build_notification("INV-1003", "Bright Logistics", 15_000.0, decision)
    check("SLA falls back to relative wording without a submission time",
          "hours of assignment" in n_no_time.sla_line, n_no_time.sla_line)

    n_dict = n.to_dict()
    check("to_dict round-trips the fields the renderer needs",
          {"invoice_id", "route", "is_actionable", "priority", "recipients",
           "subject", "summary_line", "reasons", "sla_line"} <= n_dict.keys())


def test_notification_dual_control():
    print("\nNOTIFICATION — DUAL CONTROL")

    decision = rules.evaluate(75_000.0, CLEAN_FEATURES,
                               confidence.assess({"LogReg": 0.1, "RBF-SVM": 0.1, "Quantum": 0.1}))
    check("a large invoice actually reaches dual control", decision.route == "dual_control", decision.route)

    n = notif.build_notification("INV-1004", "Kaveri Suppliers", 75_000.0, decision)
    check("dual control notifies two distinct recipients", len(set(n.recipients)) == 2, n.recipients)
    check("dual control priority reads as urgent", n.priority == "Urgent", n.priority)
    check("dual control explains the two-approver requirement",
          "two different approvers" in n.sla_line, n.sla_line)


def test_notification_missing_fields():
    print("\nNOTIFICATION — MISSING DATA")

    decision = rules.evaluate(None, {}, None, extraction_ok=False)
    check("a failed extraction still routes rather than crashing", not decision.is_auto)

    n = notif.build_notification(None, None, None, decision)
    check("missing invoice id does not crash the preview", "unassigned invoice" in n.subject, n.subject)
    check("missing amount is stated rather than rendered as text 'None'",
          "amount not extracted" in n.subject, n.subject)
    check("missing vendor is stated rather than rendered as text 'None'",
          "unidentified vendor" in n.summary_line, n.summary_line)


def main():
    print("KPI dashboard + notification preview — tests")
    test_summarize_results()
    test_touchless_and_run_history()
    test_false_positive()
    test_notification_auto_approve()
    test_notification_routed()
    test_notification_dual_control()
    test_notification_missing_fields()
    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
