"""
test_approval_workflow.py
Smoke tests for the approval workflow: routing rules over graded duplicate
scores and model confidence bands, queue state transitions, and audit-trail
integrity.

Runs against a throwaway database so it never touches data/ledger.db.

Run with:  python src/test_approval_workflow.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import approval_rules as rules
import audit_store as store
import confidence

PASSED, FAILED = 0, 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def expect_error(name, fn, *args, **kwargs):
    global PASSED, FAILED
    try:
        fn(*args, **kwargs)
    except store.WorkflowError:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name} — expected WorkflowError, none raised")


CLEAN_FEATURES = {
    "amount_zscore_norm": 0.10, "threshold_proximity": 0.05,
    "duplicate_score": 0.0, "weekend_flag": 0.0, "days_to_due_norm": 0.5,
}
# A confidently-clean ensemble: all three models well inside the cleared band.
CLEAR_CONFIDENCE = confidence.assess({"LogReg": 0.05, "RBF-SVM": 0.10, "Quantum": 0.08})


def test_routing():
    print("\nROUTING RULES")

    d = rules.evaluate(800.0, CLEAN_FEATURES, CLEAR_CONFIDENCE)
    check("small clean invoice auto-approves", d.route == "auto_approve", d.route)
    check("auto-approval fires no rules", d.rules == [])

    d = rules.evaluate(4_000.0, CLEAN_FEATURES, CLEAR_CONFIDENCE)
    check("above manager limit routes to manager", d.route == "manager_review", d.route)

    d = rules.evaluate(20_000.0, CLEAN_FEATURES, CLEAR_CONFIDENCE)
    check("above controller limit routes to controller", d.route == "controller_review", d.route)

    d = rules.evaluate(75_000.0, CLEAN_FEATURES, CLEAR_CONFIDENCE)
    check("above dual-control limit routes to dual control", d.route == "dual_control", d.route)

    d = rules.evaluate(9_800.0, CLEAN_FEATURES, CLEAR_CONFIDENCE)
    check("structuring band routes to controller", d.route == "controller_review", d.route)
    check("structuring rule is cited", any(r.code == "R-02" for r in d.rules))

    d = rules.evaluate(10_100.0, CLEAN_FEATURES, CLEAR_CONFIDENCE)
    check("just above threshold is not structuring", not any(r.code == "R-02" for r in d.rules))

    dup = dict(CLEAN_FEATURES, duplicate_score=1.0, duplicate_match_id="INV-0019")
    d = rules.evaluate(500.0, dup, CLEAR_CONFIDENCE)
    check("confirmed duplicate routes to controller", d.route == "controller_review", d.route)
    check("duplicate rule cites the matched invoice",
          any("INV-0019" in r.detail for r in d.rules), [r.detail for r in d.rules])
    d = rules.evaluate(15_000.0, dup, CLEAR_CONFIDENCE)
    check("confirmed duplicate above threshold routes to dual control",
          d.route == "dual_control", d.route)

    near = dict(CLEAN_FEATURES, duplicate_score=0.88, duplicate_match_id="INV-0019")
    d = rules.evaluate(500.0, near, CLEAR_CONFIDENCE)
    check("near-duplicate routes to manager, not controller", d.route == "manager_review", d.route)
    d = rules.evaluate(15_000.0, near, CLEAR_CONFIDENCE)
    check("near-duplicate does not reach dual control on amount alone",
          d.route == "controller_review", d.route)

    below = dict(CLEAN_FEATURES, duplicate_score=0.70)
    check("similarity below the screening threshold does not route",
          rules.evaluate(500.0, below, CLEAR_CONFIDENCE).route == "auto_approve")

    elevated = confidence.assess({"LogReg": 0.55, "RBF-SVM": 0.60, "Quantum": 0.58})
    d = rules.evaluate(500.0, CLEAN_FEATURES, elevated)
    check("elevated band routes to manager", d.route == "manager_review", d.route)

    high = confidence.assess({"LogReg": 0.90, "RBF-SVM": 0.85, "Quantum": 0.88})
    d = rules.evaluate(500.0, CLEAN_FEATURES, high)
    check("high band routes to controller", d.route == "controller_review", d.route)

    low = confidence.assess({"LogReg": 0.30, "RBF-SVM": 0.35, "Quantum": 0.32})
    d = rules.evaluate(500.0, CLEAN_FEATURES, low)
    check("low band does not route", d.route == "auto_approve", d.route)

    # A split is routed on its own, separately from where the mean lands.
    split = confidence.assess({"LogReg": 0.85, "RBF-SVM": 0.10, "Quantum": 0.15})
    d = rules.evaluate(500.0, CLEAN_FEATURES, split)
    check("split decision routes to manager", d.route == "manager_review", d.route)
    check("split decision is cited", any(r.code == "R-09" for r in d.rules), [r.code for r in d.rules])

    quiet_split = confidence.assess({"LogReg": 0.40, "RBF-SVM": 0.02, "Quantum": 0.05})
    d = rules.evaluate(500.0, CLEAN_FEATURES, quiet_split)
    check("split entirely below the review threshold does not route",
          d.route == "auto_approve", d.route)

    weekend = dict(CLEAN_FEATURES, weekend_flag=1.0)
    check("weekend below de-minimis stays auto",
          rules.evaluate(200.0, weekend, CLEAR_CONFIDENCE).route == "auto_approve")
    check("weekend above de-minimis routes to manager",
          rules.evaluate(2_000.0, weekend, CLEAR_CONFIDENCE).route == "manager_review")

    outlier = dict(CLEAN_FEATURES, amount_zscore_norm=0.95)
    check("off-pattern amount routes to manager",
          rules.evaluate(500.0, outlier, CLEAR_CONFIDENCE).route == "manager_review")

    rushed = dict(CLEAN_FEATURES, days_to_due_norm=0.0)
    check("compressed terms route to manager",
          rules.evaluate(500.0, rushed, CLEAR_CONFIDENCE).route == "manager_review")

    d = rules.evaluate(None, {}, None, extraction_ok=False)
    check("failed extraction never auto-approves", d.route == "manager_review", d.route)
    check("extraction rule is cited", any(r.code == "R-08" for r in d.rules))

    # Escalation only, never de-escalation: a manager-level flag on a dual-control
    # amount must not pull the invoice back down the ladder.
    d = rules.evaluate(75_000.0, dict(CLEAN_FEATURES, weekend_flag=1.0), CLEAR_CONFIDENCE)
    check("routing takes the highest floor", d.route == "dual_control", d.route)
    check("all triggered rules are recorded", len(d.rules) == 2, [r.code for r in d.rules])
    check("highest-authority rule listed first", d.rules[0].route == "dual_control")

    check("escalate_route steps up", rules.escalate_route("manager_review") == "controller_review")
    check("escalate_route saturates at the top", rules.escalate_route("dual_control") == "dual_control")


_UNSET = object()  # distinguishes "use the default" from "deliberately unscored"


def _submit(invoice_id, amount, features=None, assessment=_UNSET, extraction_ok=True):
    record = {"invoice_id": invoice_id, "vendor": "Acme Supply Co",
              "amount": amount, "posting_date": "2025-03-10", "due_date": "2025-04-09"}
    features = features or CLEAN_FEATURES
    assessment = CLEAR_CONFIDENCE if assessment is _UNSET else assessment
    decision = rules.evaluate(amount, features, assessment, extraction_ok)
    return store.submit_invoice(record, features, assessment, decision,
                                source_file=f"{invoice_id}.pdf", actor="intake-bot")


def test_queue_and_actions():
    print("\nREVIEW QUEUE + REVIEWER ACTIONS")

    _, status, was_new = _submit("INV-9001", 800.0)
    check("auto-approved invoice lands as auto_approved", status == store.STATUS_AUTO_APPROVED, status)
    check("first submission reports as new", was_new)

    _, status, _ = _submit("INV-9002", 4_000.0)
    check("routed invoice lands as pending", status == store.STATUS_PENDING, status)

    new_status, _ = store.apply_action("INV-9002", "request_info", "dana@ap", "Need the PO number.")
    check("request_info moves to info_requested", new_status == store.STATUS_INFO, new_status)

    new_status, _ = store.apply_action("INV-9002", "approve", "dana@ap", "PO received.")
    check("approve from info_requested closes the item", new_status == store.STATUS_APPROVED, new_status)

    expect_error("approved invoice cannot be re-actioned",
                 store.apply_action, "INV-9002", "reject", "dana@ap")

    _, status, _ = _submit("INV-9002", 4_000.0)
    check("re-submitting a decided invoice does not reopen it",
          status == store.STATUS_APPROVED, status)

    _submit("INV-9003", 4_000.0)
    new_status, _ = store.apply_action("INV-9003", "reject", "sam@ap", "Vendor not on file.")
    check("reject closes the item", new_status == store.STATUS_REJECTED, new_status)

    # Re-running a batch must refresh the scoring without erasing in-flight review state.
    _submit("INV-9007", 4_000.0)
    store.apply_action("INV-9007", "request_info", "dana@ap", "Missing receipt.")
    _, status, was_new = _submit("INV-9007", 4_000.0)
    check("re-scoring an open invoice preserves its review state",
          status == store.STATUS_INFO, status)
    check("re-submission is not reported as new", not was_new)
    check("re-scoring is logged distinctly",
          "intake.rescored" in set(store.load_events("INV-9007")["event_type"]))

    # Escalation raises the route and returns the item to the queue.
    _submit("INV-9004", 4_000.0)
    new_status, _ = store.apply_action("INV-9004", "escalate", "dana@ap", "Looks structured.")
    row = store.load_queue()
    row = row[row["invoice_id"] == "INV-9004"].iloc[0]
    check("escalation returns item to pending", new_status == store.STATUS_PENDING, new_status)
    check("escalation raises the route", row["route"] == "controller_review", row["route"])

    # Dual control: two distinct approvers required.
    _, status, _ = _submit("INV-9005", 75_000.0)
    check("dual-control invoice starts pending", status == store.STATUS_PENDING, status)
    new_status, _ = store.apply_action("INV-9005", "approve", "dana@ap", "First sign-off.")
    check("first dual-control approval awaits a second",
          new_status == store.STATUS_AWAITING_SECOND, new_status)
    expect_error("same approver cannot give both dual-control approvals",
                 store.apply_action, "INV-9005", "approve", "dana@ap")
    new_status, _ = store.apply_action("INV-9005", "approve", "sam@ap", "Second sign-off.")
    check("second distinct approver closes dual control",
          new_status == store.STATUS_APPROVED, new_status)

    expect_error("unknown invoice is rejected", store.apply_action, "INV-0000", "approve", "dana@ap")
    expect_error("blank reviewer name is rejected", store.apply_action, "INV-9001", "approve", "   ")
    expect_error("unknown action is rejected", store.apply_action, "INV-9004", "shred", "dana@ap")

    _, status, _ = _submit("INV-9006", None, features={}, assessment=None, extraction_ok=False)
    check("unextractable invoice is still queued", status == store.STATUS_PENDING, status)

    stats = store.queue_stats()
    check("queue holds every submitted invoice", stats["total"] == 7, stats)
    check("open count excludes terminal items", stats["open"] == 3, stats)

    q = store.load_queue(statuses=[store.STATUS_PENDING])
    check("status filter narrows the queue", set(q["status"]) == {store.STATUS_PENDING}, list(q["status"]))
    check("queue sorts highest route first",
          store.load_queue().iloc[0]["route"] in ("dual_control", "controller_review"))


def test_batch_runs_and_notifications():
    print("\nBATCH RUN HISTORY + NOTIFICATION LOGGING")

    import dashboard

    # A routed invoice should generate a logged notification; an
    # auto-approved one should not, since no approver is ever notified.
    _submit("INV-9101", 4_000.0)
    routed_events = set(store.load_events("INV-9101")["event_type"])
    check("routing a new invoice logs a notification event",
          "notification.generated" in routed_events, routed_events)

    _submit("INV-9102", 800.0)
    auto_events = set(store.load_events("INV-9102")["event_type"])
    check("auto-approving a new invoice logs no notification event",
          "notification.generated" not in auto_events, auto_events)

    # Re-scoring an invoice that's still open must not spam a fresh
    # notification on every re-run of the same batch.
    before = len(store.load_events("INV-9101"))
    _submit("INV-9101", 4_000.0)
    after = store.load_events("INV-9101")
    check("re-scoring an already-open invoice does not log a second notification",
          list(after["event_type"]).count("notification.generated") == 1, list(after["event_type"]))

    results = [
        {"route": "auto_approve", "band": "low", "split": False, "duplicate_score": 0.0, "failed": False},
        {"route": "manager_review", "band": "elevated", "split": False, "duplicate_score": 0.0, "failed": False},
    ]
    summary = dashboard.summarize_results(results)
    run_id = store.record_batch_run("batch_intake", "batch-intake", summary)
    check("record_batch_run returns an id", run_id is not None)

    runs = store.load_batch_runs()
    check("the recorded run is retrievable", len(runs) == 1, len(runs))
    row = runs.iloc[0]
    check("run counts match the summary that was recorded",
          row["invoice_count"] == 2 and row["auto_approved_count"] == 1, dict(row))

    run_events = set(store.load_events(limit=5000)["event_type"])
    check("a batch run is logged to the audit trail",
          "batch.completed" in run_events, run_events)

    store.record_batch_run("batch_intake", "batch-intake", summary)
    runs_after_second = store.load_batch_runs()
    check("a second run is appended rather than replacing the first",
          len(runs_after_second) == 2, len(runs_after_second))
    check("runs are returned oldest first",
          list(runs_after_second["run_id"]) == sorted(runs_after_second["run_id"]))


def test_audit_trail():
    print("\nAUDIT TRAIL")

    events = store.load_events(limit=1000)
    check("events were recorded", len(events) > 0, len(events))

    inv_events = store.load_events("INV-9005")
    types = set(inv_events["event_type"])
    check("intake is logged", "intake.scored" in types, types)
    check("routing decision is logged", "routing.assigned" in types, types)
    check("first approval is logged", "review.first_approval" in types, types)
    check("final approval is logged", "review.approved" in types, types)

    auto_events = set(store.load_events("INV-9001")["event_type"])
    check("automatic approval is logged", "approval.automatic" in auto_events, auto_events)

    dup_events = set(store.load_events("INV-9002")["event_type"])
    check("blocked re-submission is logged", "intake.duplicate_submission" in dup_events, dup_events)

    # A refused action must leave no trace beyond the queue being unchanged.
    before = len(store.load_events(limit=5000))
    try:
        store.apply_action("INV-9002", "approve", "dana@ap")
    except store.WorkflowError:
        pass
    check("refused action writes no event", len(store.load_events(limit=5000)) == before)

    ok, msg = store.verify_chain()
    check("hash chain verifies", ok, msg)

    # Tamper with a committed event and confirm the chain reports it.
    from contextlib import closing
    with closing(store.connect()) as conn:
        conn.execute("UPDATE audit_events SET actor = 'someone-else' WHERE seq = 2")
    ok, msg = store.verify_chain()
    check("tampering breaks the chain", not ok, msg)
    check("tamper report names the event", "#2" in msg, msg)


def main():
    global FAILED
    tmp = tempfile.mkdtemp(prefix="ledger-test-")
    store.DB_PATH = os.path.join(tmp, "test_ledger.db")
    store.init_db()

    print(f"Approval workflow — smoke tests (db: {store.DB_PATH})")
    test_routing()
    test_queue_and_actions()
    test_batch_runs_and_notifications()
    test_audit_trail()

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
