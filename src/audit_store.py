"""
audit_store.py
Persistence for the approval workflow: a review queue, a batch-run history,
and an append-only audit trail, all in a local SQLite file (`data/ledger.db`).

Two design choices worth stating, because they're the point of the module:

1. The audit trail is append-only and hash-chained. Every event stores the
   hash of the one before it, so a deleted or edited row breaks the chain and
   `verify_chain()` reports where. An audit log that can be quietly rewritten
   isn't evidence of anything.
2. Queue state changes and their audit events are written in the same
   transaction, so the queue can never show a status the trail can't account
   for.

SQLite is used rather than a CSV because the workflow needs atomic
read-modify-write on queue rows; it's stdlib, so it adds no dependency.
"""

import hashlib
import json
import os
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone

import pandas as pd

from approval_rules import ROUTE_ORDER, escalate_route
from notifications import build_notification

# Overridable so tests (and any future deployment) can point at another file
# without touching the working ledger.
DB_PATH = os.environ.get("LEDGER_DB_PATH") or os.path.join(
    os.path.dirname(__file__), "..", "data", "ledger.db"
)

GENESIS_HASH = "0" * 64

# Queue statuses. `pending_review` and `info_requested` are actionable;
# `awaiting_second_approval` needs a second, different approver; the rest are terminal.
STATUS_PENDING = "pending_review"
STATUS_INFO = "info_requested"
STATUS_AWAITING_SECOND = "awaiting_second_approval"
STATUS_APPROVED = "approved"
STATUS_AUTO_APPROVED = "auto_approved"
STATUS_REJECTED = "rejected"

OPEN_STATUSES = (STATUS_PENDING, STATUS_INFO, STATUS_AWAITING_SECOND)
TERMINAL_STATUSES = (STATUS_APPROVED, STATUS_AUTO_APPROVED, STATUS_REJECTED)

STATUS_META = {
    STATUS_PENDING: {"label": "Pending review", "badge": "badge-controller"},
    STATUS_INFO: {"label": "Info requested", "badge": "badge-manager"},
    STATUS_AWAITING_SECOND: {"label": "Awaiting 2nd approval", "badge": "badge-dual"},
    STATUS_APPROVED: {"label": "Approved", "badge": "badge-auto"},
    STATUS_AUTO_APPROVED: {"label": "Auto-approved", "badge": "badge-auto"},
    STATUS_REJECTED: {"label": "Rejected", "badge": "badge-neutral"},
}

ACTIONS = {
    "approve": "Approve",
    "reject": "Reject",
    "request_info": "Request information",
    "escalate": "Escalate",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS review_queue (
    invoice_id       TEXT PRIMARY KEY,
    source_file      TEXT,
    vendor           TEXT,
    amount           REAL,
    posting_date     TEXT,
    due_date         TEXT,
    features_json    TEXT,
    confidence_json  TEXT,
    confidence_band  TEXT,
    confidence_mean  REAL,
    duplicate_score  REAL,
    duplicate_match_id TEXT,
    route            TEXT NOT NULL,
    risk_tier        TEXT,
    rules_json       TEXT,
    policy_version   TEXT,
    status           TEXT NOT NULL,
    submitted_at     TEXT NOT NULL,
    submitted_by     TEXT,
    updated_at       TEXT NOT NULL,
    first_approver   TEXT,
    first_approved_at TEXT,
    decided_by       TEXT,
    decided_at       TEXT,
    decision_note    TEXT
);

CREATE TABLE IF NOT EXISTS audit_events (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    invoice_id   TEXT,
    event_type   TEXT NOT NULL,
    actor        TEXT NOT NULL,
    from_status  TEXT,
    to_status    TEXT,
    detail_json  TEXT,
    prev_hash    TEXT NOT NULL,
    entry_hash   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS batch_runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at          TEXT NOT NULL,
    source              TEXT NOT NULL,
    actor               TEXT,
    invoice_count       INTEGER NOT NULL,
    auto_approved_count INTEGER NOT NULL,
    failed_count        INTEGER NOT NULL,
    high_band_count     INTEGER NOT NULL,
    split_count         INTEGER NOT NULL,
    duplicate_count     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS vendor_profiles (
    vendor              TEXT PRIMARY KEY,
    total_invoices      INTEGER NOT NULL DEFAULT 0,
    flagged_invoices    INTEGER NOT NULL DEFAULT 0,
    approved_flagged    INTEGER NOT NULL DEFAULT 0,
    recent_flagged_5    INTEGER NOT NULL DEFAULT 0,
    updated_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_invoice ON audit_events (invoice_id);
CREATE INDEX IF NOT EXISTS idx_queue_status ON review_queue (status);
"""

# Columns added after the first release. CREATE TABLE IF NOT EXISTS won't add
# them to a ledger that already exists, so they're applied by _migrate().
ADDED_COLUMNS = {
    "confidence_json": "TEXT",
    "confidence_band": "TEXT",
    "confidence_mean": "REAL",
    "duplicate_score": "REAL",
    "duplicate_match_id": "TEXT",
    "reviewer_feedback": "TEXT",
    "feedback_at": "TEXT",
}


class WorkflowError(Exception):
    """Raised when a reviewer action isn't legal for the invoice's current state."""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect():
    """
    Open a connection in autocommit mode. Transactions are opened explicitly by
    `_write_tx` rather than left to sqlite3's implicit handling, so a queue
    update and its audit event are guaranteed to land together or not at all.
    """
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


@contextmanager
def _write_tx():
    """Exclusive write transaction: commits on success, rolls back on any error."""
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()


def _migrate(conn):
    """
    Bring an existing ledger up to the current column set.

    Only additive: columns are added, never dropped or rewritten, so the audit
    events already on file keep hashing to the same values and the chain stays
    verifiable across upgrades.
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(review_queue)")}
    for column, column_type in ADDED_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE review_queue ADD COLUMN {column} {column_type}")


def init_db():
    with closing(connect()) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _hash_entry(prev_hash, payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{prev_hash}|{canonical}".encode("utf-8")).hexdigest()


def _append_event(conn, invoice_id, event_type, actor, from_status=None, to_status=None, detail=None):
    """Append one hash-chained event. Must run inside an open transaction."""
    row = conn.execute("SELECT entry_hash FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
    prev_hash = row["entry_hash"] if row else GENESIS_HASH

    ts = _now()
    detail_json = json.dumps(detail or {}, sort_keys=True, default=str)
    payload = {
        "ts": ts,
        "invoice_id": invoice_id,
        "event_type": event_type,
        "actor": actor,
        "from_status": from_status,
        "to_status": to_status,
        "detail": detail_json,
    }
    entry_hash = _hash_entry(prev_hash, payload)

    conn.execute(
        """INSERT INTO audit_events
           (ts, invoice_id, event_type, actor, from_status, to_status, detail_json, prev_hash, entry_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ts, invoice_id, event_type, actor, from_status, to_status, detail_json, prev_hash, entry_hash),
    )
    return entry_hash


# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------

def submit_invoice(record, features, confidence, decision, source_file=None, actor="system"):
    """
    Persist a scored + routed invoice into the queue and log its intake.

    `confidence` is a confidence.Assessment (or None when the invoice could not
    be scored). Its full breakdown is stored, not just the band, so a decision
    can be re-read later against the scores that actually drove it.

    Re-submitting an invoice that is still open re-scores it in place and logs
    the re-run. Re-submitting one that has already been decided is refused —
    a closed decision shouldn't silently reopen because someone re-ran a batch.

    Returns (invoice_id, status, was_new).
    """
    invoice_id = record.get("invoice_id") or f"UNREAD::{source_file or 'unknown'}"
    features = features or {}
    confidence_dict = confidence.to_dict() if confidence is not None else None

    now = _now()

    with _write_tx() as conn:
        existing = conn.execute(
            "SELECT status FROM review_queue WHERE invoice_id = ?", (invoice_id,)
        ).fetchone()

        # An invoice already in review keeps its place in the workflow: re-running a
        # batch refreshes the scoring and routing, but must not erase that a reviewer
        # asked for information or already gave a first dual-control approval.
        if existing:
            status = existing["status"]
        else:
            status = STATUS_AUTO_APPROVED if decision.is_auto else STATUS_PENDING

        if existing and existing["status"] in TERMINAL_STATUSES:
            _append_event(
                conn, invoice_id, "intake.duplicate_submission", actor,
                from_status=existing["status"], to_status=existing["status"],
                detail={"reason": "invoice already decided; re-scoring skipped",
                        "source_file": source_file},
            )
            return invoice_id, existing["status"], False

        payload = (
            invoice_id, source_file, record.get("vendor"), record.get("amount"),
            record.get("posting_date"), record.get("due_date"),
            json.dumps(features, sort_keys=True),
            json.dumps(confidence_dict, sort_keys=True) if confidence_dict else None,
            confidence.band if confidence is not None else None,
            confidence.mean if confidence is not None else None,
            features.get("duplicate_score"), features.get("duplicate_match_id"),
            decision.route, decision.tier, json.dumps(decision.rules_as_dicts()),
            decision.policy_version, status, now, actor, now,
        )
        conn.execute(
            """INSERT INTO review_queue
               (invoice_id, source_file, vendor, amount, posting_date, due_date,
                features_json, confidence_json, confidence_band, confidence_mean,
                duplicate_score, duplicate_match_id, route, risk_tier, rules_json,
                policy_version, status, submitted_at, submitted_by, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(invoice_id) DO UPDATE SET
                 source_file=excluded.source_file, vendor=excluded.vendor,
                 amount=excluded.amount, posting_date=excluded.posting_date,
                 due_date=excluded.due_date, features_json=excluded.features_json,
                 confidence_json=excluded.confidence_json,
                 confidence_band=excluded.confidence_band,
                 confidence_mean=excluded.confidence_mean,
                 duplicate_score=excluded.duplicate_score,
                 duplicate_match_id=excluded.duplicate_match_id,
                 route=excluded.route, risk_tier=excluded.risk_tier,
                 rules_json=excluded.rules_json, policy_version=excluded.policy_version,
                 status=excluded.status, updated_at=excluded.updated_at""",
            payload,
        )

        if not existing:
            _update_vendor_profiles(conn, record.get("vendor"), decision.route, status)

        _append_event(
            conn, invoice_id, "intake.rescored" if existing else "intake.scored", actor,
            from_status=existing["status"] if existing else None, to_status=status,
            detail={"source_file": source_file, "vendor": record.get("vendor"),
                    "amount": record.get("amount"), "features": features,
                    "confidence": confidence_dict},
        )
        _append_event(
            conn, invoice_id, "routing.assigned", "policy-engine", to_status=status,
            detail={"route": decision.route, "approver": decision.approver,
                    "tier": decision.tier, "policy_version": decision.policy_version,
                    "confidence_band": confidence.band if confidence is not None else None,
                    "rules": decision.rules_as_dicts()},
        )
        if decision.is_auto and not existing:
            _append_event(
                conn, invoice_id, "approval.automatic", "policy-engine",
                from_status=None, to_status=STATUS_AUTO_APPROVED,
                detail={"reason": "no approval rules triggered"},
            )
        elif not decision.is_auto and not existing:
            # A routed invoice would trigger a real alert to its approver. The
            # simulated notification is logged here, at the moment routing
            # actually happens, rather than only rendered on demand later —
            # so the audit trail shows what an approver would have been told,
            # not just what the app can currently reconstruct from the route.
            notification = build_notification(
                invoice_id, record.get("vendor"), record.get("amount"), decision,
                submitted_at=now,
            )
            _append_event(
                conn, invoice_id, "notification.generated", "policy-engine", to_status=status,
                detail=notification.to_dict(),
            )

    return invoice_id, status, existing is None


# ---------------------------------------------------------------------------
# Batch run history
# ---------------------------------------------------------------------------

def record_batch_run(source, actor, summary):
    """
    Persist one batch's aggregate counts as a run-history row.

    `summary` is the dict dashboard.summarize_results() returns — total,
    auto_approved, failed, high_band, split, duplicates — so the row and the
    on-screen batch metrics are always built from the same numbers rather
    than two independent computations that could quietly drift apart.

    Every run also gets a "batch.completed" audit event, consistent with
    every other step of the workflow being written to the trail.
    """
    now = _now()
    with _write_tx() as conn:
        cur = conn.execute(
            """INSERT INTO batch_runs
               (started_at, source, actor, invoice_count, auto_approved_count,
                failed_count, high_band_count, split_count, duplicate_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, source, actor, summary["total"], summary["auto_approved"],
             summary["failed"], summary["high_band"], summary["split"],
             summary["duplicates"]),
        )
        run_id = cur.lastrowid
        _append_event(
            conn, None, "batch.completed", actor,
            detail={"run_id": run_id, "source": source, **summary},
        )
    return run_id


def load_batch_runs(limit=200):
    """Batch run history, oldest first, for trend reporting."""
    with closing(connect()) as conn:
        return pd.read_sql_query(
            "SELECT * FROM batch_runs ORDER BY started_at ASC LIMIT ?", conn, params=(limit,)
        )


# ---------------------------------------------------------------------------
# Vendor risk profiles
# ---------------------------------------------------------------------------

def _update_vendor_profiles(conn, vendor, route, status):
    """
    Update vendor risk metrics when an invoice is submitted or decided.
    Tracks total invoices, flagged count, approved-flagged count, and
    recent flagged count for the last 5 invoices.
    """
    if not vendor or vendor.strip() == "":
        return

    now = _now()
    is_flagged = route != "auto_approve"
    is_approved = status == STATUS_APPROVED

    conn.execute(
        """INSERT INTO vendor_profiles (vendor, total_invoices, flagged_invoices,
           approved_flagged, recent_flagged_5, updated_at)
           VALUES (?, 1, ?, ?, 0, ?)
           ON CONFLICT(vendor) DO UPDATE SET
             total_invoices = total_invoices + 1,
             flagged_invoices = flagged_invoices + ?,
             approved_flagged = approved_flagged + ?,
             updated_at = ?""",
        (vendor, 1 if is_flagged else 0, 1 if (is_flagged and is_approved) else 0, now,
         1 if is_flagged else 0, 1 if (is_flagged and is_approved) else 0, now),
    )


def load_vendor_profiles(limit=100):
    """Load vendor risk profiles sorted by recent risk activity."""
    with closing(connect()) as conn:
        df = pd.read_sql_query(
            """SELECT vendor, total_invoices, flagged_invoices, approved_flagged,
                      recent_flagged_5, updated_at
               FROM vendor_profiles
               WHERE total_invoices > 0
               ORDER BY flagged_invoices DESC, updated_at DESC
               LIMIT ?""",
            conn, params=(limit,)
        )
    return df


def vendor_risk_summary(queue_df):
    """
    Compute per-vendor risk summaries from the queue.
    Returns a DataFrame with: vendor, total, flagged, flagged_rate, recent_flagged,
    recent_rate (flagged / last 5 invoices).
    """
    if queue_df is None or queue_df.empty:
        return pd.DataFrame(columns=["vendor", "total", "flagged", "flagged_rate"])

    vendors = queue_df.groupby("vendor").agg(
        total=("invoice_id", "count"),
        flagged=(
            "route",
            lambda routes: (routes != "auto_approve").sum()
        ),
        approved_flagged=(
            "status",
            lambda statuses: ((queue_df.loc[statuses.index, "route"] != "auto_approve") &
                             (statuses == STATUS_APPROVED)).sum()
        ),
    ).reset_index()

    vendors["flagged_rate"] = (vendors["flagged"] / vendors["total"]).round(2)

    for idx, row in vendors.iterrows():
        vendor_invoices = queue_df[queue_df["vendor"] == row["vendor"]].sort_values("submitted_at", ascending=False)
        recent_5 = vendor_invoices.head(5)
        vendors.at[idx, "recent_flagged"] = (recent_5["route"] != "auto_approve").sum()

    vendors = vendors[
        ["vendor", "total", "flagged", "flagged_rate", "approved_flagged", "recent_flagged"]
    ].sort_values("flagged", ascending=False)

    return vendors


# ---------------------------------------------------------------------------
# Reviewer feedback for model retraining
# ---------------------------------------------------------------------------

def _record_feedback(conn, invoice_id, status, actor):
    """
    Record reviewer feedback when a decision is made. Stores the reviewer's
    decision (approved/rejected) as training signal for model retraining.
    """
    if status not in (STATUS_APPROVED, STATUS_REJECTED):
        return

    feedback_type = "false_positive" if status == STATUS_APPROVED else "true_positive"
    now = _now()

    conn.execute(
        """UPDATE review_queue SET reviewer_feedback = ?, feedback_at = ?
           WHERE invoice_id = ?""",
        (feedback_type, now, invoice_id),
    )


def load_feedback_for_retraining(limit=None):
    """
    Load invoices with reviewer feedback (approved/rejected decisions)
    for model retraining. Returns DataFrame with invoice features and
    feedback labels.
    """
    with closing(connect()) as conn:
        query = """
            SELECT invoice_id, vendor, amount, features_json, confidence_json,
                   reviewer_feedback, feedback_at, route, status
            FROM review_queue
            WHERE reviewer_feedback IS NOT NULL
            ORDER BY feedback_at DESC
        """
        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql_query(query, conn)

    if df.empty:
        return df

    df["true_label"] = df["reviewer_feedback"].map({
        "false_positive": 0,
        "true_positive": 1,
    })

    return df


def get_feedback_summary():
    """
    Summary of collected reviewer feedback: total decisions, split between
    false positives and true positives, useful for reporting retraining readiness.
    """
    with closing(connect()) as conn:
        row = conn.execute(
            """SELECT COUNT(*) as total,
                      SUM(CASE WHEN reviewer_feedback = 'false_positive' THEN 1 ELSE 0 END) as false_positives,
                      SUM(CASE WHEN reviewer_feedback = 'true_positive' THEN 1 ELSE 0 END) as true_positives
               FROM review_queue WHERE reviewer_feedback IS NOT NULL"""
        ).fetchone()

    total = row["total"] or 0
    false_pos = row["false_positives"] or 0
    true_pos = row["true_positives"] or 0

    return {
        "total": total,
        "false_positives": false_pos,
        "true_positives": true_pos,
        "ready_for_retraining": total >= 10,
    }


# ---------------------------------------------------------------------------
# Reviewer actions
# ---------------------------------------------------------------------------

def apply_action(invoice_id, action, actor, note=""):
    """
    Apply a reviewer action to a queued invoice.

    Enforces two controls that matter for an approval workflow:
      * terminal invoices cannot be actioned again, and
      * a dual-control invoice needs two *different* approvers.

    Returns (new_status, message). Raises WorkflowError if the action is illegal.
    """
    if action not in ACTIONS:
        raise WorkflowError(f"Unknown action '{action}'.")
    actor = (actor or "").strip()
    if not actor:
        raise WorkflowError("A reviewer name is required before acting on an invoice.")

    now = _now()
    with _write_tx() as conn:
        row = conn.execute("SELECT * FROM review_queue WHERE invoice_id = ?", (invoice_id,)).fetchone()
        if row is None:
            raise WorkflowError(f"{invoice_id} is not in the review queue.")

        current = row["status"]
        if current in TERMINAL_STATUSES:
            raise WorkflowError(
                f"{invoice_id} is already {STATUS_META[current]['label'].lower()}, "
                f"reopening a closed decision isn't permitted."
            )

        route = row["route"]
        updates = {"updated_at": now}
        detail = {"note": note, "route": route}

        if action == "approve":
            needs_dual = route == "dual_control"
            if needs_dual and current != STATUS_AWAITING_SECOND:
                new_status = STATUS_AWAITING_SECOND
                updates.update({"status": new_status, "first_approver": actor, "first_approved_at": now})
                event = "review.first_approval"
                message = f"First approval recorded. {invoice_id} needs a second, different approver."
            elif needs_dual and current == STATUS_AWAITING_SECOND:
                if (row["first_approver"] or "").lower() == actor.lower():
                    raise WorkflowError(
                        f"Dual control: {actor} gave the first approval and cannot give the second."
                    )
                new_status = STATUS_APPROVED
                updates.update({"status": new_status, "decided_by": actor,
                                "decided_at": now, "decision_note": note})
                detail["first_approver"] = row["first_approver"]
                event = "review.approved"
                message = f"{invoice_id} approved under dual control."
            else:
                new_status = STATUS_APPROVED
                updates.update({"status": new_status, "decided_by": actor,
                                "decided_at": now, "decision_note": note})
                event = "review.approved"
                message = f"{invoice_id} approved."

        elif action == "reject":
            new_status = STATUS_REJECTED
            updates.update({"status": new_status, "decided_by": actor,
                            "decided_at": now, "decision_note": note})
            event = "review.rejected"
            message = f"{invoice_id} rejected."

        elif action == "request_info":
            new_status = STATUS_INFO
            updates.update({"status": new_status, "decision_note": note})
            event = "review.info_requested"
            message = f"Information requested on {invoice_id}."

        else:  # escalate
            new_route = escalate_route(route)
            new_status = STATUS_PENDING
            updates.update({"status": new_status, "route": new_route,
                            "first_approver": None, "first_approved_at": None,
                            "decision_note": note})
            detail.update({"route_from": route, "route_to": new_route})
            event = "review.escalated"
            message = (
                f"{invoice_id} escalated to {new_route.replace('_', ' ')}."
                if new_route != route
                else f"{invoice_id} is already at the highest approval route."
            )

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(
            f"UPDATE review_queue SET {set_clause} WHERE invoice_id = ?",
            (*updates.values(), invoice_id),
        )

        if action == "approve" and new_status == STATUS_APPROVED:
            vendor = row["vendor"]
            _update_vendor_profiles(conn, vendor, row["route"], new_status)
            _record_feedback(conn, invoice_id, new_status, actor)
        elif action == "reject" and new_status == STATUS_REJECTED:
            _record_feedback(conn, invoice_id, new_status, actor)

        _append_event(conn, invoice_id, event, actor,
                      from_status=current, to_status=new_status, detail=detail)

    return new_status, message


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def load_queue(statuses=None, routes=None):
    """Queue rows as a DataFrame, most urgent route first, then oldest first."""
    query = "SELECT * FROM review_queue"
    clauses, params = [], []
    if statuses:
        clauses.append(f"status IN ({','.join('?' * len(statuses))})")
        params.extend(statuses)
    if routes:
        clauses.append(f"route IN ({','.join('?' * len(routes))})")
        params.extend(routes)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    with closing(connect()) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df

    df["route_rank"] = df["route"].map(lambda r: -ROUTE_ORDER.index(r) if r in ROUTE_ORDER else 0)
    df = df.sort_values(["route_rank", "submitted_at"]).drop(columns=["route_rank"]).reset_index(drop=True)
    return df


def load_events(invoice_id=None, limit=500):
    query = "SELECT * FROM audit_events"
    params = []
    if invoice_id:
        query += " WHERE invoice_id = ?"
        params.append(invoice_id)
    query += " ORDER BY seq DESC LIMIT ?"
    params.append(limit)

    with closing(connect()) as conn:
        return pd.read_sql_query(query, conn, params=params)


def queue_stats():
    with closing(connect()) as conn:
        rows = conn.execute("SELECT status, COUNT(*) AS n FROM review_queue GROUP BY status").fetchall()
        total_events = conn.execute("SELECT COUNT(*) AS n FROM audit_events").fetchone()["n"]

    counts = {r["status"]: r["n"] for r in rows}
    return {
        "total": sum(counts.values()),
        "open": sum(counts.get(s, 0) for s in OPEN_STATUSES),
        "by_status": counts,
        "events": total_events,
    }


def verify_chain():
    """
    Recompute the hash chain end to end.

    Returns (ok, message). A mismatch means a row was edited or removed
    outside the application — which is exactly what the chain exists to show.
    """
    with closing(connect()) as conn:
        rows = conn.execute("SELECT * FROM audit_events ORDER BY seq ASC").fetchall()

    prev_hash = GENESIS_HASH
    for row in rows:
        payload = {
            "ts": row["ts"],
            "invoice_id": row["invoice_id"],
            "event_type": row["event_type"],
            "actor": row["actor"],
            "from_status": row["from_status"],
            "to_status": row["to_status"],
            "detail": row["detail_json"],
        }
        if row["prev_hash"] != prev_hash:
            return False, f"Chain break at event #{row['seq']}: predecessor hash does not match."
        if _hash_entry(prev_hash, payload) != row["entry_hash"]:
            return False, f"Tampered content at event #{row['seq']}: recomputed hash does not match."
        prev_hash = row["entry_hash"]

    return True, f"Verified {len(rows)} event(s); hash chain intact."
