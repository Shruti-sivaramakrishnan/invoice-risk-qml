"""
notifications.py
Simulated approver notifications.

Nothing here sends anything. Before any real integration exists to page an
approver, the workflow needs a way to check exactly what they would see —
subject line, urgency, recipients, and the rules that drove the routing — so
the wording and severity of an alert can be reviewed against the policy that
generates it. This module builds that preview as a plain data object; app.py
renders it.

Recipients are mapped from the approval route, not looked up from a real
directory — there is no HR or distribution-list integration here, and the
addresses below are placeholders on the reserved documentation domain
(RFC 2606) rather than anything that could resolve to a real inbox.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Reserved-for-documentation domain (RFC 2606) so nothing here could ever be
# mistaken for, or accidentally resolve to, a real address.
RECIPIENTS = {
    "manager_review": ["ap-manager@yourcompany.example"],
    "controller_review": ["controller@yourcompany.example"],
    "dual_control": ["controller@yourcompany.example", "internal-audit@yourcompany.example"],
}

PRIORITY_BY_TIER = {
    "routine": "Routine",
    "elevated": "Standard",
    "high": "High",
    "critical": "Urgent",
}


@dataclass(frozen=True)
class Notification:
    """
    A rendered preview of the alert a routing decision would generate.

    `to_dict()` is the canonical shape: it's what gets written to the audit
    trail, and app.py renders both a freshly built Notification and one read
    back from a stored event through the same dict-shaped view, so a preview
    always matches what the trail says was generated.
    """
    invoice_id: str
    route: str
    is_actionable: bool
    priority: str
    recipients: list = field(default_factory=list)
    subject: str = ""
    summary_line: str = ""
    reasons: list = field(default_factory=list)   # rule dicts: code/name/detail
    sla_line: str = ""

    def to_dict(self):
        return {
            "invoice_id": self.invoice_id,
            "route": self.route,
            "is_actionable": self.is_actionable,
            "priority": self.priority,
            "recipients": list(self.recipients),
            "subject": self.subject,
            "summary_line": self.summary_line,
            "reasons": list(self.reasons),
            "sla_line": self.sla_line,
        }


def _format_amount(amount):
    return f"${amount:,.2f}" if amount is not None else "amount not extracted"


def _due_by(submitted_at, sla_hours):
    """Absolute SLA deadline as a UTC string, or None if it can't be computed."""
    if not submitted_at or not sla_hours:
        return None
    try:
        submitted_dt = datetime.fromisoformat(submitted_at)
    except (TypeError, ValueError):
        return None
    return (submitted_dt + timedelta(hours=sla_hours)).strftime("%Y-%m-%d %H:%M UTC")


def build_notification(invoice_id, vendor, amount, decision, submitted_at=None):
    """
    Build the notification preview for one routed invoice.

    invoice_id, vendor, amount — the fields a real alert would name
    decision      — an approval_rules.RoutingDecision
    submitted_at  — ISO timestamp the invoice entered the queue, used to
                    compute an absolute SLA deadline; omitted when previewing
                    a decision before submission, in which case the SLA is
                    stated relative to assignment instead of as a clock time.

    Returns a Notification. Auto-approved invoices never reach an approver, so
    their notification is a non-actionable placeholder rather than empty —
    a preview should always show *something* for the route it describes.
    """
    invoice_id = invoice_id or "unassigned invoice"
    vendor_text = vendor or "an unidentified vendor"
    amount_text = _format_amount(amount)

    if decision.is_auto:
        return Notification(
            invoice_id=invoice_id,
            route=decision.route,
            is_actionable=False,
            priority="None",
            recipients=[],
            subject=f"No notification: {invoice_id} cleared automatically",
            summary_line=(
                f"{invoice_id} from {vendor_text} for {amount_text} triggered no approval "
                f"rules and was auto-approved. No approver is notified."
            ),
            sla_line="No action required.",
        )

    recipients = RECIPIENTS.get(decision.route, [])
    priority = PRIORITY_BY_TIER.get(decision.tier, "Standard")
    subject = f"[{priority}] {invoice_id} needs {decision.label.lower()}, {amount_text}"

    summary_line = (
        f"{invoice_id} from {vendor_text} for {amount_text} was routed to "
        f"{decision.label.lower()} under policy v{decision.policy_version}."
    )

    due_by = _due_by(submitted_at, decision.sla_hours)
    if due_by:
        sla_line = f"Respond by {due_by} ({decision.sla_hours}h SLA)."
    elif decision.sla_hours:
        sla_line = f"Respond within {decision.sla_hours} hours of assignment."
    else:
        sla_line = "Respond as soon as practical."

    if decision.route == "dual_control":
        sla_line += (
            " Requires sign-off from two different approvers. A single "
            "approval will not release this invoice."
        )

    return Notification(
        invoice_id=invoice_id,
        route=decision.route,
        is_actionable=True,
        priority=priority,
        recipients=recipients,
        subject=subject,
        summary_line=summary_line,
        reasons=decision.rules_as_dicts(),
        sla_line=sla_line,
    )
