"""
approval_rules.py
Deterministic, rules-based approval routing for scored invoices.

The models say *how risky* an invoice looks; this module says *who has to sign
off on it*. Routing is deliberately rule-based rather than model-based: an
approval path has to be explainable and reproducible to an auditor, so every
decision here is a pure function of the invoice fields, the engineered risk
features, and the model confidence — no randomness, no learned thresholds.

Each rule that fires names the route floor it demands. The final route is the
highest floor demanded by any fired rule, so routing can only ever escalate,
never soften. If nothing fires, the invoice auto-approves.

Graded inputs get graded responses: the duplicate and model-confidence rules
read continuous scores and route proportionately, rather than tripping a single
threshold and losing the distinction between a marginal case and a certain one.
"""

import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from duplicate_matching import CONFIRMED_DUPLICATE, NEAR_DUPLICATE

POLICY_VERSION = "2.0.0"

# Policy constants — the dollar figures an AP policy document would state.
REPORTING_THRESHOLD = 10_000.0   # reporting / approval threshold invoices get structured around
STRUCTURING_BAND = 0.95          # amounts within 5% *below* the threshold look deliberate
DE_MINIMIS = 1_000.0             # below this, low-severity flags aren't worth a reviewer's time
MANAGER_LIMIT = 2_500.0          # above this, a manager signs off
CONTROLLER_LIMIT = 10_000.0      # above this, the controller signs off
DUAL_CONTROL_LIMIT = 50_000.0    # above this, two people sign off
OUTLIER_ZSCORE = 0.70            # normalized z-score at which an amount is off-pattern for the vendor
RUSHED_TERMS = 0.05              # normalized payment window that counts as unusually short

# The approval ladder, lowest authority first. Routing takes the max.
ROUTE_ORDER = ["auto_approve", "manager_review", "controller_review", "dual_control"]

ROUTE_META = {
    "auto_approve": {
        "label": "Auto-approved",
        "approver": "System",
        "tier": "routine",
        "sla_hours": 0,
        "badge": "badge-auto",
    },
    "manager_review": {
        "label": "Manager review",
        "approver": "AP Manager",
        "tier": "elevated",
        "sla_hours": 48,
        "badge": "badge-manager",
    },
    "controller_review": {
        "label": "Controller review",
        "approver": "Financial Controller",
        "tier": "high",
        "sla_hours": 24,
        "badge": "badge-controller",
    },
    "dual_control": {
        "label": "Dual control",
        "approver": "Controller + Internal Audit",
        "tier": "critical",
        "sla_hours": 8,
        "badge": "badge-dual",
    },
}


@dataclass(frozen=True)
class RuleHit:
    """One rule that fired, with the evidence that made it fire."""
    code: str
    name: str
    route: str
    detail: str

    def to_dict(self):
        return {"code": self.code, "name": self.name, "route": self.route, "detail": self.detail}


@dataclass
class RoutingDecision:
    route: str
    label: str
    approver: str
    tier: str
    sla_hours: int
    badge: str
    rules: list = field(default_factory=list)
    policy_version: str = POLICY_VERSION

    @property
    def is_auto(self):
        return self.route == "auto_approve"

    def rules_as_dicts(self):
        return [r.to_dict() for r in self.rules]

    def summary(self):
        if not self.rules:
            return "No approval rules triggered, cleared for automatic payment."
        return f"{len(self.rules)} rule(s) triggered; highest authority required: {self.label}."


def _higher(route_a, route_b):
    """Return whichever route sits higher on the approval ladder."""
    return route_a if ROUTE_ORDER.index(route_a) >= ROUTE_ORDER.index(route_b) else route_b


# ---------------------------------------------------------------------------
# Individual rules. Each takes the evaluation context and returns a RuleHit or
# None. Keeping them separate means a rule can be read, cited, and changed on
# its own — which is what an auditor asking "why did this need my signature?"
# actually needs.
#
# `confidence` is a confidence.Assessment (or None when the invoice could not
# be scored).
# ---------------------------------------------------------------------------

def _rule_authority_limit(amount, features, confidence, extraction_ok):
    if amount is None:
        return None
    if amount >= DUAL_CONTROL_LIMIT:
        return RuleHit(
            "R-01", "Delegated authority limit", "dual_control",
            f"${amount:,.2f} is at or above the ${DUAL_CONTROL_LIMIT:,.0f} dual-control limit.",
        )
    if amount >= CONTROLLER_LIMIT:
        return RuleHit(
            "R-01", "Delegated authority limit", "controller_review",
            f"${amount:,.2f} exceeds the ${CONTROLLER_LIMIT:,.0f} manager limit.",
        )
    if amount >= MANAGER_LIMIT:
        return RuleHit(
            "R-01", "Delegated authority limit", "manager_review",
            f"${amount:,.2f} exceeds the ${MANAGER_LIMIT:,.0f} auto-approval limit.",
        )
    return None


def _rule_threshold_structuring(amount, features, confidence, extraction_ok):
    if amount is None:
        return None
    floor = REPORTING_THRESHOLD * STRUCTURING_BAND
    if floor <= amount < REPORTING_THRESHOLD:
        gap = REPORTING_THRESHOLD - amount
        return RuleHit(
            "R-02", "Threshold structuring", "controller_review",
            f"${amount:,.2f} sits ${gap:,.2f} below the ${REPORTING_THRESHOLD:,.0f} "
            f"reporting threshold, consistent with structuring.",
        )
    return None


def _rule_duplicate_payment(amount, features, confidence, extraction_ok):
    """
    Graded response to the similarity score. A near-match is worth a look; a
    confirmed match is worth a controller; a confirmed match above the
    reporting threshold is worth two signatures.
    """
    score = features.get("duplicate_score", 0.0)
    if score is None or score < NEAR_DUPLICATE:
        return None

    match_id = features.get("duplicate_match_id")
    against = f" against {match_id}" if match_id else ""

    if score < CONFIRMED_DUPLICATE:
        return RuleHit(
            "R-03", "Possible duplicate payment", "manager_review",
            f"{score:.0%} similarity{against}, close enough to check before paying, "
            f"below the {CONFIRMED_DUPLICATE:.0%} confirmed-duplicate threshold.",
        )

    if amount is not None and amount >= REPORTING_THRESHOLD:
        return RuleHit(
            "R-03", "Duplicate payment risk", "dual_control",
            f"{score:.0%} similarity{against}, and ${amount:,.2f} clears the "
            f"${REPORTING_THRESHOLD:,.0f} reporting threshold.",
        )

    return RuleHit(
        "R-03", "Duplicate payment risk", "controller_review",
        f"{score:.0%} similarity{against}, effectively the same invoice already on file.",
    )


def _rule_model_confidence(amount, features, confidence, extraction_ok):
    """Route on where the ensemble score lands, not on a binary flag."""
    if confidence is None:
        return None

    if confidence.band == "high":
        return RuleHit(
            "R-04", "High model confidence", "controller_review",
            f"Ensemble risk score {confidence.mean:.0%} falls in the "
            f"{confidence.label} band ({confidence.summary()}).",
        )
    if confidence.band == "elevated":
        return RuleHit(
            "R-04", "Elevated model confidence", "manager_review",
            f"Ensemble risk score {confidence.mean:.0%} falls in the "
            f"{confidence.label} band ({confidence.summary()}).",
        )
    return None


def _rule_amount_outlier(amount, features, confidence, extraction_ok):
    z = features.get("amount_zscore_norm")
    if z is None or z < OUTLIER_ZSCORE:
        return None
    return RuleHit(
        "R-05", "Off-pattern amount", "manager_review",
        f"Normalized amount z-score {z:.2f} is above the {OUTLIER_ZSCORE:.2f} "
        f"threshold for this vendor's history.",
    )


def _rule_weekend_posting(amount, features, confidence, extraction_ok):
    if features.get("weekend_flag", 0) < 1.0:
        return None
    if amount is not None and amount < DE_MINIMIS:
        return None  # off-cycle, but too small to spend a reviewer on
    return RuleHit(
        "R-06", "Off-cycle posting", "manager_review",
        f"Posted on a weekend, above the ${DE_MINIMIS:,.0f} de-minimis.",
    )


def _rule_rushed_terms(amount, features, confidence, extraction_ok):
    d = features.get("days_to_due_norm")
    if d is None or d > RUSHED_TERMS:
        return None
    return RuleHit(
        "R-07", "Compressed payment terms", "manager_review",
        f"Payment window is in the shortest {RUSHED_TERMS:.0%} of observed terms, "
        f"time pressure is a common override tactic.",
    )


def _rule_incomplete_extraction(amount, features, confidence, extraction_ok):
    if extraction_ok:
        return None
    return RuleHit(
        "R-08", "Incomplete extraction", "manager_review",
        "One or more required fields could not be read from the document, so the "
        "invoice cannot be cleared automatically.",
    )


def _rule_model_disagreement(amount, features, confidence, extraction_ok):
    """
    Three models splitting on an invoice is a signal about that invoice.
    Averaging them into one number is useful for routing, but it would hide
    exactly the cases where the evidence is genuinely ambiguous — so the split
    is reported and routed on separately.

    Only meaningful when at least one model actually calls the invoice risky;
    a wide spread entirely below the review threshold is not a disagreement
    worth a reviewer's time.
    """
    if confidence is None or not confidence.is_split:
        return None
    if max(confidence.probabilities.values()) < 0.50:
        return None

    detail = ", ".join(
        f"{name} {p:.0%}" for name, p in sorted(confidence.probabilities.items())
    )
    return RuleHit(
        "R-09", "Split model decision", "manager_review",
        f"Models disagree by {confidence.spread:.0%} ({detail}); the ensemble mean "
        f"alone would not represent this invoice.",
    )


RULES = [
    _rule_authority_limit,
    _rule_threshold_structuring,
    _rule_duplicate_payment,
    _rule_model_confidence,
    _rule_amount_outlier,
    _rule_weekend_posting,
    _rule_rushed_terms,
    _rule_incomplete_extraction,
    _rule_model_disagreement,
]


def evaluate(amount, features=None, confidence=None, extraction_ok=True):
    """
    Route one invoice.

    amount        — invoice amount, or None if it couldn't be extracted
    features      — the 5 engineered risk features (dict); missing keys are
                    skipped. May also carry `duplicate_match_id` for citation.
    confidence    — a confidence.Assessment, or None when the invoice could
                    not be scored
    extraction_ok — False when any required field failed to extract

    Returns a RoutingDecision carrying the route and every rule that demanded it.
    """
    features = features or {}
    hits = []
    for rule in RULES:
        hit = rule(amount, features, confidence, extraction_ok)
        if hit is not None:
            hits.append(hit)

    route = "auto_approve"
    for hit in hits:
        route = _higher(route, hit.route)

    # Show the rules that actually drove the outcome first.
    hits.sort(key=lambda h: (-ROUTE_ORDER.index(h.route), h.code))

    meta = ROUTE_META[route]
    return RoutingDecision(
        route=route,
        label=meta["label"],
        approver=meta["approver"],
        tier=meta["tier"],
        sla_hours=meta["sla_hours"],
        badge=meta["badge"],
        rules=hits,
    )


def escalate_route(route):
    """Next route up the ladder; the top rung escalates to itself."""
    idx = ROUTE_ORDER.index(route)
    return ROUTE_ORDER[min(idx + 1, len(ROUTE_ORDER) - 1)]
