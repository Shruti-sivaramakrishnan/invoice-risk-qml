# Roadmap

This project was built in four phases, each adding a layer on top of the classification pipeline described in the [README](README.md). All four are complete; this document records what was built and why, not what's planned.

---

## Phase 1 — Approval routing, audit trail, review queue

**Goal:** turn a risk score into an actual decision about who has to sign off on an invoice, and make that decision reproducible and auditable.

- Nine deterministic policy rules (`R-01`–`R-09`) in [`src/approval_rules.py`](src/approval_rules.py), covering delegated authority limits, threshold structuring, duplicate-payment risk, model confidence, amount outliers, weekend postings, compressed payment terms, incomplete extraction, and split model decisions.
- A four-rung approval ladder (auto-approve → manager → controller → dual control), where an invoice takes the highest route any fired rule demands.
- A review queue backed by SQLite ([`src/audit_store.py`](src/audit_store.py)), with reviewer actions (approve, reject, request information, escalate) enforced server-side: a decided invoice can't be reopened, and dual control requires two different approvers.
- An append-only, hash-chained audit trail: every intake, routing decision, and reviewer action is logged, each entry hashing the one before it, verified on every app load.

## Phase 2 — Graded risk scores

**Goal:** replace binary risky/clean verdicts with scores a reviewer can actually triage on.

- Duplicate detection moved from an exact vendor+amount match to a graded similarity score ([`src/duplicate_matching.py`](src/duplicate_matching.py)), weighting vendor identity, amount closeness, and date proximity, so near-duplicates and confirmed duplicates route differently instead of collapsing into one flag.
- Confidence bands ([`src/confidence.py`](src/confidence.py)) turn the three models' raw probabilities into an ensemble mean plus a spread, banded into Cleared / Low / Elevated / High, so a 0.51 and a 0.97 read as the different situations they are.
- Split decisions (models disagreeing by 50 points or more) are surfaced and routed on their own, rather than averaged away.

## Phase 3 — Notifications and KPI dashboard

**Goal:** show what an approver would actually be told, and give the process operational metrics instead of just a queue.

- A simulated notification preview ([`src/notifications.py`](src/notifications.py)) builds the subject line, recipient, priority, cited rules, and SLA a real alert would carry, from the same routing decision that queued the invoice, and logs it as an audit event at the moment of routing.
- A KPI dashboard ([`src/dashboard.py`](src/dashboard.py)) computes touchless processing rate (per-run and cumulative), the flagged-vs-cleared trend across batch runs, and a false-positive rate built from the reviewer's own decisions rather than dataset labels, since a real deployment never has ground truth to check against.
- Fixed a rendering defect where batch result tables and card components were displayed as literal HTML text rather than parsed markup, caused by Streamlit's markdown parser misreading indented multi-line HTML as a code block.

## Phase 4 — Operational intelligence and integration

**Goal:** let the system learn from what reviewers actually decide, track risk at the vendor level instead of only per-invoice, and let other systems use the pipeline without a browser.

- **Vendor risk profiles** — `vendor_risk_summary()` in [`src/audit_store.py`](src/audit_store.py) aggregates the queue by vendor: total invoices, flagged count and rate, how many flagged invoices were ultimately approved, and how many of a vendor's last 5 invoices were flagged. A `vendor_profiles` table persists running counters independent of how much queue history is loaded.
- **Human-in-the-loop retraining** — every reviewer approve/reject is recorded as a labeled example against the invoice's feature vector. [`src/retrain.py`](src/retrain.py) fits a fresh Logistic Regression and RBF-SVM once 10 or more labeled decisions have accumulated, on demand from a button on the Dashboard tab rather than automatically.
- **API endpoint** — [`src/api.py`](src/api.py) exposes the same scoring pipeline the app uses (`/score`, `/batch`, `/health`) over FastAPI, so another system can submit an invoice and get a routing decision back programmatically, landing in the same queue and audit trail as an invoice scored through the UI.

### Also completed: presentation layer

On top of Phase 4's functional work, the app's UI was rebuilt from a dark, tool-like interface to a light-themed landing page (hero section, a "What It Does" walkthrough of the three pipeline phases) sitting above the six working tabs, which are unchanged. This included a full contrast audit fixing text that was illegible under a visitor's dark-mode browser preference, independent of the app's own color scheme.

---

## What's out of scope

Documented in the README's [Scope & limitations](README.md#scope--limitations) section: the 40-invoice dataset size, balanced-ish class distribution, and intentionally-detectable (non-adversarial) synthetic risk patterns are all deliberate choices for a demo-scale project, not gaps in the roadmap above.
