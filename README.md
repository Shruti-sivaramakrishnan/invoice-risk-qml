# QML-Powered Automated Invoice Intake & Risk Classification Pipeline

A prototype pipeline that ingests invoice PDFs, extracts structured fields, engineers audit-relevant risk features, and classifies invoices as **risky** or **clean** — comparing a classical machine learning approach against a quantum kernel method  then routes each invoice for approval, queues what needs a human, and records every step in an append-only audit trail.

Built to explore whether quantum kernel methods offer any practical edge over classical models on a small-scale, audit-style tabular classification task, using domain framing drawn from real invoice-audit risk indicators (duplicate payments, threshold structuring near reporting limits, weekend postings, and amount outliers).

See [ROADMAP.md](ROADMAP.md) for how the project was built up in phases, from routing and audit trail through vendor risk tracking, human-in-the-loop retraining, and the API endpoint.

---

## Why this problem

In audit and accounts-payable review, catching risky invoices *before* payment is far cheaper than catching them after (chasing repayments, reversing entries, flagging control failures in a later audit). This project targets that prevention window: given an incoming invoice, flag it for review before it's paid.

## Pipeline overview

```
Invoice PDFs → Field Extraction → Feature Engineering → Classification → Approval Routing → Review Queue
  (40 docs)     (pdfplumber)       (5 normalized         (3 models,       (9 policy rules)   (+ audit trail)
                                     risk features)        banded scores)
```

1. **Synthetic data generation** — 40 invoice PDFs generated with realistic vendor/amount/date fields, with known risk patterns deliberately injected (duplicates, threshold structuring, weekend postings, amount outliers) so results can be checked against ground truth.
2. **Field extraction** — `pdfplumber` text extraction + regex parsing pulls invoice number, vendor, amount, posting date, and due date from each PDF.
3. **Feature engineering** — 5 features, each normalized to `[0, 1]`:
   - `amount_zscore_norm` — how unusual the amount is relative to that vendor's typical invoices
   - `threshold_proximity` — how close the amount sits to a $10,000 reporting/approval threshold (a classic structuring red flag)
   - `duplicate_score` — how closely the invoice resembles the nearest other invoice on file (see below)
   - `weekend_flag` — whether the invoice was posted on a Saturday/Sunday
   - `days_to_due_norm` — normalized payment term length
4. **Classification** — three models trained and compared on the same train/test split:
   - Logistic Regression (baseline)
   - RBF-kernel SVM (classical, tuned `C=10`)
   - **Quantum Kernel SVM** — 5-qubit angle-embedding feature map (ZZFeatureMap-style, with entangling layers) implemented in [PennyLane](https://pennylane.ai/), used to compute a quantum kernel matrix fed into a classical SVM
5. **Approval routing** — a scored invoice is routed to an approver by deterministic policy rules (see below), then queued for review and recorded in an audit trail.
6. **Notification preview & KPI dashboard** — each routed invoice gets a simulated approver alert logged alongside it, and a dashboard tracks touchless processing rate, the flagged-vs-cleared trend across batch runs, false-positive rate, and per-vendor risk.
7. **Feedback loop & API** — reviewer decisions double as training labels for periodic model retraining, and the same scoring pipeline is exposed over HTTP so another system can submit an invoice and get a routing decision back.

## Duplicate screening

An exact vendor+amount match catches only the laziest duplicate payment. The ones that actually get paid twice differ in small ways: the vendor is keyed differently on re-entry (`Kaveri Suppliers` vs `Kaveri Suppliers Ltd.`), the amount moves by a rounding or a line-item correction, and the second submission lands days or weeks after the first.

[`src/duplicate_matching.py`](src/duplicate_matching.py) scores every invoice against every other one and keeps the closest match:

```
score = vendor_similarity × (0.75 × amount_similarity + 0.25 × date_proximity)
```

- **Vendor** — normalized (lowercased, punctuation stripped, legal-form suffixes like *Ltd/Pvt/Inc* removed) then compared as strings, so spelling variants of the same vendor collapse to 1.0.
- **Amount** — 1.0 for identical, decaying to 0 at 10% apart.
- **Date** — 1.0 for same-day, decaying to 0 at 90 days apart.

Vendor gates the whole score deliberately. Two invoices from genuinely different vendors that happen to share an amount and a date are not a duplicate-payment risk; an additive weighting would score them around 0.6 and bury the real matches.

Scores at or above **0.80** are screened as possible duplicates; at or above **0.95** they're treated as the same invoice submitted twice. On the sample set this recovers both deliberately injected duplicate pairs at 1.00, with the next-highest unrelated pair at 0.61 — and no false positives above the screening threshold. The closest match per invoice is written to `data/processed/duplicate_matches.csv` so a flag can be traced to what it matched.

## Confidence bands

A binary risky/clean flag throws away the part a reviewer needs. An invoice scored 0.51 and one scored 0.97 both come back "risky", and a queue sorted on that flag can't tell them apart. [`src/confidence.py`](src/confidence.py) keeps the score and bands it:

| Band | Ensemble score | Meaning |
|---|---|---|
| Cleared | 0–25% | Well below the review threshold |
| Low | 25–50% | Below the review threshold, but not comfortably |
| Elevated | 50–75% | Above the review threshold |
| High | 75–100% | Strong risk signal across the feature set |

The ensemble score is the **mean** of the three models rather than a majority vote, so a single model's near-certainty still moves the result. The **spread** is carried alongside it: when the models differ by 50 points or more — one model at least twice as concerned as another — the invoice is reported as a *split decision* and routed on that basis separately (`R-09`). Three models splitting on an invoice is a signal about that invoice, and averaging them into one number would hide exactly the cases where the evidence is ambiguous.

**What the bands actually look like on this dataset**, which is less flattering than the table above suggests:

- The ensemble mean stays roughly within **0.30–0.65** across the 40 sample invoices, so the *High* band is never reached and almost everything lands in *Low* or *Elevated*. The band boundaries are fixed quartiles of the probability scale rather than percentiles of this sample, deliberately: tuning them until all four bands filled would make the demo look better and mean less.
- The three models disagree substantially. Median spread is around **0.45**, and most invoices have at least one model on each side of the 0.5 line; **12 of 40** clear the 0.50 split threshold.

Both are findings about three weak models trained on 28 rows, not properties of the banding, and they're reported rather than tuned away. A split rate near a third is the honest read on how much the ensemble mean is worth at this scale — and it's the main argument for carrying the spread alongside it instead of collapsing to a single number.

## Approval workflow

Scoring an invoice only answers *how risky does this look*. Paying it needs an answer to *who has to sign off*, and that answer has to survive an auditor asking why. So routing is rules-based, not model-based: it's a pure function of the invoice fields, the engineered features, and the model confidence — reproducible, and able to cite the rule behind every decision.

**Approval ladder.** Each rule names the lowest route it will accept; the invoice takes the highest route any fired rule demands, so routing can only escalate. If nothing fires, the invoice auto-approves and never reaches a human.

| Route | Approver | Triggered by |
|---|---|---|
| Auto-approve | — | No rule fired |
| Manager review | AP Manager | ≥ $2,500 · elevated confidence band · split model decision · possible duplicate (0.80–0.95) · off-pattern amount · weekend posting · compressed payment terms · failed extraction |
| Controller review | Financial Controller | ≥ $10,000 · threshold structuring · confirmed duplicate · high confidence band |
| Dual control | Controller + Internal Audit | ≥ $50,000 · confirmed duplicate above the reporting threshold |

The nine rules (`R-01`–`R-09`) live in [`src/approval_rules.py`](src/approval_rules.py), each with the dollar figures an AP policy document would state. `R-02` (threshold structuring) fires on amounts sitting *just below* the $10,000 reporting threshold, and `R-08` guarantees an invoice whose fields couldn't be read is never auto-approved.

Graded inputs get graded responses. `R-03` routes a 0.88-similarity near-match to a manager to be checked, but a 0.99 match above the reporting threshold to dual control — one threshold would have collapsed both into the same response. `R-04` routes on which confidence band the ensemble lands in rather than on a binary flag, and `R-09` routes split decisions on their own.

**Review queue.** Routed invoices land in a queue ordered by required authority, then by risk score within a route. A reviewer can **approve**, **reject**, **request information**, or **escalate** (which raises the route one rung and returns the invoice to the queue). Two controls are enforced in the store rather than the UI: a decided invoice cannot be reopened, and a dual-control invoice requires two *different* approvers — the reviewer who gives the first sign-off is refused the second.

**Audit trail.** Every intake, routing decision, and reviewer action is appended to a hash-chained log in SQLite (`data/ledger.db`), where each entry hashes the one before it. Queue updates and their audit events are written in the same transaction, so the queue can never show a state the trail can't account for, and a refused action leaves no entry at all. The app re-verifies the chain on every load and reports the exact event where it breaks — an audit log that can be quietly rewritten isn't evidence of anything.

**A note on the touch rate.** Running the policy over the 40 sample invoices auto-approves only 2 of them. That's the dataset, not the thresholds: the synthetic invoices were generated to cluster around the $10,000 reporting threshold (median ~$7,000), so 32 of 40 clear the $2,500 auto-approval limit on amount alone. A real AP population is dominated by small invoices, where the same rules would clear the large majority without a human.

## Notification preview

Nothing in this project sends an email or a Slack message. What [`src/notifications.py`](src/notifications.py) does is build the message a real integration *would* send, from the same routing decision that put the invoice in the queue — subject line, recipient, priority, the rules cited as reasons, and the SLA — and render it as a preview styled like the message itself, not just another table row. That preview appears in the **Single Invoice** tab before anything is submitted, and again in the **Review Queue** tab against the invoice's current route.

Recipients are derived from the route, not looked up anywhere real: `manager_review` → an AP-manager address, `controller_review` → a controller address, `dual_control` → both a controller and internal audit, since that route needs two different sign-offs. Every address sits on `yourcompany.example`, the domain IANA reserves for documentation (RFC 2606), so nothing here could resolve to an inbox even by accident. Auto-approved invoices get a non-actionable placeholder rather than nothing — "no approver is notified" is itself information worth previewing.

Every routed invoice also gets a `notification.generated` audit event at the moment it's queued, recording exactly what the alert said — so the audit trail shows what an approver would have been told, not just the route they were assigned.

## KPI dashboard

[`src/dashboard.py`](src/dashboard.py) computes three operational metrics from the queue and a new `batch_runs` history table, rendered on the **Dashboard** tab:

- **Touchless processing rate** — the cumulative share of every submitted invoice that never reached a human, plus the same rate per batch run and running cumulatively across runs, since a single run's rate is one data point and the cumulative line is what shows whether automation is actually holding up as more volume runs through it.
- **Flagged vs. cleared trend across batch runs** — each run from the Batch Intake tab (a single-invoice submission doesn't count as a "batch") as a proportional cleared/flagged bar, plus the exact counts in a table underneath.
- **False-positive rate** — of the invoices the policy routed for review that have since been decided, the share a reviewer *approved* rather than rejected: a flag that turned out not to be a problem. This is computed from the **reviewer's own decision**, not the dataset's known labels, even though the 40-invoice demo set happens to have them — a real invoice never carries a ground-truth label, so a real deployment has no other signal to measure this against, and building the dashboard around a signal that only exists in the demo would make it useless the moment it saw a real invoice. A `coverage` figure (decided ÷ flagged) travels alongside the rate, and the dashboard says so explicitly when it's built on fewer than 5 decisions, because a rate from a handful of decisions swings hard and shouldn't be read as settled.

## Vendor risk profiles

A single invoice's risk score is a point-in-time read. Whether a *vendor* is a recurring source of flagged invoices is a pattern that only shows up across their submission history, and that's a different question the queue alone can't answer.

`vendor_risk_summary()` in [`src/audit_store.py`](src/audit_store.py) aggregates the current queue by vendor: total invoices submitted, how many were flagged for review, the flagged rate, how many of those flagged invoices were ultimately approved (a per-vendor false-positive count), and how many of the vendor's most recent 5 invoices were flagged, an "N of last 5" read that a reviewer can act on directly. A `vendor_profiles` table also persists running per-vendor counters as invoices are submitted and decided, so the same numbers survive independently of how much queue history is loaded at once.

This renders as a table on the **Dashboard** tab, under **Vendor Risk Profiles**.

## Human-in-the-loop retraining

The three models are trained once, offline, on the 40-invoice sample set. They never see what a reviewer actually decides. `src/retrain.py` closes that loop: every **approve** or **reject** in the Review Queue is recorded as a labeled example (rejected = risky, approved = a false positive) against the invoice's stored feature vector, via `_record_feedback()` in [`src/audit_store.py`](src/audit_store.py).

Once 10 or more labeled decisions have accumulated, `retrain.train_retrained_models()` fits a fresh Logistic Regression and RBF-SVM on that reviewer-labeled set and `save_retrained_models()` writes the result to `outputs/retrained_models/` with its training metadata (sample count, false-positive/true-positive split, timestamp). This is manual and on-demand rather than automatic: the **Dashboard** tab shows feedback-collection progress toward the threshold and a **Retrain models** button, so retraining happens when a maintainer chooses to run it, not silently in the background.

The quantum kernel SVM is intentionally not part of this loop: retraining it means recomputing an *n × n* quantum kernel matrix, expensive enough (see Results below) that doing it automatically on every batch of reviewer feedback isn't a reasonable default for a demo-scale project.

## API endpoint

Everything the Streamlit app does when it scores an invoice, feature computation, model scoring, confidence banding, policy routing, is also reachable over HTTP, so another system can submit an invoice and get a routing decision back without a person in a browser. [`src/api.py`](src/api.py) is a small FastAPI service:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/score` | POST | Score one invoice, return its risk features, confidence band, routing decision, and (for routed invoices) the notification preview |
| `/batch` | POST | Score a list of invoices in one request |
| `/docs` | GET | Interactive OpenAPI docs (generated automatically by FastAPI) |

A request to `/score` runs the same `submit_invoice()` path the app uses, so an invoice scored through the API lands in the same queue and audit trail as one scored through the UI, there's one pipeline, not two. Run it with:

```bash
uvicorn src.api:app --reload --port 8000
```

## Results

| Model               | Accuracy | Precision | Recall | F1    | Train Time (s) |
|---------------------|----------|-----------|--------|-------|-----------------|
| Logistic Regression | 0.583    | 0.500     | 0.400  | 0.444 | 0.013           |
| RBF-SVM              | 0.833    | 0.714     | 1.000  | 0.833 | 0.008           |
| Quantum Kernel SVM  | 0.833    | 0.800     | 0.800  | 0.800 | 5.408           |

*(Full metrics in [`outputs/metrics_comparison.csv`](outputs/metrics_comparison.csv), plots in [`outputs/figures/`](outputs/figures/).)*

**Honest takeaway:** the classical RBF-SVM still edges the quantum kernel SVM on F1 (0.833 vs 0.800) and trains roughly 675x faster. They now tie on accuracy, and the quantum kernel takes a different precision/recall trade: RBF-SVM catches every risky invoice in the test set at the cost of more false positives, the quantum kernel is more precise and misses one. Neither is "winning" in a meaningful sense — the test set is 12 invoices, so a single invoice moves accuracy by 8 percentage points, and differences this size are well inside the noise.

The one change worth reporting is *why* the quantum numbers moved (F1 0.400 → 0.800 against the previous binary-duplicate feature set): replacing the exact-match duplicate flag with a graded similarity score turned a feature that was zero for 38 of 40 invoices into one that varies across the whole range. Angle-embedding maps each feature to a rotation, so a near-constant input contributes a near-constant rotation and the kernel gets almost nothing from that qubit. That is a statement about feature encoding, not about quantum advantage — the classical models were largely indifferent to the same change.

Quantum kernel methods compute an *n × n* kernel matrix, so cost scales quadratically with dataset size, and near-term simulators carry heavy per-circuit-evaluation overhead that classical kernels don't. The interesting research question isn't "did quantum win on 40 rows" — it's *whether the quantum feature map captures decision boundaries the classical kernels miss*, which is better tested via kernel-alignment analysis than raw accuracy on a small sample.

## Scope & limitations

- **Dataset size (n=40)** is a deliberate choice, not an oversight: the quantum kernel matrix scales as O(n²) circuit evaluations, so keeping the dataset small keeps quantum training time tractable on a classical simulator.
- **Balanced-ish classes (~45% risky)** are used instead of realistic real-world rarity (fraud/errors are usually a small minority). At this sample size, realistic class rarity would risk a test set with zero positive examples, making recall unmeasurable. This is a measurability trade-off, not a claim that real invoice populations look like this.
- **Risk patterns are intentionally detectable.** The synthetic anomalies (structuring, duplicates, weekend postings, outliers) are common-error and rule-of-thumb audit signals — not adversarial fraud designed to evade detection. This system is scoped to catch unsophisticated errors and common risk indicators, not to defeat a deliberately evasive bad actor.

## Project structure

```
├── data/
│   ├── invoices/              # 40 generated invoice PDFs
│   ├── processed/             # extracted fields, engineered features, duplicate matches
│   ├── ground_truth.csv       # labels for evaluation
│   └── ledger.db              # review queue + audit trail (created at runtime)
├── src/
│   ├── generate_invoices.py   # synthetic invoice + label generation
│   ├── extract_fields.py      # PDF → structured fields
│   ├── feature_engineering.py # structured fields → 5 normalized risk features
│   ├── train_and_compare.py   # classical + quantum model training & comparison
│   ├── pipeline_utils.py      # single-invoice extraction + feature helpers
│   ├── duplicate_matching.py  # similarity-based duplicate detection
│   ├── confidence.py          # confidence bands over model risk scores
│   ├── approval_rules.py      # rules-based approval routing (R-01 … R-09)
│   ├── audit_store.py         # review queue, batch-run history, hash-chained audit trail (SQLite)
│   ├── notifications.py       # simulated approver notification preview
│   ├── dashboard.py           # KPI computation (touchless rate, trend, false-positive rate, vendor risk)
│   ├── retrain.py             # human-in-the-loop retraining from reviewer feedback
│   ├── api.py                 # FastAPI endpoint exposing the scoring pipeline
│   ├── test_scoring.py        # tests for duplicate matching + confidence bands
│   ├── test_reporting.py      # tests for the dashboard + notification preview
│   ├── test_approval_workflow.py  # tests for routing, queue, batch runs, audit trail
│   └── app.py                 # Streamlit app
├── outputs/
│   ├── figures/                # comparison charts, confusion matrices
│   ├── metrics_comparison.csv
│   └── retrained_models/       # retrained model checkpoints (created at runtime)
├── requirements.txt
└── README.md
```

## Running it yourself

```bash
pip install -r requirements.txt
python src/generate_invoices.py      # generates 40 invoice PDFs + ground truth
python src/extract_fields.py         # extracts fields from the PDFs
python src/feature_engineering.py    # builds the 5 risk features
python src/train_and_compare.py      # trains all 3 models, saves metrics + plots
```

Then launch the app to score, route, and review invoices:

```bash
streamlit run src/app.py
```

The app opens on a landing page (a hero introducing the project and a "What It Does" section walking through the three pipeline phases) before the working tool starts. The tool itself has six tabs: **Batch Intake** (score + route a batch, writing it to the queue, the run history, and the audit trail), **Single Invoice** (includes a notification preview for the routing decision), **Review Queue** (reviewer actions, each with its own notification preview), **Dashboard** (touchless rate, the flagged-vs-cleared trend, false-positive rate, vendor risk profiles, and retraining status), **Audit Trail** (event log + chain verification), and **Model Comparison**. Set a reviewer name in the sidebar before actioning anything — every decision is attributed.

To expose the same scoring pipeline over HTTP instead:

```bash
uvicorn src.api:app --reload --port 8000
```

Interactive docs are then available at `http://localhost:8000/docs`.

To run the approval-workflow tests (they use a throwaway database and never touch `data/ledger.db`):

```bash
python src/test_scoring.py            # duplicate matching + confidence bands
python src/test_reporting.py          # KPI dashboard + notification preview
python src/test_approval_workflow.py  # routing, queue transitions, batch runs, audit trail
```

## Tech stack

Python · PennyLane (quantum simulation) · scikit-learn (classical ML) · pdfplumber (PDF text extraction) · reportlab (synthetic PDF generation) · SQLite (review queue + audit trail) · Streamlit (app) · FastAPI / Pydantic / uvicorn (API) · pandas / numpy · matplotlib

## Background

Built with an auditing background (KPMG) informing the choice of risk features — threshold structuring and duplicate-payment detection are standard audit red flags, adapted here into a quantifiable, model-ready form.

---

*Academic mini-project — quantum machine learning applied to a tabular business-risk classification task.*
