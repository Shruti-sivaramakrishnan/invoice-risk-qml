"""
test_scoring.py
Tests for similarity-based duplicate detection and model confidence bands.

Includes an end-to-end check that the matcher recovers the duplicate pairs the
invoice generator deliberately injected, which is the only real accuracy claim
either module makes.

Run with:  python src/test_scoring.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import confidence
import duplicate_matching as dup

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GT_PATH = os.path.join(BASE_DIR, "data", "ground_truth.csv")
EXTRACTED_PATH = os.path.join(BASE_DIR, "data", "processed", "extracted_fields.csv")

PASSED, FAILED = 0, 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def test_vendor_similarity():
    print("\nVENDOR SIMILARITY")

    check("identical names score 1.0", dup.vendor_similarity("Acme Ltd", "Acme Ltd") == 1.0)
    check("legal suffix is ignored",
          dup.vendor_similarity("Kaveri Suppliers", "Kaveri Suppliers Ltd.") == 1.0,
          dup.vendor_similarity("Kaveri Suppliers", "Kaveri Suppliers Ltd."))
    check("punctuation and case are ignored",
          dup.vendor_similarity("bright logistics", "Bright  Logistics!") == 1.0)
    check("private limited variants collapse",
          dup.vendor_similarity("Bright Logistics Pvt Ltd", "Bright Logistics") == 1.0)

    typo = dup.vendor_similarity("Sunrise Stationery", "Sunrise Stationary")
    check("a single-letter typo stays high", typo > 0.90, typo)

    unrelated = dup.vendor_similarity("Sunrise Stationery", "Kaveri Suppliers")
    check("unrelated vendors score low", unrelated < 0.50, unrelated)

    check("empty vendor scores zero", dup.vendor_similarity("", "Acme") == 0.0)
    check("missing vendor scores zero", dup.vendor_similarity(None, "Acme") == 0.0)


def test_amount_and_date():
    print("\nAMOUNT + DATE SIMILARITY")

    check("identical amounts score 1.0", dup.amount_similarity(1000.0, 1000.0) == 1.0)
    check("1% apart stays high", dup.amount_similarity(1000.0, 1010.0) > 0.85,
          dup.amount_similarity(1000.0, 1010.0))
    check("beyond tolerance scores zero", dup.amount_similarity(1000.0, 1200.0) == 0.0)
    check("similarity is symmetric",
          dup.amount_similarity(1000.0, 1050.0) == dup.amount_similarity(1050.0, 1000.0))
    check("missing amount scores zero", dup.amount_similarity(None, 1000.0) == 0.0)

    check("same-day postings score 1.0",
          dup.date_proximity("2025-06-02", "2025-06-02") == 1.0)
    check("45 days apart is mid-scale",
          0.45 < dup.date_proximity("2025-06-02", "2025-07-17") < 0.55,
          dup.date_proximity("2025-06-02", "2025-07-17"))
    check("beyond the window scores zero",
          dup.date_proximity("2025-01-01", "2025-12-31") == 0.0)
    check("proximity never goes negative",
          dup.date_proximity("2020-01-01", "2025-12-31") == 0.0)


def test_pair_scoring():
    print("\nPAIR SCORING")

    exact, _, _, _ = dup.pair_similarity(
        "Acme Ltd", 500.0, "2025-03-01", "Acme Ltd", 500.0, "2025-03-01")
    check("an exact re-submission scores 1.0", abs(exact - 1.0) < 1e-9, exact)

    # Vendor gates the score: same amount and date under a different vendor is
    # not a duplicate-payment risk, and must not rank alongside real matches.
    other_vendor, _, _, _ = dup.pair_similarity(
        "Acme Ltd", 500.0, "2025-03-01", "Zenith Traders", 500.0, "2025-03-01")
    check("different vendor is heavily discounted", other_vendor < 0.40, other_vendor)
    check("different vendor scores below an exact match", other_vendor < exact)

    later, _, _, _ = dup.pair_similarity(
        "Acme Ltd", 500.0, "2025-03-01", "Acme Ltd", 500.0, "2025-04-15")
    check("same invoice resubmitted weeks later still screens as a duplicate",
          later >= dup.NEAR_DUPLICATE, later)

    typo_match, _, _, _ = dup.pair_similarity(
        "Kaveri Suppliers", 3760.74, "2025-06-02",
        "Kaveri Suppliers Ltd", 3760.74, "2025-06-05")
    check("vendor keyed differently on re-entry still screens",
          typo_match >= dup.NEAR_DUPLICATE, typo_match)

    check("score never exceeds 1.0", exact <= 1.0)


def test_closest_match():
    print("\nCLOSEST MATCH")

    candidates = pd.DataFrame([
        {"invoice_id": "INV-0001", "vendor": "Acme Ltd", "amount": 500.0, "posting_date": "2025-03-01"},
        {"invoice_id": "INV-0002", "vendor": "Zenith Traders", "amount": 9000.0, "posting_date": "2025-03-02"},
        {"invoice_id": "INV-0003", "vendor": "Acme Limited", "amount": 502.0, "posting_date": "2025-03-04"},
    ])

    record = {"invoice_id": "INV-0009", "vendor": "Acme Ltd",
              "amount": 500.0, "posting_date": "2025-03-01"}
    match = dup.find_closest_match(record, candidates)
    check("closest match is the exact one", match.matched_invoice_id == "INV-0001",
          match.matched_invoice_id)
    check("exact match is flagged confirmed", match.is_confirmed, match.score)
    check("days apart is reported", match.days_apart == 0, match.days_apart)
    check("description names the match", "INV-0001" in match.describe(), match.describe())

    # An invoice already on file must not match itself.
    self_record = {"invoice_id": "INV-0001", "vendor": "Acme Ltd",
                   "amount": 500.0, "posting_date": "2025-03-01"}
    match = dup.find_closest_match(self_record, candidates, exclude_invoice_id="INV-0001")
    check("an invoice never matches itself", match.matched_invoice_id == "INV-0003",
          match.matched_invoice_id)
    check("the near-variant is still caught", match.is_near, match.score)

    lonely = {"invoice_id": "INV-0010", "vendor": "Northwind Freight",
              "amount": 42.0, "posting_date": "2025-08-01"}
    match = dup.find_closest_match(lonely, candidates)
    check("an unrelated invoice scores below the screen", not match.is_near, match.score)

    check("empty candidate set returns no match",
          dup.find_closest_match(record, pd.DataFrame()).matched_invoice_id is None)
    check("missing amount returns no match",
          dup.find_closest_match({"vendor": "Acme", "amount": None}, candidates).score == 0.0)


def test_against_ground_truth():
    print("\nGROUND TRUTH RECALL")

    if not (os.path.exists(GT_PATH) and os.path.exists(EXTRACTED_PATH)):
        check("ground-truth files available", False, "run the pipeline scripts first")
        return

    gt = pd.read_csv(GT_PATH)
    extracted = pd.read_csv(EXTRACTED_PATH)
    injected = gt[gt["is_duplicate_of"].notna()][["invoice_id", "is_duplicate_of"]]

    scores, matched = dup.score_dataframe(extracted)
    result = pd.DataFrame({
        "invoice_id": extracted["invoice_id"], "score": scores, "matched": matched,
    })

    check("the generator injected duplicates to find", len(injected) > 0, len(injected))

    for _, row in injected.iterrows():
        found = result[result["invoice_id"] == row["invoice_id"]].iloc[0]
        check(f"{row['invoice_id']} matched to {row['is_duplicate_of']}",
              found["matched"] == row["is_duplicate_of"], found["matched"])
        check(f"{row['invoice_id']} scored as a confirmed duplicate",
              found["score"] >= dup.CONFIRMED_DUPLICATE, found["score"])

    # Every invoice not involved in an injected pair must stay below the screen,
    # or the feature is just noise with a threshold on it.
    involved = set(injected["invoice_id"]) | set(injected["is_duplicate_of"])
    false_positives = result[
        (~result["invoice_id"].isin(involved)) & (result["score"] >= dup.NEAR_DUPLICATE)
    ]
    check("no false positives above the screening threshold",
          len(false_positives) == 0, false_positives.to_dict("records"))

    graded = result["score"].nunique()
    check("the score is graded, not binary", graded > 10, graded)


def test_confidence_bands():
    print("\nCONFIDENCE BANDS")

    check("zero lands in cleared", confidence.band_for(0.0) == "cleared")
    check("just below a boundary stays low", confidence.band_for(0.4999) == "low")
    check("a boundary opens the next band", confidence.band_for(0.50) == "elevated")
    check("one lands in high", confidence.band_for(1.0) == "high")
    check("out-of-range input is clamped", confidence.band_for(1.7) == "high")
    check("negative input is clamped", confidence.band_for(-0.3) == "cleared")

    bands = [confidence.band_for(p) for p in (0.1, 0.35, 0.6, 0.9)]
    check("all four bands are reachable", bands == ["cleared", "low", "elevated", "high"], bands)

    a = confidence.assess({"LogReg": 0.80, "RBF-SVM": 0.90, "Quantum": 0.85})
    check("mean is the ensemble score", abs(a.mean - 0.85) < 1e-9, a.mean)
    check("tight agreement is not a split", not a.is_split, a.spread)
    check("band follows the mean", a.band == "high", a.band)
    check("agreement is reported as unanimous", a.agreement == "unanimous", a.agreement)
    check("at_or_above compares on the ladder",
          a.at_or_above("elevated") and a.at_or_above("high")
          and not confidence.assess({"m": 0.3}).at_or_above("elevated"))

    split = confidence.assess({"LogReg": 0.95, "RBF-SVM": 0.10, "Quantum": 0.20})
    check("a wide spread is a split", split.is_split, split.spread)
    check("split is reported as such", split.agreement == "split", split.agreement)
    check("averaging does not hide the split", split.band == "low", split.band)

    aligned = confidence.assess({"LogReg": 0.20, "RBF-SVM": 0.30, "Quantum": 0.24})
    check("models in different bands but close are aligned, not split",
          aligned.agreement == "aligned", aligned.agreement)

    check("per-model bands are kept",
          split.per_model_bands["LogReg"] == "high"
          and split.per_model_bands["RBF-SVM"] == "cleared",
          split.per_model_bands)

    check("no scores yields no assessment", confidence.assess({}) is None)

    payload = a.to_dict()
    check("assessment serializes for storage",
          payload["band"] == "high" and "probabilities" in payload, payload)
    check("round-trip through storage preserves the band",
          confidence.assess(payload["probabilities"]).band == a.band)


def main():
    print("Duplicate matching + confidence bands — tests")
    test_vendor_similarity()
    test_amount_and_date()
    test_pair_scoring()
    test_closest_match()
    test_against_ground_truth()
    test_confidence_bands()
    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
