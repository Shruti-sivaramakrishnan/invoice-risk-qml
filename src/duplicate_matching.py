"""
duplicate_matching.py
Similarity-based duplicate detection for invoices.

An exact vendor+amount match catches only the laziest duplicate payment. The
ones that actually get paid twice differ in small ways: the vendor is keyed
slightly differently on re-entry ("Kaveri Suppliers" vs "Kaveri Suppliers
Ltd."), the amount moves by a rounding or a line-item correction, and the
second submission lands days or weeks after the first.

This module scores how closely an invoice resembles any other invoice on file
and returns the closest match with its component scores, so a reviewer can see
*why* something was flagged rather than being handed a bare 1.0.

Similarity is multiplicative in the vendor term:

    score = vendor_similarity * (0.75 * amount_similarity + 0.25 * date_proximity)

Vendor gates the whole score deliberately. Two invoices from genuinely
different vendors that happen to share an amount and a date are not a
duplicate-payment risk, and an additive weighting would score them around 0.6
and bury the real matches.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import pandas as pd

# An amount this far apart (as a fraction of the larger amount) scores zero.
AMOUNT_TOLERANCE = 0.10
# Postings this many days apart score zero on proximity.
DATE_WINDOW_DAYS = 90

W_AMOUNT = 0.75
W_DATE = 0.25

# Score at or above which a pair is treated as the same invoice submitted twice.
CONFIRMED_DUPLICATE = 0.95
# Score at or above which a pair is close enough that a human should look.
NEAR_DUPLICATE = 0.80

# Legal-form suffixes carry no identifying information and are the most common
# difference between two spellings of the same vendor.
_VENDOR_SUFFIXES = {
    "ltd", "limited", "inc", "incorporated", "llc", "llp", "plc", "co",
    "corp", "corporation", "company", "pvt", "private", "gmbh", "bv", "sa", "ag",
}


@dataclass(frozen=True)
class DuplicateMatch:
    """The closest invoice on file, and how close it is."""
    score: float
    matched_invoice_id: str = None
    vendor_similarity: float = 0.0
    amount_similarity: float = 0.0
    date_proximity: float = 0.0
    matched_vendor: str = None
    matched_amount: float = None
    days_apart: int = None

    @property
    def is_confirmed(self):
        return self.score >= CONFIRMED_DUPLICATE

    @property
    def is_near(self):
        return self.score >= NEAR_DUPLICATE

    def describe(self):
        if self.matched_invoice_id is None:
            return "No comparable invoice on file."
        gap = "same day" if self.days_apart == 0 else f"{self.days_apart}d apart"
        return (
            f"{self.matched_invoice_id} at {self.score:.0%} similarity "
            f"(vendor {self.vendor_similarity:.0%}, amount {self.amount_similarity:.0%}, {gap})"
        )

    def to_dict(self):
        return {
            "score": round(self.score, 4),
            "matched_invoice_id": self.matched_invoice_id,
            "vendor_similarity": round(self.vendor_similarity, 4),
            "amount_similarity": round(self.amount_similarity, 4),
            "date_proximity": round(self.date_proximity, 4),
            "matched_vendor": self.matched_vendor,
            "matched_amount": self.matched_amount,
            "days_apart": self.days_apart,
        }


NO_MATCH = DuplicateMatch(score=0.0)


def normalize_vendor(name):
    """Lowercase, strip punctuation, and drop legal-form suffixes."""
    if not name or (isinstance(name, float) and pd.isna(name)):
        return ""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", str(name).lower())
    tokens = [t for t in cleaned.split() if t and t not in _VENDOR_SUFFIXES]
    return " ".join(tokens)


def vendor_similarity(a, b):
    """Similarity of two vendor names in [0, 1], suffix- and punctuation-insensitive."""
    na, nb = normalize_vendor(a), normalize_vendor(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def amount_similarity(a, b):
    """1.0 for identical amounts, decaying to 0 at AMOUNT_TOLERANCE relative difference."""
    if a is None or b is None:
        return 0.0
    a, b = float(a), float(b)
    scale = max(abs(a), abs(b), 1.0)
    relative_gap = abs(a - b) / scale
    return max(0.0, 1.0 - relative_gap / AMOUNT_TOLERANCE)


def date_proximity(a, b):
    """1.0 for same-day postings, decaying to 0 at DATE_WINDOW_DAYS apart."""
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return 0.0
    gap = abs((pd.to_datetime(a) - pd.to_datetime(b)).days)
    return max(0.0, 1.0 - gap / DATE_WINDOW_DAYS)


def pair_similarity(vendor_a, amount_a, date_a, vendor_b, amount_b, date_b):
    """Overall similarity of two invoices, and the components behind it."""
    v = vendor_similarity(vendor_a, vendor_b)
    amt = amount_similarity(amount_a, amount_b)
    d = date_proximity(date_a, date_b)
    return v * (W_AMOUNT * amt + W_DATE * d), v, amt, d


def find_closest_match(record, candidates, exclude_invoice_id=None):
    """
    Score one invoice against every candidate and return the closest match.

    record     — dict with vendor / amount / posting_date
    candidates — DataFrame of invoices already on file
    exclude_invoice_id — the record's own id, so an invoice never matches itself
    """
    if candidates is None or len(candidates) == 0:
        return NO_MATCH

    vendor, amount = record.get("vendor"), record.get("amount")
    posting_date = record.get("posting_date")
    if amount is None:
        return NO_MATCH

    best = NO_MATCH
    for _, other in candidates.iterrows():
        if exclude_invoice_id is not None and other.get("invoice_id") == exclude_invoice_id:
            continue

        score, v, amt, d = pair_similarity(
            vendor, amount, posting_date,
            other.get("vendor"), other.get("amount"), other.get("posting_date"),
        )
        if score <= best.score:
            continue

        try:
            days_apart = abs((pd.to_datetime(posting_date) - pd.to_datetime(other.get("posting_date"))).days)
        except (ValueError, TypeError):
            days_apart = None

        best = DuplicateMatch(
            score=score,
            matched_invoice_id=other.get("invoice_id"),
            vendor_similarity=v,
            amount_similarity=amt,
            date_proximity=d,
            matched_vendor=other.get("vendor"),
            matched_amount=other.get("amount"),
            days_apart=days_apart,
        )

    return best


def score_dataframe(df):
    """
    Score every invoice in a DataFrame against every other one.

    Returns (scores, matched_ids) as lists aligned to df's row order. O(n²) by
    construction — fine at this scale, and the honest shape of the problem: a
    production version would block on normalized vendor first.
    """
    scores, matched = [], []
    for _, row in df.iterrows():
        match = find_closest_match(
            {"vendor": row.get("vendor"), "amount": row.get("amount"),
             "posting_date": row.get("posting_date")},
            df,
            exclude_invoice_id=row.get("invoice_id"),
        )
        scores.append(round(match.score, 4))
        matched.append(match.matched_invoice_id)
    return scores, matched
