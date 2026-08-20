"""
confidence.py
Confidence bands over model risk scores.

A binary risky/clean flag throws away the part a reviewer actually needs. An
invoice scored 0.51 and one scored 0.97 both come back "risky", and a queue
sorted on that flag can't tell them apart — so the reviewer re-derives the
distinction by hand, from the raw probabilities, every time.

This module converts probabilities into four ordered bands, and combines the
three models into a single assessment that carries how much they agreed. The
disagreement is reportable on its own: three models splitting on an invoice is
a signal about that invoice, not noise to be averaged away.
"""

from dataclasses import dataclass

# Ordered low to high. Upper bounds are exclusive except for the top band.
BANDS = [
    {"key": "cleared",  "label": "Cleared",  "lower": 0.00, "upper": 0.25,
     "badge": "badge-auto",       "blurb": "Well below the review threshold."},
    {"key": "low",      "label": "Low",      "lower": 0.25, "upper": 0.50,
     "badge": "badge-manager",    "blurb": "Below the review threshold, but not comfortably."},
    {"key": "elevated", "label": "Elevated", "lower": 0.50, "upper": 0.75,
     "badge": "badge-controller", "blurb": "Above the review threshold."},
    {"key": "high",     "label": "High",     "lower": 0.75, "upper": 1.01,
     "badge": "badge-dual",       "blurb": "Strong risk signal across the feature set."},
]

BAND_ORDER = [b["key"] for b in BANDS]
BAND_BY_KEY = {b["key"]: b for b in BANDS}

# Spread (max - min across models) at or above which the models are treated as
# materially disagreeing rather than merely differing at the margin. Half the
# probability scale: one model is at least twice as concerned as another.
#
# Deliberately not tuned to the sample set. These three models disagree heavily
# at this data size (median spread ~0.56), so a threshold fitted to make the
# flag look selective here would be fitting the noise rather than measuring it.
SPLIT_SPREAD = 0.50


@dataclass(frozen=True)
class Assessment:
    """The three models' scores on one invoice, reduced to a single view."""
    probabilities: dict          # model name -> risk probability
    mean: float
    spread: float
    band: str                    # band key for the ensemble mean
    per_model_bands: dict        # model name -> band key
    is_split: bool

    @property
    def label(self):
        return BAND_BY_KEY[self.band]["label"]

    @property
    def badge(self):
        return BAND_BY_KEY[self.band]["badge"]

    @property
    def rank(self):
        return BAND_ORDER.index(self.band)

    def at_or_above(self, band_key):
        return self.rank >= BAND_ORDER.index(band_key)

    @property
    def agreement(self):
        if self.is_split:
            return "split"
        return "unanimous" if len(set(self.per_model_bands.values())) == 1 else "aligned"

    def summary(self):
        detail = ", ".join(f"{name} {p:.0%}" for name, p in sorted(self.probabilities.items()))
        return f"{self.label} ({self.mean:.0%} mean): {detail}"

    def to_dict(self):
        return {
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "mean": round(self.mean, 4),
            "spread": round(self.spread, 4),
            "band": self.band,
            "label": self.label,
            "per_model_bands": self.per_model_bands,
            "agreement": self.agreement,
        }


def band_for(probability):
    """Band key for a single probability."""
    p = min(max(float(probability), 0.0), 1.0)
    for band in BANDS:
        if p < band["upper"]:
            return band["key"]
    return BANDS[-1]["key"]


def band_label(key):
    return BAND_BY_KEY[key]["label"]


def band_badge(key):
    return BAND_BY_KEY[key]["badge"]


def assess(probabilities):
    """
    Combine per-model risk probabilities into one banded assessment.

    probabilities — {"LogReg": 0.62, "RBF-SVM": 0.71, "Quantum": 0.30}

    The ensemble uses the mean rather than a majority vote so that a single
    model's near-certainty still moves the result; the spread is kept
    alongside it so that averaging never hides a split decision.
    """
    if not probabilities:
        return None

    values = [float(v) for v in probabilities.values()]
    mean = sum(values) / len(values)
    spread = max(values) - min(values)

    return Assessment(
        probabilities={k: float(v) for k, v in probabilities.items()},
        mean=mean,
        spread=spread,
        band=band_for(mean),
        per_model_bands={k: band_for(v) for k, v in probabilities.items()},
        is_split=spread >= SPLIT_SPREAD,
    )
