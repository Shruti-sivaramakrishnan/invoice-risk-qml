"""
retrain.py
Human-in-the-loop model retraining pipeline. Uses reviewer feedback
(approved/rejected decisions) to periodically retrain and improve models.
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from audit_store import load_feedback_for_retraining, get_feedback_summary

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "outputs" / "retrained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RETRAINING_THRESHOLD = 10


def extract_features_from_json(features_json):
    """Parse features JSON and return as list."""
    if not features_json:
        return None
    try:
        features_dict = json.loads(features_json)
        return [
            features_dict.get("amount_zscore_norm", 0),
            features_dict.get("threshold_proximity", 0),
            features_dict.get("duplicate_score", 0),
            features_dict.get("weekend_flag", 0),
            features_dict.get("days_to_due_norm", 0),
        ]
    except (json.JSONDecodeError, TypeError):
        return None


def prepare_training_data():
    """
    Load reviewed invoices and prepare training data from reviewer feedback.

    Returns (X, y) where X is feature matrix and y is binary labels (1=risky, 0=clean).
    Returns (None, None) if insufficient feedback.
    """
    feedback_df = load_feedback_for_retraining()

    if feedback_df.empty or len(feedback_df) < RETRAINING_THRESHOLD:
        return None, None

    X = []
    y = []

    for _, row in feedback_df.iterrows():
        features = extract_features_from_json(row["features_json"])
        if features is not None:
            X.append(features)
            y.append(row["true_label"])

    if not X:
        return None, None

    return np.array(X), np.array(y)


def train_retrained_models():
    """
    Train LogisticRegression and SVM on feedback data.
    Returns dict with trained models and metadata, or None if insufficient data.
    """
    X, y = prepare_training_data()

    if X is None or len(X) < RETRAINING_THRESHOLD:
        return None

    summary = get_feedback_summary()
    logger.info(f"Retraining on {summary['total']} reviewed invoices")

    models = {}

    try:
        log_reg = LogisticRegression(random_state=42, max_iter=1000)
        log_reg.fit(X, y)
        models["logistic_regression"] = log_reg
    except Exception as e:
        logger.warning(f"Failed to train LogisticRegression: {e}")

    try:
        svm = SVC(kernel="rbf", C=10, probability=True, random_state=42)
        svm.fit(X, y)
        models["rbf_svm"] = svm
    except Exception as e:
        logger.warning(f"Failed to train RBF-SVM: {e}")

    if not models:
        return None

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "timestamp": timestamp,
        "feedback_count": summary["total"],
        "false_positives": summary["false_positives"],
        "true_positives": summary["true_positives"],
        "training_samples": len(X),
        "models_trained": list(models.keys()),
    }

    return {
        "models": models,
        "metadata": metadata,
    }


def save_retrained_models(result):
    """Save trained models and metadata to disk."""
    if not result:
        return None

    timestamp = result["metadata"]["timestamp"].replace(":", "-").replace(".", "-")
    model_path = MODELS_DIR / f"models_{timestamp}.pkl"

    try:
        with open(model_path, "wb") as f:
            pickle.dump(result, f)
        logger.info(f"Saved retrained models to {model_path}")
        return str(model_path)
    except Exception as e:
        logger.error(f"Failed to save retrained models: {e}")
        return None


def get_latest_retrained_models():
    """Load the most recent retrained models, if available."""
    pkl_files = sorted(MODELS_DIR.glob("models_*.pkl"), reverse=True)
    if not pkl_files:
        return None

    try:
        with open(pkl_files[0], "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load retrained models: {e}")
        return None


def retraining_readiness():
    """
    Check if enough feedback has been collected for retraining.
    Returns dict with readiness status and feedback counts.
    """
    summary = get_feedback_summary()
    return {
        "ready": summary["ready_for_retraining"],
        "total_feedback": summary["total"],
        "threshold": RETRAINING_THRESHOLD,
        "progress": min(1.0, summary["total"] / RETRAINING_THRESHOLD),
    }
