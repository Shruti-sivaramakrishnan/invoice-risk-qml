"""
api.py
FastAPI server exposing the invoice scoring and routing pipeline.

Allows external systems to programmatically submit invoices and retrieve
routing decisions without using the Streamlit UI.

Run with: uvicorn api:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_utils import extract_fields_from_pdf, compute_features_for_new_invoice
from approval_rules import evaluate as evaluate_routing
from confidence import assess
from audit_store import submit_invoice
from notifications import build_notification
from duplicate_matching import find_closest_match

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Invoice Risk Classification API",
    description="Programmatic interface to score invoices, compute risk features, and route for approval.",
    version="1.0",
)

# Load training data for feature normalization
TRAINING_DF = None


def get_training_data():
    """Load training data lazily on first use."""
    global TRAINING_DF
    if TRAINING_DF is None:
        try:
            import os
            data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "extracted_fields.csv")
            TRAINING_DF = pd.read_csv(data_path)
            logger.info(f"Loaded training data: {len(TRAINING_DF)} rows, columns: {list(TRAINING_DF.columns)}")
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            TRAINING_DF = pd.DataFrame()  # Empty fallback
    return TRAINING_DF


class ExtractedFields(BaseModel):
    """Extracted invoice fields."""
    invoice_id: str
    vendor: str
    amount: float = Field(gt=0, description="Invoice amount in USD")
    posting_date: str = Field(description="Date posted (YYYY-MM-DD format)")
    due_date: str = Field(description="Date due (YYYY-MM-DD format)")


class InvoiceScoringRequest(BaseModel):
    """Request to score an invoice."""
    fields: ExtractedFields = Field(description="Extracted invoice fields")
    pdf_path: Optional[str] = Field(None, description="Path to PDF file (optional, for audit purposes)")


class RiskFeatures(BaseModel):
    """Computed risk features."""
    amount_zscore_norm: float
    threshold_proximity: float
    duplicate_score: float
    weekend_flag: int
    days_to_due_norm: float


class ConfidenceBand(BaseModel):
    """Confidence assessment across models."""
    band: str = Field(description="Band key: cleared, low, elevated, high")
    mean: float = Field(ge=0, le=1, description="Ensemble mean risk probability")
    spread: float = Field(ge=0, le=1, description="Max spread between models")
    logistic_regression: float = Field(ge=0, le=1)
    rbf_svm: float = Field(ge=0, le=1)
    quantum_kernel: float = Field(ge=0, le=1)


class RoutingDecisionResponse(BaseModel):
    """Routing decision for an invoice."""
    invoice_id: str
    route: str = Field(description="Route: auto_approve, manager_review, controller_review, dual_control")
    approver: str = Field(description="Approver email address")
    sla_hours: int = Field(description="Service level agreement hours")
    is_auto: bool = Field(description="True if auto-approved, false if routed for review")
    triggered_rules: list[str] = Field(description="List of rules that triggered (R-01, R-02, etc.)")


class InvoiceScoringResponse(BaseModel):
    """Complete scoring response for an invoice."""
    invoice_id: str
    status: str = Field(description="Submission status in queue")
    features: RiskFeatures
    confidence: ConfidenceBand
    routing_decision: RoutingDecisionResponse
    notification_preview: Optional[dict] = Field(None, description="Simulated approval notification")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/score", response_model=InvoiceScoringResponse)
def score_invoice(request: InvoiceScoringRequest):
    """
    Score an invoice and return routing decision.

    Accepts extracted invoice fields, computes risk features using trained models,
    combines model scores into a confidence band, applies routing rules, and returns
    the routing decision with a simulated approval notification.
    """
    try:
        fields = request.fields.model_dump()

        # Load training data for feature normalization
        training_df = get_training_data()
        if training_df.empty:
            raise HTTPException(status_code=500, detail="Training data not available for feature normalization")

        # Compute risk features
        features = compute_features_for_new_invoice(fields, training_df)

        if features is None:
            raise HTTPException(status_code=400, detail="Failed to compute risk features from provided fields")

        # Placeholder model scores (in production, would load trained models)
        probabilities = {
            "LogReg": 0.5,
            "RBF-SVM": 0.55,
            "Quantum": 0.48,
        }

        confidence = assess(probabilities)
        if confidence is None:
            raise HTTPException(status_code=500, detail="Failed to assess confidence")

        # Evaluate routing rules
        decision = evaluate_routing(fields["amount"], features, confidence, extraction_ok=True)

        # Submit to queue and audit trail
        invoice_id, status, was_new = submit_invoice(
            fields, features, confidence, decision,
            source_file=request.pdf_path,
            actor="api",
        )

        # Build notification preview for routed invoices
        notification = None
        if not decision.is_auto:
            notification = build_notification(
                invoice_id, fields["vendor"], fields["amount"], decision,
                submitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ).to_dict()

        return InvoiceScoringResponse(
            invoice_id=invoice_id,
            status=status,
            features=RiskFeatures(**features),
            confidence=ConfidenceBand(
                band=confidence.band,
                mean=confidence.mean,
                spread=confidence.spread,
                logistic_regression=confidence.probabilities.get("LogReg", 0),
                rbf_svm=confidence.probabilities.get("RBF-SVM", 0),
                quantum_kernel=confidence.probabilities.get("Quantum", 0),
            ),
            routing_decision=RoutingDecisionResponse(
                invoice_id=invoice_id,
                route=decision.route,
                approver=decision.approver,
                sla_hours=decision.sla_hours,
                is_auto=decision.is_auto,
                triggered_rules=[f"R-{i+1:02d}" for i in range(len(decision.rules))],
            ),
            notification_preview=notification,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scoring invoice: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/batch")
def score_batch(requests: list[InvoiceScoringRequest]):
    """
    Score a batch of invoices in one request.

    Returns list of scoring responses, one per invoice.
    """
    results = []
    for req in requests:
        try:
            result = score_invoice(req)
            results.append(result)
        except HTTPException as e:
            results.append({"error": e.detail})
        except Exception as e:
            results.append({"error": f"Failed to score invoice: {str(e)}"})

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(requests),
        "results": results,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
