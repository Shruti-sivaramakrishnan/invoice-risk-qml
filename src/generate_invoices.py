"""
generate_invoices.py
Generates synthetic invoice PDFs with realistic fields, and injects known
risk patterns (duplicates, threshold structuring, weekend postings, amount
outliers) so we have ground-truth labels to train/evaluate against.
"""

import os
import random
import csv
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "invoices")
os.makedirs(OUT_DIR, exist_ok=True)

VENDORS = [
    "Sundar Traders", "Bright Logistics Pvt Ltd", "Kaveri Suppliers",
    "NextGen Office Solutions", "Prime Facility Services", "Orion IT Consulting",
    "Global Freight Co", "Metro Print Works", "Sunrise Stationery",
    "Apex Maintenance Ltd",
]

N_INVOICES = 40
START_DATE = datetime(2025, 1, 1)


def random_weekday_date(avoid_weekend=True):
    d = START_DATE + timedelta(days=random.randint(0, 300))
    if avoid_weekend:
        while d.weekday() >= 5:
            d += timedelta(days=1)
    return d


def random_weekend_date():
    d = START_DATE + timedelta(days=random.randint(0, 300))
    while d.weekday() < 5:
        d += timedelta(days=1)
    return d


def make_invoice_record(inv_id, risk_type=None):
    vendor = random.choice(VENDORS)
    posting_date = random_weekday_date()
    is_weekend_flag = 0
    amount = round(random.uniform(500, 8000), 2)
    is_duplicate_of = ""

    if risk_type == "threshold":
        # structuring: amount placed just under a $10,000 reporting threshold
        amount = round(random.uniform(9500, 9999), 2)
    elif risk_type == "weekend":
        posting_date = random_weekend_date()
        is_weekend_flag = 1
    elif risk_type == "outlier":
        amount = round(random.uniform(25000, 60000), 2)

    due_date = posting_date + timedelta(days=random.choice([15, 30, 45, 60]))

    record = {
        "invoice_id": f"INV-{inv_id:04d}",
        "vendor": vendor,
        "amount": amount,
        "posting_date": posting_date.strftime("%Y-%m-%d"),
        "due_date": due_date.strftime("%Y-%m-%d"),
        "is_weekend_posting": is_weekend_flag,
        "risk_type": risk_type if risk_type else "none",
        "is_duplicate_of": is_duplicate_of,
        "label": 1 if risk_type else 0,
    }
    return record


def draw_invoice_pdf(record, filepath):
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 60, "INVOICE")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 90, f"Invoice Number: {record['invoice_id']}")
    c.drawString(50, height - 105, f"Vendor: {record['vendor']}")
    c.drawString(50, height - 120, f"Posting Date: {record['posting_date']}")
    c.drawString(50, height - 135, f"Due Date: {record['due_date']}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 170, f"Amount Due: ${record['amount']:,.2f}")

    c.setFont("Helvetica", 8)
    c.drawString(50, 60, "Thank you for your business.")

    c.save()


def main():
    records = []
    n_risky = int(N_INVOICES * 0.4)  # 40% risky, balanced-ish for measurability
    risk_types = (
        ["threshold"] * (n_risky // 3)
        + ["weekend"] * (n_risky // 3)
        + ["outlier"] * (n_risky - 2 * (n_risky // 3))
    )
    random.shuffle(risk_types)

    n_clean = N_INVOICES - len(risk_types) - 2  # reserve 2 slots for duplicates
    labels_plan = risk_types + [None] * n_clean

    for i, risk_type in enumerate(labels_plan, start=1):
        rec = make_invoice_record(i, risk_type)
        records.append(rec)

    # Inject 2 duplicate invoices (same vendor+amount+near-date as an existing clean one)
    clean_records = [r for r in records if r["label"] == 0]
    for j in range(2):
        base = random.choice(clean_records)
        dup_id = len(records) + 1
        dup = dict(base)
        dup["invoice_id"] = f"INV-{dup_id:04d}"
        dup["risk_type"] = "duplicate"
        dup["is_duplicate_of"] = base["invoice_id"]
        dup["label"] = 1
        records.append(dup)

    # Render PDFs
    for rec in records:
        path = os.path.join(OUT_DIR, f"{rec['invoice_id']}.pdf")
        draw_invoice_pdf(rec, path)

    # Write ground truth
    gt_path = os.path.join(os.path.dirname(__file__), "..", "data", "ground_truth.csv")
    with open(gt_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {len(records)} invoices in {OUT_DIR}")
    print(f"Ground truth written to {gt_path}")
    print(f"Risky: {sum(r['label'] for r in records)} / {len(records)}")


if __name__ == "__main__":
    main()
