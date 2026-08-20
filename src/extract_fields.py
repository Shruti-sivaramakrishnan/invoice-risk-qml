"""
extract_fields.py
Extracts structured fields (invoice number, vendor, amount, posting date,
due date) from invoice PDFs using pdfplumber text extraction + regex.
This simulates the "document intake" stage of the pipeline.
"""

import os
import re
import csv
import pdfplumber

IN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "invoices")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "extracted_fields.csv")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

PATTERNS = {
    "invoice_id": r"Invoice Number:\s*(INV-\d+)",
    "vendor": r"Vendor:\s*(.+)",
    "posting_date": r"Posting Date:\s*([\d-]+)",
    "due_date": r"Due Date:\s*([\d-]+)",
    "amount": r"Amount Due:\s*\$([\d,]+\.\d{2})",
}


def extract_from_pdf(filepath):
    with pdfplumber.open(filepath) as pdf:
        text = pdf.pages[0].extract_text()

    record = {}
    for field, pattern in PATTERNS.items():
        match = re.search(pattern, text)
        record[field] = match.group(1).strip() if match else None

    if record.get("amount"):
        record["amount"] = float(record["amount"].replace(",", ""))

    return record


def main():
    files = sorted(f for f in os.listdir(IN_DIR) if f.endswith(".pdf"))
    records = []
    failures = 0

    for fname in files:
        path = os.path.join(IN_DIR, fname)
        rec = extract_from_pdf(path)
        if any(v is None for v in rec.values()):
            failures += 1
            print(f"  WARNING: incomplete extraction for {fname}: {rec}")
        records.append(rec)

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PATTERNS.keys()))
        writer.writeheader()
        writer.writerows(records)

    total_fields = len(records) * len(PATTERNS)
    filled_fields = sum(1 for r in records for v in r.values() if v is not None)
    accuracy = filled_fields / total_fields * 100

    print(f"\nExtracted {len(records)} invoices -> {OUT_PATH}")
    print(f"Field-level extraction accuracy: {accuracy:.1f}% ({filled_fields}/{total_fields})")
    print(f"Invoices with at least one missing field: {failures}")


if __name__ == "__main__":
    main()
