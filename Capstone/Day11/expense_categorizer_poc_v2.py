"""
HisabDo Capstone - Day 11
AI Feature POC (Upgraded): Smart Expense Categorization
Now handles realistic, structured application-style input with validation.
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# -----------------------------
# 2. Input data structure definition
# -----------------------------
# Every transaction sent to the AI service from the app is expected in this shape:
#
# {
#   "transaction_id": str,       (required)
#   "description": str,          (required) - free text entered by user or from OCR
#   "amount": float,             (required) - must be > 0
#   "vendor": str,               (optional)
#   "date": "YYYY-MM-DD"         (required)
# }
#
# This mirrors the real fields captured by HisabDo when a user adds a transaction.

REQUIRED_FIELDS = ["transaction_id", "description", "amount", "date"]

# -----------------------------
# Training data (expanded from Day 10)
# -----------------------------
training_data = [
    ("milk bread eggs grocery store", "Grocery"),
    ("vegetables fruits sabzi wala", "Grocery"),
    ("monthly grocery shopping", "Grocery"),
    ("rice flour atta chawal", "Grocery"),
    ("chicken meat mutton shop", "Grocery"),
    ("uber ride to office", "Transport"),
    ("careem fare", "Transport"),
    ("petrol fuel bike", "Transport"),
    ("bus fare to university", "Transport"),
    ("rickshaw fare", "Transport"),
    ("electricity bill payment", "Utilities"),
    ("gas bill sui gas", "Utilities"),
    ("internet wifi bill", "Utilities"),
    ("mobile balance recharge", "Utilities"),
    ("water bill payment", "Utilities"),
    ("dinner at restaurant", "Food & Dining"),
    ("lunch with friends cafe", "Food & Dining"),
    ("biryani order online", "Food & Dining"),
    ("chai samosa nashta", "Food & Dining"),
    ("pizza delivery", "Food & Dining"),
    ("tuition fee payment", "Education"),
    ("books stationery purchase", "Education"),
    ("university semester fee", "Education"),
    ("online course subscription", "Education"),
    ("exam form fee", "Education"),
    ("doctor consultation fee", "Health"),
    ("medicine pharmacy purchase", "Health"),
    ("hospital bill", "Health"),
    ("dental checkup", "Health"),
    ("lab test blood report", "Health"),
    ("shirt jeans clothing shop", "Shopping"),
    ("shoes purchase", "Shopping"),
    ("mobile phone accessories", "Shopping"),
    ("online shopping order", "Shopping"),
    ("gift for friend birthday", "Shopping"),
    ("client payment received sales", "Income"),
    ("salary credited", "Income"),
    ("freelance project payment", "Income"),
    ("shop daily sales collection", "Income"),
    ("customer udhar payment received", "Income"),
]

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])
model.fit([t[0] for t in training_data], [t[1] for t in training_data])


# -----------------------------
# 5 & 6. Process AI response + handle invalid/incomplete input
# -----------------------------
def validate_transaction(txn: dict):
    """Returns (is_valid, error_message)."""
    for field in REQUIRED_FIELDS:
        if field not in txn or txn[field] in (None, ""):
            return False, f"Missing required field: '{field}'"
    if not isinstance(txn["description"], str) or len(txn["description"].strip()) == 0:
        return False, "Field 'description' must be a non-empty string"
    try:
        amount = float(txn["amount"])
        if amount <= 0:
            return False, "Field 'amount' must be greater than 0"
    except (ValueError, TypeError):
        return False, "Field 'amount' must be a valid number"
    return True, None


def categorize_transaction(txn: dict) -> dict:
    """
    Application -> Backend/API -> AI Service -> Response -> Application
    This function represents the AI Service step: it receives a structured
    transaction object, validates it, and returns a categorization result.
    """
    is_valid, error = validate_transaction(txn)
    if not is_valid:
        return {
            "transaction_id": txn.get("transaction_id", "unknown"),
            "status": "error",
            "error": error,
            "predicted_category": None,
            "confidence": None
        }

    description = txn["description"]
    prediction = model.predict([description])[0]
    confidence = float(max(model.predict_proba([description])[0]))

    return {
        "transaction_id": txn["transaction_id"],
        "status": "success",
        "error": None,
        "predicted_category": prediction,
        "confidence": round(confidence, 3)
    }


# -----------------------------
# 3 & 4. Realistic sample data (application-style input, including bad data)
# -----------------------------
sample_transactions = [
    {"transaction_id": "TXN1001", "description": "Grocery shopping at Imtiaz Super Market", "amount": 3450, "vendor": "Imtiaz Super Market", "date": "2026-08-10"},
    {"transaction_id": "TXN1002", "description": "Careem ride to office", "amount": 320, "vendor": "Careem", "date": "2026-08-10"},
    {"transaction_id": "TXN1003", "description": "K-Electric bill payment", "amount": 8500, "vendor": "K-Electric", "date": "2026-08-09"},
    {"transaction_id": "TXN1004", "description": "Dinner with family at Kolachi restaurant", "amount": 4200, "vendor": "Kolachi", "date": "2026-08-08"},
    {"transaction_id": "TXN1005", "description": "SMIU semester fee payment", "amount": 45000, "vendor": "SMIU", "date": "2026-08-05"},
    # --- Invalid / incomplete inputs, included intentionally to test error handling ---
    {"transaction_id": "TXN1006", "description": "", "amount": 500, "vendor": "Unknown", "date": "2026-08-10"},          # empty description
    {"transaction_id": "TXN1007", "description": "medicine from pharmacy", "amount": -200, "vendor": "Dawakhana", "date": "2026-08-10"},  # negative amount
    {"transaction_id": "TXN1008", "description": "grocery items", "amount": "abc", "vendor": "Store", "date": "2026-08-10"},  # invalid amount type
    {"transaction_id": "TXN1009", "amount": 600, "vendor": "Unknown", "date": "2026-08-10"},                              # missing description field entirely
]

results = [categorize_transaction(txn) for txn in sample_transactions]

output_report = {
    "model": "TF-IDF + Logistic Regression (scikit-learn)",
    "total_samples": len(sample_transactions),
    "successful": sum(1 for r in results if r["status"] == "success"),
    "failed_validation": sum(1 for r in results if r["status"] == "error"),
    "results": results
}

print(json.dumps(output_report, indent=2))

with open("day11_sample_output.json", "w") as f:
    json.dump(output_report, f, indent=2)

with open("day11_sample_input.json", "w") as f:
    json.dump(sample_transactions, f, indent=2)

print("\nSaved day11_sample_input.json and day11_sample_output.json")
