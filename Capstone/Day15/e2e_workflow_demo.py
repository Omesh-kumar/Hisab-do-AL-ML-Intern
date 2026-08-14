"""
HisabDo Capstone - Day 14
AI/ML Task: End-to-end workflow demo for the AI feature (Smart Expense Categorization)

This simulates the full real-world flow:
  User enters a transaction -> App calls AI service -> AI service validates
  and processes -> Response returned -> App applies result (or falls back
  to manual entry on error).

Reuses the FastAPI service built on Day 12 (categorization_service.py).
"""

import sys


from fastapi.testclient import TestClient
from categorization_service import app
import json

client = TestClient(app)


def simulate_app_add_transaction(txn: dict):
    """
    This function represents what the HisabDo app does when a user adds
    a transaction: call the AI service, then decide what to show the user.
    """
    response = client.post("/categorize", json=txn)

    if response.status_code != 200:
        # Validation failed at the schema level (missing/invalid field)
        return {
            "transaction_id": txn.get("transaction_id", "unknown"),
            "app_behavior": "Show manual category dropdown (input rejected)",
            "detail": response.json()["detail"][0]["msg"]
        }

    body = response.json()
    if body["status"] == "success":
        return {
            "transaction_id": body["transaction_id"],
            "app_behavior": f"Auto-fill category: '{body['predicted_category']}' (confidence {body['confidence']})",
            "detail": None
        }
    else:
        return {
            "transaction_id": body["transaction_id"],
            "app_behavior": "Show manual category dropdown (AI unsure)",
            "detail": body["error"]
        }


# -----------------------------
# End-to-end sample workflow: 5 realistic transactions a user might add in one session
# -----------------------------
workflow_transactions = [
    {"transaction_id": "TXN4001", "description": "grocery shopping weekly", "amount": 2800, "date": "2026-08-14"},
    {"transaction_id": "TXN4002", "description": "petrol for bike", "amount": 1500, "date": "2026-08-14"},
    {"transaction_id": "TXN4003", "description": "", "amount": 300, "date": "2026-08-14"},           # invalid input
    {"transaction_id": "TXN4004", "description": "asdkj random text", "amount": 200, "date": "2026-08-14"},  # low confidence
    {"transaction_id": "TXN4005", "description": "salary credited this month", "amount": 90000, "date": "2026-08-14"},
]

print("=== End-to-End Workflow Simulation ===\n")
results = []
for txn in workflow_transactions:
    result = simulate_app_add_transaction(txn)
    results.append(result)
    print(f"{result['transaction_id']}: {result['app_behavior']}" + (f" — {result['detail']}" if result['detail'] else ""))

with open("day14_workflow_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved day14_workflow_results.json")
