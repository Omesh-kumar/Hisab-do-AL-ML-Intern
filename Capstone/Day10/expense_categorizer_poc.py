"""
HisabDo Capstone - Day 10
AI Feature POC: Smart Expense Categorization

Flow: Expense description (text) -> AI Model -> Predicted Category

This is a working proof-of-concept using a TF-IDF + Logistic Regression
text classifier. It is lightweight enough to run fully on-device (as a
TensorFlow Lite / ONNX model later), matching HisabDo's offline-first
architecture.
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Prepare input training data
# (Sample labeled expense descriptions - representative of real HisabDo
#  ledger entries. In production this would be pulled from anonymized,
#  aggregated user transaction history.)
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

texts = [t[0] for t in training_data]
labels = [t[1] for t in training_data]

# -----------------------------
# 2. Select AI model: TF-IDF vectorizer + Logistic Regression classifier
# (lightweight, fast, explainable, and portable to on-device formats)
# -----------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# -----------------------------
# 3. Implement / train the POC model
# -----------------------------
model.fit(X_train, y_train)

# Evaluate on held-out test data
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# -----------------------------
# 4. Send sample input to the AI system (simulating real app usage)
# -----------------------------
sample_inputs = [
    "bought vegetables and milk from store",
    "paid careem for airport ride",
    "monthly internet bill payment",
    "dinner with family at restaurant",
    "paid semester fee for university",
    "bought medicine from pharmacy",
    "new shoes bought from mall",
    "received payment from client for project",
]

# -----------------------------
# 5. Process the AI response
# -----------------------------
predictions = model.predict(sample_inputs)
probabilities = model.predict_proba(sample_inputs)
classes = model.classes_

results = []
for text, pred, probs in zip(sample_inputs, predictions, probabilities):
    confidence = float(max(probs))
    results.append({
        "input_expense_description": text,
        "predicted_category": pred,
        "confidence": round(confidence, 3)
    })

# -----------------------------
# 6. Document expected output
# -----------------------------
output_report = {
    "model": "TF-IDF + Logistic Regression (scikit-learn)",
    "test_accuracy_on_holdout_data": round(acc, 3),
    "sample_predictions": results
}

print(json.dumps(output_report, indent=2))

with open("sample_output.json", "w") as f:
    json.dump(output_report, f, indent=2)

print("\nSaved results to sample_output.json")
