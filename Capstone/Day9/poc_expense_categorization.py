"""
Day 9 POC - Smart Expense Categorization for HisabDo
Trains a simple text classifier to predict expense category from transaction description.
Run: python poc_expense_categorization.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Small labeled sample dataset (in production this would come from real HisabDo transactions)
training_data = [
    ("Careem ride to office", "Transport"),
    ("Uber to airport", "Transport"),
    ("Petrol station fill up", "Transport"),
    ("Bykea delivery fare", "Transport"),
    ("KFC lunch order", "Food"),
    ("Grocery shopping at Imtiaz", "Food"),
    ("Dinner at restaurant", "Food"),
    ("Foodpanda order", "Food"),
    ("K-Electric bill payment", "Bills"),
    ("SSGC gas bill", "Bills"),
    ("Internet bill PTCL", "Bills"),
    ("Mobile balance recharge", "Bills"),
    ("Netflix subscription", "Entertainment"),
    ("Movie tickets Cinepax", "Entertainment"),
    ("Spotify premium", "Entertainment"),
    ("Pharmacy medicine purchase", "Health"),
    ("Doctor consultation fee", "Health"),
    ("Gym membership fee", "Health"),
    ("Shirt purchase from mall", "Shopping"),
    ("Online order from Daraz", "Shopping"),
    ("Shoes purchase", "Shopping"),
    ("University fee payment", "Education"),
    ("Books purchase", "Education"),
    ("Course fee Coursera", "Education"),
]

texts = [t[0] for t in training_data]
labels = [t[1] for t in training_data]

# Build pipeline: TF-IDF -> Logistic Regression
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000)),
])
model.fit(texts, labels)


def categorize_transaction(description: str, confidence_threshold: float = 0.35):
    """Predict category for a new transaction description."""
    probs = model.predict_proba([description])[0]
    classes = model.classes_
    best_idx = probs.argmax()
    category = classes[best_idx]
    confidence = round(float(probs[best_idx]), 2)

    result = {
        "description": description,
        "predicted_category": category,
        "confidence": confidence,
        "needs_review": confidence < confidence_threshold,
    }
    return result


if __name__ == "__main__":
    test_transactions = [
        "Careem ride home",
        "Bill for electricity this month",
        "Bought new jacket from Outfitters",
        "Paid tuition fee for semester",
        "Chicken karahi from restaurant",
        "Random unclear text xyz",
    ]

    print("=== Smart Expense Categorization POC ===\n")
    for txn in test_transactions:
        result = categorize_transaction(txn)
        flag = "  ⚠ needs manual review" if result["needs_review"] else ""
        print(f"'{result['description']}' -> {result['predicted_category']} "
              f"(confidence: {result['confidence']}){flag}")
