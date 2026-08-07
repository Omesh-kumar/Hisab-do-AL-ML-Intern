"""
Day 7 - HisabDo AI/ML Internship
Train and save the Student Performance Prediction model (RandomForestClassifier,
tuned with GridSearchCV) so it can be loaded by the FastAPI prediction service.

This reuses the same modeling approach from Day 4-6 (Logistic Regression ->
Decision Tree -> tuned RandomForestClassifier with feature engineering).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. Load / generate the student performance dataset
#    Replace this block with pd.read_csv("student_performance.csv") if you
#    already have the dataset used in Day 4-6.
# ---------------------------------------------------------------------------
N = 1200

attendance = np.clip(np.random.normal(78, 12, N), 30, 100)
assignment_score = np.clip(np.random.normal(70, 15, N), 0, 100)
midterm_score = np.clip(np.random.normal(65, 18, N), 0, 100)
final_score = np.clip(np.random.normal(60, 20, N), 0, 100)

# Weighted composite drives the pass/fail label (mirrors real grading logic)
composite = (
    0.20 * attendance
    + 0.20 * assignment_score
    + 0.25 * midterm_score
    + 0.35 * final_score
)
noise = np.random.normal(0, 5, N)
label = np.where(composite + noise >= 60, "Pass", "Fail")

df = pd.DataFrame({
    "Attendance": attendance.round(2),
    "AssignmentScore": assignment_score.round(2),
    "MidtermScore": midterm_score.round(2),
    "FinalScore": final_score.round(2),
    "Result": label,
})

df.to_csv(os.path.join(OUTPUT_DIR, "student_performance.csv"), index=False)
print(f"Dataset shape: {df.shape}")
print(df["Result"].value_counts())

# ---------------------------------------------------------------------------
# 2. Feature engineering (same features the API will require)
# ---------------------------------------------------------------------------
FEATURES = ["Attendance", "AssignmentScore", "MidtermScore", "FinalScore"]
X = df[FEATURES]
y = df["Result"].map({"Fail": 0, "Pass": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. GridSearchCV tuning (same approach used in Day 6)
# ---------------------------------------------------------------------------
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

rf = RandomForestClassifier(random_state=RANDOM_STATE)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print("Best params:", grid_search.best_params_)

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
y_pred = best_model.predict(X_test_scaled)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

# ---------------------------------------------------------------------------
# 5. Save model + scaler + feature order for the API
# ---------------------------------------------------------------------------
joblib.dump(best_model, os.path.join(OUTPUT_DIR, "student_performance_model.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(FEATURES, os.path.join(OUTPUT_DIR, "feature_order.pkl"))

print("\nSaved: student_performance_model.pkl, scaler.pkl, feature_order.pkl")
