"""
HisabDo AI/ML Internship - Day 4 Task
Student Performance Prediction: Pass / Fail Classification

Pipeline: Data -> Cleaning -> Feature Selection -> Train/Test Split
          -> Logistic Regression -> Predictions -> Evaluation

Author: Omesh Kumar
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# -----------------------------------------
# STEP 1: Load the dataset
# -----------------------------------------
df = pd.read_csv("student_performance.csv")

print("=" * 60)
print("STEP 1: DATASET LOADED")
print("=" * 60)
print(f"Shape: {df.shape}")
print(df.head().to_string(index=False))


# -----------------------------------------
# STEP 2: Clean the data
# -----------------------------------------
print("\n" + "=" * 60)
print("STEP 2: DATA CLEANING")
print("=" * 60)

score_cols = ["AssignmentScore", "MidtermScore", "FinalScore"]

# Fix impossible ages
invalid_age_mask = (df["Age"] < 15) | (df["Age"] > 60)
df.loc[invalid_age_mask, "Age"] = df["Age"].median()

# Fix out-of-range scores (must be 0-100) -> mark as missing first
for col in score_cols:
    invalid_mask = (df[col] < 0) | (df[col] > 100)
    df.loc[invalid_mask, col] = pd.NA

# Ensure numeric types
for col in score_cols + ["Attendance"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing numeric values with the column median
for col in ["Attendance"] + score_cols:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled {missing_count} missing value(s) in '{col}' with median = {median_val:.2f}")

print("Missing values after cleaning:")
print(df[["Attendance"] + score_cols].isnull().sum())


# -----------------------------------------
# STEP 3: Select useful features
#
# NOTE: FinalScore is deliberately NOT used as a feature. Since the target
# (Pass/Fail) is defined directly from FinalScore, including it as an input
# would leak the answer into the model (the model could just learn the
# threshold rule instead of actually learning from behaviour). Attendance
# and earlier assessments (Assignment, Midterm) are used instead, since in
# a real scenario these would be known BEFORE the final result.
# -----------------------------------------
feature_cols = ["Attendance", "AssignmentScore", "MidtermScore"]
print("\n" + "=" * 60)
print("STEP 3: FEATURES SELECTED")
print("=" * 60)
print(feature_cols)
print("(FinalScore excluded from features to avoid data leakage - it is used only to define the target)")


# -----------------------------------------
# STEP 4: Create the target column (Pass = 1, Fail = 0)
# Rule: a student PASSES if their FinalScore is 50 or above.
# -----------------------------------------
PASS_THRESHOLD = 50
df["Result"] = (df["FinalScore"] >= PASS_THRESHOLD).astype(int)

print("\n" + "=" * 60)
print("STEP 4: TARGET COLUMN CREATED (Pass = 1, Fail = 0)")
print("=" * 60)
print(f"Pass threshold: FinalScore >= {PASS_THRESHOLD}")
print(df["Result"].value_counts().rename({1: "Pass", 0: "Fail"}).to_string())


# -----------------------------------------
# STEP 5: Train/Test split
# -----------------------------------------
X = df[feature_cols]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("STEP 5: TRAIN/TEST SPLIT")
print("=" * 60)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# -----------------------------------------
# STEP 6: Train Logistic Regression model
# -----------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\n" + "=" * 60)
print("STEP 6: MODEL TRAINED (Logistic Regression)")
print("=" * 60)
print("Model coefficients (feature importance direction):")
for feature, coef in zip(feature_cols, model.coef_[0]):
    print(f"  {feature}: {coef:.4f}")


# -----------------------------------------
# STEP 7: Make predictions on test data
# -----------------------------------------
y_pred = model.predict(X_test)

print("\n" + "=" * 60)
print("STEP 7: PREDICTIONS ON TEST DATA")
print("=" * 60)
results_df = X_test.copy()
results_df["Actual"] = y_test.map({1: "Pass", 0: "Fail"})
results_df["Predicted"] = pd.Series(y_pred, index=X_test.index).map({1: "Pass", 0: "Fail"})
print(results_df.to_string(index=False))


# -----------------------------------------
# STEP 8: Evaluate the model
# -----------------------------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Fail", "Pass"])

print("\n" + "=" * 60)
print("STEP 8: MODEL EVALUATION")
print("=" * 60)
print(f"Accuracy: {accuracy:.2%}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# Save evaluation results to a text file for the repo
with open("evaluation_results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2%}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report)


# ==========================================
# VISUALIZATIONS (at least 2 required)
# ==========================================

# Chart 1: Confusion Matrix heatmap
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Pass/Fail Prediction")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/1_confusion_matrix.png", dpi=150)
plt.close()

# Chart 2: Feature importance (logistic regression coefficients)
plt.figure(figsize=(7, 5))
plt.barh(feature_cols, model.coef_[0], color="#1E3A8A")
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/2_feature_importance.png", dpi=150)
plt.close()

# Chart 3 (bonus): Actual vs Predicted result counts
plt.figure(figsize=(7, 5))
comparison = pd.DataFrame({
    "Actual": y_test.map({1: "Pass", 0: "Fail"}).value_counts(),
    "Predicted": pd.Series(y_pred).map({1: "Pass", 0: "Fail"}).value_counts(),
}).fillna(0)
comparison.plot(kind="bar", color=["#1E3A8A", "#14B8A6"])
plt.title("Actual vs Predicted Pass/Fail Counts (Test Set)")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/3_actual_vs_predicted.png", dpi=150)
plt.close()

print("\n" + "=" * 60)
print("CHARTS SAVED in the 'charts/' folder:")
print("=" * 60)
print("1_confusion_matrix.png")
print("2_feature_importance.png")
print("3_actual_vs_predicted.png")

print("\nEvaluation results saved to 'evaluation_results.txt'")
print("\nDone.")
