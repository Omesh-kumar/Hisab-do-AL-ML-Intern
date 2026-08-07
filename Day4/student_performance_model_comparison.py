"""
HisabDo AI/ML Internship - Day 5 Task
Model Comparison: Logistic Regression vs Decision Tree
Student Performance Prediction (Pass / Fail)

Pipeline: Data -> Cleaning -> Train/Test Split -> Train Model 1 & Model 2
          -> Predictions -> Compare (Accuracy, Precision, Recall, F1, CM)

Author: Omesh Kumar
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# -----------------------------------------
# STEP 1: Prepare the dataset (load)
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

invalid_age_mask = (df["Age"] < 15) | (df["Age"] > 60)
df.loc[invalid_age_mask, "Age"] = df["Age"].median()

for col in score_cols:
    invalid_mask = (df[col] < 0) | (df[col] > 100)
    df.loc[invalid_mask, col] = pd.NA

for col in score_cols + ["Attendance"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in ["Attendance"] + score_cols:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled {missing_count} missing value(s) in '{col}' with median = {median_val:.2f}")

# Target column: Pass = 1 if FinalScore >= 50, else Fail = 0
PASS_THRESHOLD = 50
df["Result"] = (df["FinalScore"] >= PASS_THRESHOLD).astype(int)

# Features: FinalScore excluded on purpose (it defines the target -> would leak)
feature_cols = ["Attendance", "AssignmentScore", "MidtermScore"]

print(f"\nFeatures used: {feature_cols}")
print("(FinalScore excluded from features - it is used only to define Pass/Fail, "
      "including it would leak the answer into the model)")

print("\nClass balance (Pass=1, Fail=0):")
print(df["Result"].value_counts().rename({1: "Pass", 0: "Fail"}).to_string())
pass_pct = df["Result"].mean() * 100
print(f"Pass rate: {pass_pct:.1f}%  |  Fail rate: {100 - pass_pct:.1f}%")


# -----------------------------------------
# STEP 3: Split into training/testing sets
# -----------------------------------------
X = df[feature_cols]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("STEP 3: TRAIN/TEST SPLIT")
print("=" * 60)
print(f"Training samples: {len(X_train)}  |  Testing samples: {len(X_test)}")


# -----------------------------------------
# STEP 4: Train Model 1 - Logistic Regression
# -----------------------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
print("\n" + "=" * 60)
print("STEP 4: MODEL 1 TRAINED - Logistic Regression")
print("=" * 60)
print("Done.")


# -----------------------------------------
# STEP 5: Train Model 2 - Decision Tree
# -----------------------------------------
# max_depth=3 keeps the tree simple and interpretable, and helps
# reduce overfitting on this small dataset.
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)
print("\n" + "=" * 60)
print("STEP 5: MODEL 2 TRAINED - Decision Tree (max_depth=3)")
print("=" * 60)
print("Done.")


# -----------------------------------------
# STEP 6: Generate predictions
# -----------------------------------------
log_pred = log_model.predict(X_test)
tree_pred = tree_model.predict(X_test)

print("\n" + "=" * 60)
print("STEP 6: PREDICTIONS GENERATED")
print("=" * 60)
comparison_preds = X_test.copy()
comparison_preds["Actual"] = y_test.map({1: "Pass", 0: "Fail"})
comparison_preds["LogReg_Pred"] = pd.Series(log_pred, index=X_test.index).map({1: "Pass", 0: "Fail"})
comparison_preds["Tree_Pred"] = pd.Series(tree_pred, index=X_test.index).map({1: "Pass", 0: "Fail"})
print(comparison_preds.to_string(index=False))


# -----------------------------------------
# STEP 7: Compare performance (Accuracy, Precision, Recall, F1, CM)
# -----------------------------------------
def evaluate(name, y_true, y_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
    }

log_metrics = evaluate("Logistic Regression", y_test, log_pred)
tree_metrics = evaluate("Decision Tree", y_test, tree_pred)

results_table = pd.DataFrame([log_metrics, tree_metrics]).set_index("Model")
results_table = results_table.round(3)

print("\n" + "=" * 60)
print("STEP 7: MODEL COMPARISON TABLE")
print("=" * 60)
print(results_table.to_string())

log_cm = confusion_matrix(y_test, log_pred)
tree_cm = confusion_matrix(y_test, tree_pred)

print("\nConfusion Matrix - Logistic Regression:")
print(log_cm)
print("\nConfusion Matrix - Decision Tree:")
print(tree_cm)

print("\nClassification Report - Logistic Regression:")
print(classification_report(y_test, log_pred, target_names=["Fail", "Pass"], zero_division=0))
print("\nClassification Report - Decision Tree:")
print(classification_report(y_test, tree_pred, target_names=["Fail", "Pass"], zero_division=0))

# Save comparison table + reports to file
with open("evaluation_results.txt", "w") as f:
    f.write("MODEL COMPARISON TABLE\n")
    f.write("=" * 60 + "\n")
    f.write(results_table.to_string() + "\n\n")

    f.write("Confusion Matrix - Logistic Regression:\n")
    f.write(str(log_cm) + "\n\n")
    f.write("Confusion Matrix - Decision Tree:\n")
    f.write(str(tree_cm) + "\n\n")

    f.write("Classification Report - Logistic Regression:\n")
    f.write(classification_report(y_test, log_pred, target_names=["Fail", "Pass"], zero_division=0) + "\n\n")
    f.write("Classification Report - Decision Tree:\n")
    f.write(classification_report(y_test, tree_pred, target_names=["Fail", "Pass"], zero_division=0))

results_table.to_csv("model_comparison_table.csv")


# ==========================================
# VISUALIZATIONS
# ==========================================

# Chart 1: Confusion matrices side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(log_cm, display_labels=["Fail", "Pass"]).plot(ax=axes[0], cmap="Blues", values_format="d")
axes[0].set_title("Logistic Regression")
ConfusionMatrixDisplay(tree_cm, display_labels=["Fail", "Pass"]).plot(ax=axes[1], cmap="Greens", values_format="d")
axes[1].set_title("Decision Tree")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/1_confusion_matrices.png", dpi=150)
plt.close()

# Chart 2: Metric comparison bar chart
plt.figure(figsize=(8, 5))
results_table[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(kind="bar", figsize=(8, 5))
plt.title("Model Comparison: Logistic Regression vs Decision Tree")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/2_metric_comparison.png", dpi=150)
plt.close()

# Chart 3: Decision tree structure (helps explain WHY it made its decisions)
plt.figure(figsize=(12, 7))
plot_tree(tree_model, feature_names=feature_cols, class_names=["Fail", "Pass"],
          filled=True, rounded=True, fontsize=9)
plt.title("Decision Tree Structure")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/3_decision_tree_structure.png", dpi=150)
plt.close()

print("\n" + "=" * 60)
print("CHARTS SAVED in 'charts/':")
print("=" * 60)
print("1_confusion_matrices.png")
print("2_metric_comparison.png")
print("3_decision_tree_structure.png")

print("\nResults saved to 'evaluation_results.txt' and 'model_comparison_table.csv'")
print("\nDone.")
