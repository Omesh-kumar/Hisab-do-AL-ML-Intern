"""
HisabDo AI/ML Internship - Day 6 Task
Feature Engineering + Hyperparameter Tuning (GridSearchCV)
Student Performance Prediction (Pass / Fail)

Pipeline: Data -> Cleaning -> Feature Engineering -> Feature Selection
          -> Feature Scaling -> Train/Test Split -> GridSearchCV Tuning
          -> Evaluate Tuned Model -> Compare vs Day 5 Baseline (Logistic Regression)

Author: Omesh Kumar
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)
RANDOM_STATE = 42

# ==========================================================
# STEP 1: Load dataset
# ==========================================================
df = pd.read_csv("student_performance.csv", sep=None, engine="python")

print("=" * 60)
print("STEP 1: DATASET LOADED")
print("=" * 60)
print(f"Shape: {df.shape}")
print(df.head().to_string(index=False))


# ==========================================================
# STEP 2: Data Cleaning (same rules as Day 3-5, for consistency)
# ==========================================================
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

# Target: Pass = 1 if FinalScore >= 50, else Fail = 0
PASS_THRESHOLD = 50
df["Result"] = (df["FinalScore"] >= PASS_THRESHOLD).astype(int)


# ==========================================================
# STEP 3: Feature Engineering
# ==========================================================
print("\n" + "=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

# New engineered features (built only from Attendance/Assignment/Midterm -
# FinalScore is still excluded everywhere to avoid target leakage)
df["AvgAssessmentScore"] = (df["AssignmentScore"] + df["MidtermScore"]) / 2
df["Assignment_Midterm_Gap"] = df["AssignmentScore"] - df["MidtermScore"]
df["Attendance_x_Midterm"] = (df["Attendance"] / 100) * df["MidtermScore"]
df["LowAttendanceFlag"] = (df["Attendance"] < 75).astype(int)
df["Gender_Encoded"] = df["Gender"].map({"Male": 0, "Female": 1})
df["Course_Encoded"] = df["Course"].astype("category").cat.codes

print("New features created:")
print(" - AvgAssessmentScore     = mean(AssignmentScore, MidtermScore)")
print(" - Assignment_Midterm_Gap = AssignmentScore - MidtermScore")
print(" - Attendance_x_Midterm   = (Attendance/100) * MidtermScore  (interaction term)")
print(" - LowAttendanceFlag      = 1 if Attendance < 75 else 0")
print(" - Gender_Encoded         = Male=0, Female=1")
print(" - Course_Encoded         = category codes for Course")


# ==========================================================
# STEP 4: Feature Selection
# ==========================================================
print("\n" + "=" * 60)
print("STEP 4: FEATURE SELECTION")
print("=" * 60)

candidate_features = [
    "Attendance", "AssignmentScore", "MidtermScore",
    "AvgAssessmentScore", "Assignment_Midterm_Gap",
    "Attendance_x_Midterm", "LowAttendanceFlag",
    "Gender_Encoded", "Course_Encoded",
]

corr_with_target = df[candidate_features + ["Result"]].corr()["Result"].drop("Result").sort_values(key=abs, ascending=False)
print("Correlation of each candidate feature with Result (Pass/Fail):")
print(corr_with_target.round(3).to_string())

# Keep features with the strongest absolute correlation with the target,
# dropping ones that add near-zero signal (helps a small, noisy dataset)
SELECTION_THRESHOLD = 0.05
feature_cols = corr_with_target[corr_with_target.abs() >= SELECTION_THRESHOLD].index.tolist()
if len(feature_cols) < 3:  # safety floor - keep the original 3 core features at least
    feature_cols = list(dict.fromkeys(feature_cols + ["Attendance", "AssignmentScore", "MidtermScore"]))

print(f"\nSelected features (|corr| >= {SELECTION_THRESHOLD}): {feature_cols}")
print("(FinalScore is never included - it defines the target and would leak the answer)")


# ==========================================================
# STEP 5: Feature Scaling
# ==========================================================
X = df[feature_cols].copy()
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

print("\n" + "=" * 60)
print("STEP 5: FEATURE SCALING (StandardScaler)")
print("=" * 60)
print("All features standardized to mean=0, std=1 (fit on train, applied to test).")


# ==========================================================
# STEP 6: Train/Test Split summary
# ==========================================================
print("\n" + "=" * 60)
print("STEP 6: TRAIN/TEST SPLIT")
print("=" * 60)
print(f"Training samples: {len(X_train)}  |  Testing samples: {len(X_test)}")
print(f"Train class balance:\n{y_train.value_counts().rename({1:'Pass',0:'Fail'}).to_string()}")


# ==========================================================
# STEP 7: Baseline model (Day 5 result, re-declared for fair comparison)
# ==========================================================
BASELINE_FEATURES = ["Attendance", "AssignmentScore", "MidtermScore"]
baseline_scaler = StandardScaler()
X_train_baseline = pd.DataFrame(
    baseline_scaler.fit_transform(X_train_full := df.loc[X_train.index, BASELINE_FEATURES]),
    columns=BASELINE_FEATURES, index=X_train.index,
)
X_test_baseline = pd.DataFrame(
    baseline_scaler.transform(df.loc[X_test.index, BASELINE_FEATURES]),
    columns=BASELINE_FEATURES, index=X_test.index,
)

baseline_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
baseline_model.fit(X_train_baseline, y_train)


# ==========================================================
# STEP 8: Hyperparameter Tuning with GridSearchCV
# ==========================================================
print("\n" + "=" * 60)
print("STEP 8: HYPERPARAMETER TUNING (GridSearchCV)")
print("=" * 60)

cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 4, None],
    "min_samples_split": [2, 4],
    "min_samples_leaf": [1, 2],
}

rf_base = RandomForestClassifier(random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1,
)
grid_search.fit(X_train_scaled, y_train)

print(f"Model tuned: RandomForestClassifier")
print(f"Param grid size: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])} combinations")
print(f"CV strategy: StratifiedKFold(n_splits=3)")
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV F1 score: {grid_search.best_score_:.3f}")

tuned_model = grid_search.best_estimator_


# ==========================================================
# STEP 9: Predictions
# ==========================================================
baseline_pred = baseline_model.predict(X_test_baseline)
baseline_proba = baseline_model.predict_proba(X_test_baseline)[:, 1]

tuned_pred = tuned_model.predict(X_test_scaled)
tuned_proba = tuned_model.predict_proba(X_test_scaled)[:, 1]


# ==========================================================
# STEP 10: Evaluate & Compare
# ==========================================================
def evaluate(name, y_true, y_pred, y_proba):
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba)
    except ValueError:
        metrics["ROC-AUC"] = np.nan
    return metrics

baseline_metrics = evaluate("Baseline (Day5 Logistic Regression)", y_test, baseline_pred, baseline_proba)
tuned_metrics = evaluate("Tuned Random Forest (GridSearchCV)", y_test, tuned_pred, tuned_proba)

results_table = pd.DataFrame([baseline_metrics, tuned_metrics]).set_index("Model").round(3)

print("\n" + "=" * 60)
print("STEP 10: MODEL COMPARISON — BASELINE vs TUNED")
print("=" * 60)
print(results_table.to_string())

baseline_cm = confusion_matrix(y_test, baseline_pred)
tuned_cm = confusion_matrix(y_test, tuned_pred)

print("\nConfusion Matrix - Baseline (Logistic Regression):")
print(baseline_cm)
print("\nConfusion Matrix - Tuned (Random Forest):")
print(tuned_cm)

print("\nClassification Report - Baseline:")
print(classification_report(y_test, baseline_pred, target_names=["Fail", "Pass"], zero_division=0))
print("\nClassification Report - Tuned:")
print(classification_report(y_test, tuned_pred, target_names=["Fail", "Pass"], zero_division=0))

with open("evaluation_results.txt", "w") as f:
    f.write("DAY 6 - MODEL COMPARISON: BASELINE (Day 5 Logistic Regression) vs TUNED (GridSearchCV Random Forest)\n")
    f.write("=" * 70 + "\n")
    f.write(results_table.to_string() + "\n\n")
    f.write(f"Best hyperparameters found: {grid_search.best_params_}\n")
    f.write(f"Best CV F1 score: {grid_search.best_score_:.3f}\n\n")
    f.write("Confusion Matrix - Baseline:\n" + str(baseline_cm) + "\n\n")
    f.write("Confusion Matrix - Tuned:\n" + str(tuned_cm) + "\n\n")
    f.write("Classification Report - Baseline:\n")
    f.write(classification_report(y_test, baseline_pred, target_names=["Fail", "Pass"], zero_division=0) + "\n\n")
    f.write("Classification Report - Tuned:\n")
    f.write(classification_report(y_test, tuned_pred, target_names=["Fail", "Pass"], zero_division=0))

results_table.to_csv("model_comparison_table.csv")
pd.DataFrame([grid_search.best_params_]).to_csv("best_hyperparameters.csv", index=False)


# ==========================================================
# VISUALIZATIONS
# ==========================================================

# Chart 1: Confusion matrices side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(baseline_cm, display_labels=["Fail", "Pass"]).plot(ax=axes[0], cmap="Blues", values_format="d")
axes[0].set_title("Baseline: Logistic Regression")
ConfusionMatrixDisplay(tuned_cm, display_labels=["Fail", "Pass"]).plot(ax=axes[1], cmap="Oranges", values_format="d")
axes[1].set_title("Tuned: Random Forest (GridSearchCV)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/1_confusion_matrices.png", dpi=150)
plt.close()

# Chart 2: Metric comparison bar chart
plt.figure(figsize=(9, 5))
results_table.plot(kind="bar", figsize=(9, 5))
plt.title("Day 6: Baseline vs Tuned Model — Metric Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/2_metric_comparison.png", dpi=150)
plt.close()

# Chart 3: ROC curves
plt.figure(figsize=(7, 6))
for name, y_proba, color in [
    ("Baseline (Logistic Regression)", baseline_proba, "tab:blue"),
    ("Tuned (Random Forest)", tuned_proba, "tab:orange"),
]:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=color)
plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Baseline vs Tuned Model")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/3_roc_curves.png", dpi=150)
plt.close()

# Chart 4: Feature importance (tuned Random Forest)
importances = pd.Series(tuned_model.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="teal")
plt.title("Feature Importance — Tuned Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/4_feature_importance.png", dpi=150)
plt.close()

print("\n" + "=" * 60)
print("CHARTS SAVED in 'charts/':")
print("=" * 60)
print("1_confusion_matrices.png")
print("2_metric_comparison.png")
print("3_roc_curves.png")
print("4_feature_importance.png")

print("\nResults saved to 'evaluation_results.txt', 'model_comparison_table.csv', 'best_hyperparameters.csv'")
print("\nDone.")
