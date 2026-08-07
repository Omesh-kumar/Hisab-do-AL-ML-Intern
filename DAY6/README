# HisabDo AI/ML Internship — Day 6 Task
### Feature Engineering + Hyperparameter Tuning (GridSearchCV)

## 📌 Objective
Continue the Student Performance Prediction project by improving the model
through **feature engineering** and **hyperparameter tuning**, then compare
the tuned model against the Day 5 baseline (Logistic Regression) using
multiple evaluation metrics.

## 📂 Files in this folder
| File | Description |
|---|---|
| `student_performance.csv` | Dataset (same as Day 3/4/5 — 35 students, some messy values) |
| `student_performance_hyperparameter_tuning.py` | Main script — cleaning, feature engineering, scaling, GridSearchCV tuning, evaluation |
| `student_performance_hyperparameter_tuning.ipynb` | Same pipeline as an executed Jupyter notebook |
| `model_comparison_table.csv` | Baseline vs Tuned comparison table (Accuracy, Precision, Recall, F1, ROC-AUC) |
| `best_hyperparameters.csv` | Best hyperparameter combination found by GridSearchCV |
| `evaluation_results.txt` | Full text output: comparison table, confusion matrices, classification reports |
| `charts/1_confusion_matrices.png` | Side-by-side confusion matrices for both models |
| `charts/2_metric_comparison.png` | Bar chart comparing Accuracy/Precision/Recall/F1/ROC-AUC |
| `charts/3_roc_curves.png` | ROC curves for both models with AUC scores |
| `charts/4_feature_importance.png` | Feature importance from the tuned Random Forest |

## 🧠 Pipeline

**1. Data Cleaning** — Same rules as Day 3–5: invalid ages/scores set to
missing, then filled with the column median.

**2. Feature Engineering** — Six new features built only from
Attendance/AssignmentScore/MidtermScore (FinalScore is *never* used as an
input — it defines the target, so including it would leak the answer):
- `AvgAssessmentScore` — mean of Assignment & Midterm scores
- `Assignment_Midterm_Gap` — difference between the two, flags inconsistency
- `Attendance_x_Midterm` — interaction term (attendance scaled × midterm)
- `LowAttendanceFlag` — binary flag, 1 if Attendance < 75
- `Gender_Encoded`, `Course_Encoded` — categorical encodings

**3. Feature Selection** — Kept features with `|correlation| >= 0.05`
against the target (Result), based on a correlation matrix against all
9 candidate features. This drops near-zero-signal features like
`Attendance_x_Midterm` and `AssignmentScore` on their own.

**4. Feature Scaling** — `StandardScaler` (fit on train, applied to test)
so all features are on comparable scales — required for Logistic
Regression and generally good practice before tuning tree ensembles too.

**5. Train/Test Split** — 75/25 stratified split (same split ratio as
Day 5, `random_state=42`), so the comparison to the baseline is apples-to-apples.

**6. Hyperparameter Tuning — GridSearchCV** — Tuned a `RandomForestClassifier`
over 48 combinations of `n_estimators`, `max_depth`, `min_samples_split`,
`min_samples_leaf`, using 3-fold stratified cross-validation and optimizing
for F1 score (chosen over accuracy because the dataset has class imbalance —
see Day 5 analysis).

## 📊 Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (Day 5 Logistic Regression) | 0.667 | 0.667 | 1.000 | 0.800 | 0.278 |
| Tuned Random Forest (GridSearchCV) | 0.556 | 0.625 | 0.833 | 0.714 | 0.556 |

**Best hyperparameters found:** `max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200`
**Best CV F1 score (train, 3-fold):** 0.722

## 🧠 Analysis

**Did tuning improve the model?**
Not on this test set — the baseline Logistic Regression actually scored
higher on Accuracy, Precision, Recall, and F1. However, the tuned Random
Forest had a much better **ROC-AUC (0.556 vs 0.278)**, meaning it ranks
Pass/Fail cases more sensibly across all thresholds — the baseline's
ROC-AUC below 0.5 is a red flag that it's barely better than random at
separating the classes, it just happens to benefit here from predicting
"Pass" almost every time on an imbalanced, tiny test set (6 Pass vs 3 Fail).

**Why didn't tuning help more?**
With only 35 total rows (26 train / 9 test), there isn't enough data for
GridSearchCV's cross-validation folds to reliably estimate which
hyperparameters generalize, and a 9-row test set means each single
prediction shifts every metric by ~11%. The added engineered features
(course, gender, attendance interactions) also had weak correlation with
the target (max |r| ≈ 0.29), consistent with Day 5's finding that this
synthetic dataset doesn't contain a strong underlying signal.

**What would make this comparison more meaningful?**
- A real, larger dataset (hundreds of rows) so CV folds and the test set
  are statistically stable.
- `class_weight="balanced"` in the Random Forest / Logistic Regression to
  address the Pass/Fail imbalance more directly.
- Repeated k-fold CV reported with mean ± std, instead of a single
  train/test split, to see how much the metrics genuinely vary.

## ▶️ How to Run
```bash
pip install pandas matplotlib scikit-learn
python student_performance_hyperparameter_tuning.py
# or open/run student_performance_hyperparameter_tuning.ipynb
```

## 🛠️ Tech Used
- Python 3, Pandas, Matplotlib
- Scikit-learn (`StandardScaler`, `GridSearchCV`, `StratifiedKFold`,
  `RandomForestClassifier`, `LogisticRegression`, evaluation metrics)

## ✅ Note
FinalScore was excluded from every feature set (original + engineered) to
avoid target leakage, since it directly defines the Pass/Fail label. Model
selection was based on all five metrics together, not just accuracy — the
writeup above explicitly flags that the baseline's higher accuracy is
partly an artifact of the small, imbalanced test set, not proof it's the
stronger model.
