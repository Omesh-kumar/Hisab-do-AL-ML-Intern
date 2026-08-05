# HisabDo AI/ML Internship — Day 5 Task
### Model Comparison — Logistic Regression vs Decision Tree

## 📌 Objective
Continue the Day 4 Student Performance Prediction project by training a
**second classification algorithm** (Decision Tree) alongside the original
Logistic Regression model, and properly compare them using multiple
evaluation metrics — not just accuracy.

## 📂 Files in this folder
| File | Description |
|---|---|
| `student_performance.csv` | Dataset (same as Day 3/4 — 35 students, some messy values) |
| `student_performance_model_comparison.py` | Main script — cleaning, training both models, evaluation, comparison |
| `model_comparison_table.csv` | Comparison table (Accuracy, Precision, Recall, F1) |
| `evaluation_results.txt` | Full text output: comparison table, confusion matrices, classification reports |
| `charts/1_confusion_matrices.png` | Side-by-side confusion matrices for both models |
| `charts/2_metric_comparison.png` | Bar chart comparing Accuracy/Precision/Recall/F1 |
| `charts/3_decision_tree_structure.png` | Visualization of the trained Decision Tree's structure |

## 🧠 Features & Target (same setup as Day 4)
- **Features:** Attendance, Assignment Score, Midterm Score
- **FinalScore excluded from features** — it's used only to define the
  target, so including it would leak the answer into the model
- **Target:** `Result = 1 (Pass)` if `FinalScore >= 50`, else `0 (Fail)`

## 🤖 Models Trained
1. **Logistic Regression** (`max_iter=1000`)
2. **Decision Tree** (`max_depth=3`, to keep it interpretable and reduce
   overfitting on this small 35-row dataset)

## 📊 Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.556 | 0.625 | 0.833 | 0.714 |
| Decision Tree | 0.444 | 0.571 | 0.667 | 0.615 |

**Confusion Matrices (Test set, 9 students: 3 Fail, 6 Pass):**
```
Logistic Regression        Decision Tree
[[0 3]                     [[0 3]
 [1 5]]                     [2 4]]
```

## 🧠 Analysis

**Which model performed better?**
Logistic Regression performed better across every metric — Accuracy (55.6%
vs 44.4%), Precision (0.625 vs 0.571), Recall (0.833 vs 0.667), and F1 Score
(0.714 vs 0.615). Neither model is genuinely "good," but Logistic Regression
was consistently less wrong.

**Why do I think it performed better?**
With only 35 total records (26 for training), the Decision Tree — even
limited to `max_depth=3` — has enough flexibility to fit noise in the
training data rather than a real pattern, since decision trees split the
data into small, specific regions. Logistic Regression instead fits a single
smooth boundary across all the features at once, which tends to generalize
a bit more reliably on very small, noisy datasets. Neither model actually
learned strong signal here, though — both failed to correctly identify any
"Fail" case in the test set (0% recall for Fail in both), which strongly
suggests the underlying data doesn't contain a strong genuine pattern
linking Attendance/Assignment/Midterm scores to Pass/Fail — expected, since
this dataset was randomly generated rather than sampled from real student
behavior.

**Was the dataset balanced?**
Not perfectly, but not extremely imbalanced either: **65.7% Pass vs 34.3%
Fail** (23 Pass, 12 Fail out of 35). This mild imbalance is part of why both
models leaned toward predicting "Pass" — they had almost twice as many Pass
examples to learn from, and it shows in the confusion matrices: both models
predicted "Pass" for every single Fail case in the test set (3/3 misclassified
by both).

**What could improve the results?**
- **More data** — 35 rows (26 for training) is very small for a classifier
  to find a reliable pattern; a few hundred+ real records would help a lot.
- **A real dataset** — since this data was synthetically randomized, there's
  no genuine underlying relationship for the model to learn. Real student
  data would likely show a much clearer link between attendance/assessment
  scores and outcomes.
- **Class balancing techniques** — e.g. class weighting
  (`class_weight="balanced"`) or oversampling the minority (Fail) class,
  since both models currently ignore the Fail class almost entirely.
- **More/better features** — e.g. participation, assignment submission
  timeliness, or previous semester GPA, which could carry more signal than
  what's available here.
- **Cross-validation** instead of a single train/test split, since with only
  9 test samples, the metrics above can shift a lot depending on which
  students happened to land in the test set.

## ▶️ How to Run
```bash
pip install pandas matplotlib scikit-learn
python student_performance_model_comparison.py
```

## 🛠️ Tech Used
- Python 3
- Pandas
- Matplotlib
- Scikit-learn (LogisticRegression, DecisionTreeClassifier, evaluation metrics)

## ✅ Note
Model selection was based on **all four metrics together**, not just
accuracy — the Decision Tree was not simply set aside because it scored
lower; both models were examined for precision/recall tradeoffs and the
class imbalance issue was explicitly called out as a likely cause of both
models' poor performance on the "Fail" class.
