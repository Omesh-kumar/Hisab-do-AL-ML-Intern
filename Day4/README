# HisabDo AI/ML Internship — Day 4 Task
### Student Performance Prediction — Pass/Fail Classification (Logistic Regression)

## 📌 What Problem Was Solved
The goal was to build a first Machine Learning model that predicts whether a
student is likely to **Pass or Fail**, based on their attendance and earlier
assessment scores (before the final result is known). This is a **binary
classification** problem — the output is one of two classes: Pass (1) or Fail (0).

## 📂 Files in this folder
| File | Description |
|---|---|
| `student_performance.csv` | Dataset (reused from Day 3 — 35 students, some messy values) |
| `student_performance_prediction.py` | Main script — cleaning, feature selection, training, evaluation |
| `evaluation_results.txt` | Saved accuracy, confusion matrix, and classification report |
| `charts/1_confusion_matrix.png` | Confusion matrix heatmap |
| `charts/2_feature_importance.png` | Logistic Regression coefficients (feature importance) |
| `charts/3_actual_vs_predicted.png` | Actual vs predicted Pass/Fail counts on the test set |

## 🧠 Features Used
- **Attendance (%)**
- **Assignment Score**
- **Midterm Score**

**Important note on FinalScore:** the target column (Pass/Fail) is defined
directly from `FinalScore` (Pass if `FinalScore >= 50`). Because of that,
`FinalScore` was **deliberately excluded** from the model's input features.
Including it would have caused **data leakage** — the model would just learn
the pass/fail threshold rule instead of actually learning a pattern from
attendance and earlier performance. This was confirmed while building the
model: including `FinalScore` as a feature gave a suspicious 100% accuracy,
which is a classic sign of leakage, not a genuinely good model.

## 🎯 Target Column
```
Result = 1 (Pass)  if FinalScore >= 50
Result = 0 (Fail)  otherwise
```

## 🛠️ Steps Performed
1. **Load the dataset** with Pandas
2. **Clean the data** — fixed impossible ages, out-of-range scores (negative
   or above 100), and filled missing values using the column median
3. **Select features** — Attendance, Assignment Score, Midterm Score
4. **Create the target column** (Pass = 1, Fail = 0) from Final Score
5. **Split the data** — 75% training, 25% testing (`train_test_split`,
   stratified so both classes are represented in the test set)
6. **Train a Logistic Regression model** using Scikit-learn
7. **Make predictions** on the unseen test data
8. **Evaluate the model** using Accuracy, Confusion Matrix, and Classification
   Report

## 🤖 Model Selected
**Logistic Regression** — chosen because it's the standard first model for
binary classification problems. It's simple, interpretable (each feature
gets a coefficient showing its direction of influence), and a good baseline
before trying more complex models later in the internship.

## 📊 Accuracy Achieved
- **Accuracy: 55.56%** on the test set (9 students)
- Confusion Matrix:
  ```
  [[0 3]     Fail: 0 correct, 3 misclassified as Pass
   [1 5]]    Pass: 1 misclassified as Fail, 5 correct
  ```
- The model struggled particularly to correctly identify **Fail** cases
  (0% recall for Fail) — full details are in `evaluation_results.txt`.

## 💡 What I Learned
- **Data leakage is a real trap.** My first version of the model used
  `FinalScore` as a feature and scored 100% accuracy — that immediately
  looked suspicious rather than impressive, because the target was literally
  derived from that same column. Removing it gave a much more honest (and
  much lower) accuracy, which is a more realistic signal of how well
  Attendance/Assignment/Midterm scores alone can predict the outcome.
- **A small, random dataset limits real-world accuracy.** Since this Day 3
  dataset was randomly generated (not real student behavior), there isn't a
  strong genuine relationship between attendance/assessment scores and the
  final result — so a model trained on it can't perform much better than
  guessing. On a real dataset with genuine patterns, I'd expect noticeably
  better performance.
- **Train/test split and stratification** matter even on small datasets, to
  make sure both classes appear in the test set.
- **Logistic Regression coefficients** are a simple way to peek inside the
  model — positive coefficients push toward "Pass," negative ones push
  toward "Fail."
- Evaluating with more than just accuracy (confusion matrix + classification
  report) gave a much clearer picture of *where* the model was going wrong,
  rather than just one overall number.

## ▶️ How to Run
```bash
pip install pandas matplotlib seaborn scikit-learn
python student_performance_prediction.py
```

## 🛠️ Tech Used
- Python 3
- Pandas
- Matplotlib / Seaborn
- Scikit-learn (Logistic Regression, train_test_split, evaluation metrics)

## ✅ Note
This project was built and understood step by step (not copied from an
existing complete project), including deliberately investigating and fixing
the data leakage issue described above.
