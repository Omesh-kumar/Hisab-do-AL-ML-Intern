# HisabDo AI/ML Internship — Day 3 Task
### Student Performance Analysis using Python, Pandas & Matplotlib

## 📌 Objective
Move from basic Python/Pandas practice into a proper data analysis project:
**Dataset → Pandas → Data Cleaning → Analysis → Visualization**

No ML model is used — this task focuses purely on understanding and analyzing data.

## 📂 Files in this folder
| File | Description |
|---|---|
| `generate_dataset.py` | Generates the raw dataset (35 students) with some intentional missing/invalid values |
| `student_performance.csv` | Raw dataset (before cleaning) |
| `student_performance_analysis.py` | Main analysis script — cleaning, analysis, and chart generation |
| `student_performance_cleaned.csv` | Cleaned dataset (output of the script) |
| `charts/1_score_distribution.png` | Histogram of final score distribution |
| `charts/2_average_score_by_course.png` | Bar chart — average final score by course |
| `charts/3_attendance_vs_final_score.png` | Scatter plot — attendance vs final score |

## 🧠 Dataset Fields
- Student Name, Age, Gender, Course
- Attendance (%)
- Assignment Score
- Midterm Score
- Final Score

35 student records were generated, with a few deliberately messy values
(missing attendance/scores, an impossible age of 150, and out-of-range scores
like -5 and 120) to practice realistic data cleaning.

## 🛠️ What the Script Does
1. **Load the dataset** from `student_performance.csv`
2. **Display basic info** — shape, data types, missing values, summary statistics
3. **Clean the data**:
   - Replaces impossible ages (e.g. 150) with the median age
   - Replaces out-of-range scores (negative or >100) with missing, then fills with the column median
   - Fills missing Attendance/Assignment/Midterm/Final values with the column median
   - Saves the result as `student_performance_cleaned.csv`
4. **Calculates average scores** (Assignment, Midterm, Final)
5. **Finds the highest and lowest scoring students** (by Final Score)
6. **Identifies students with attendance below 75%**
7. **Identifies students at risk of failing** (Final Score < 40 OR Attendance < 60)
8. **Calculates average final score by course**
9. **Checks the relationship between attendance and final score** using a correlation coefficient
10. **Generates 3 charts** using Matplotlib:
    - Score distribution (histogram)
    - Average score by course (bar chart)
    - Attendance vs Final Score (scatter plot)

## ▶️ How to Run
```bash
pip install pandas matplotlib
python generate_dataset.py          # (optional — regenerates the raw CSV)
python student_performance_analysis.py
```

## 📊 Short Conclusion / Findings
- The **average final score** across all 35 students was in the mid-60s, with scores
  spread fairly widely from below 30 to over 95 — showing a mix of strong and
  struggling students rather than a tight cluster.
- **Web Development** students had the highest average final score, while
  **Cyber Security** students had the lowest average in this sample.
- A meaningful number of students had **attendance below 75%**, and several of
  those also had low final scores — a pattern worth watching.
- Students flagged as **at risk of failing** (final score under 40, or attendance
  under 60%) made up close to two-thirds of the class in this dataset, which
  suggests the risk thresholds may need tuning on a real dataset, or that
  attendance alone isn't a strong single predictor of performance.
- The **correlation between attendance and final score was weak/negative**
  (~ -0.28) in this synthetic dataset. In other words, higher attendance did not
  reliably predict a higher final score here. This is somewhat expected since the
  data was randomly generated rather than sampled from real academic behavior —
  in a real dataset, we would expect attendance and performance to be more
  positively related.
- **Data cleaning mattered**: the raw dataset had missing attendance/score values
  and clearly invalid entries (age = 150, midterm score = -5 or 120). Handling
  these with median imputation and range checks (0–100 for scores, ~15–60 for
  age) was a necessary step before any analysis could be trusted.

## 🛠️ Tech Used
- Python 3
- Pandas
- Matplotlib

## ✅ Note
No ML model was used in this task, as instructed — the focus was on properly
understanding, cleaning, and analyzing data before moving on to modeling in a
later stage of the internship.
