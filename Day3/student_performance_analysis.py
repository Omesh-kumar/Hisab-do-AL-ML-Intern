"""
HisabDo AI/ML Internship - Day 3 Task
Student Performance Analysis using Python, Pandas & Matplotlib

Pipeline: Dataset -> Pandas -> Data Cleaning -> Analysis -> Visualization

Author: Omesh Kumar
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------------------
# STEP 1: Load the dataset
# -----------------------------------------
DATA_PATH = "student_performance.csv"
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("STEP 1: DATASET LOADED")
print("=" * 60)
print(df.head(10).to_string(index=False))


# -----------------------------------------
# STEP 2: Basic information about the dataset
# -----------------------------------------
print("\n" + "=" * 60)
print("STEP 2: BASIC DATASET INFORMATION")
print("=" * 60)
print(f"Shape (rows, columns): {df.shape}")
print("\nColumn data types:")
print(df.dtypes)
print("\nMissing values per column (before cleaning):")
print(df.isnull().sum())
print("\nStatistical summary (numeric columns):")
print(df.describe().to_string())


# -----------------------------------------
# STEP 9: Handle missing / invalid values
# (done early so all later steps use clean data)
# -----------------------------------------
print("\n" + "=" * 60)
print("STEP 9: DATA CLEANING")
print("=" * 60)

score_cols = ["AssignmentScore", "MidtermScore", "FinalScore"]

# 1) Fix invalid ages (impossible values, e.g. 150)
invalid_age_mask = (df["Age"] < 15) | (df["Age"] > 60)
print(f"Invalid age values found: {invalid_age_mask.sum()}")
df.loc[invalid_age_mask, "Age"] = df["Age"].median()

# 2) Fix invalid scores (must be between 0 and 100)
for col in score_cols:
    invalid_mask = (df[col] < 0) | (df[col] > 100)
    print(f"Invalid values in {col}: {invalid_mask.sum()}")
    df.loc[invalid_mask, col] = pd.NA

# 3) Convert score columns back to numeric (in case NA insertion changed dtype)
for col in score_cols + ["Attendance"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 4) Fill missing numeric values with column median (safe, robust to outliers)
for col in ["Attendance"] + score_cols:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled {missing_count} missing value(s) in '{col}' with median = {median_val:.2f}")

print("\nMissing values per column (after cleaning):")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("student_performance_cleaned.csv", index=False)
print("\nCleaned dataset saved as 'student_performance_cleaned.csv'")


# -----------------------------------------
# STEP 3: Find average scores
# -----------------------------------------
print("\n" + "=" * 60)
print("STEP 3: AVERAGE SCORES")
print("=" * 60)
avg_assignment = df["AssignmentScore"].mean()
avg_midterm = df["MidtermScore"].mean()
avg_final = df["FinalScore"].mean()
print(f"Average Assignment Score: {avg_assignment:.2f}")
print(f"Average Midterm Score:    {avg_midterm:.2f}")
print(f"Average Final Score:      {avg_final:.2f}")


# -----------------------------------------
# STEP 4: Highest and lowest scores (based on Final Score)
# -----------------------------------------
print("\n" + "=" * 60)
print("STEP 4: HIGHEST & LOWEST FINAL SCORES")
print("=" * 60)
top_student = df.loc[df["FinalScore"].idxmax()]
bottom_student = df.loc[df["FinalScore"].idxmin()]
print("Top scoring student:")
print(top_student.to_string())
print("\nLowest scoring student:")
print(bottom_student.to_string())


# -----------------------------------------
# STEP 5: Students with attendance below 75%
# -----------------------------------------
print("\n" + "=" * 60)
print("STEP 5: STUDENTS WITH ATTENDANCE BELOW 75%")
print("=" * 60)
low_attendance = df[df["Attendance"] < 75]
print(f"Count: {len(low_attendance)}")
print(low_attendance[["Name", "Course", "Attendance", "FinalScore"]].to_string(index=False))


# -----------------------------------------
# STEP 6: Students at risk of failing
# Rule: FinalScore < 40  OR  Attendance < 60
# -----------------------------------------
print("\n" + "=" * 60)
print("STEP 6: STUDENTS AT RISK OF FAILING")
print("=" * 60)
at_risk = df[(df["FinalScore"] < 40) | (df["Attendance"] < 60)]
print(f"Count: {len(at_risk)}")
print(at_risk[["Name", "Course", "Attendance", "FinalScore"]].to_string(index=False))


# -----------------------------------------
# STEP 7: Average score by course
# -----------------------------------------
print("\n" + "=" * 60)
print("STEP 7: AVERAGE FINAL SCORE BY COURSE")
print("=" * 60)
avg_by_course = df.groupby("Course")["FinalScore"].mean().sort_values(ascending=False)
print(avg_by_course.to_string())


# -----------------------------------------
# STEP 8: Relationship between attendance and final score
# -----------------------------------------
print("\n" + "=" * 60)
print("STEP 8: ATTENDANCE vs FINAL SCORE RELATIONSHIP")
print("=" * 60)
correlation = df["Attendance"].corr(df["FinalScore"])
print(f"Correlation coefficient: {correlation:.3f}")
if correlation > 0.5:
    strength = "a strong positive"
elif correlation > 0.2:
    strength = "a moderate positive"
elif correlation > -0.2:
    strength = "a weak/negligible"
else:
    strength = "a negative"
print(f"Interpretation: There is {strength} relationship between attendance and final score.")


# ==========================================
# VISUALIZATION (Matplotlib) - 3 charts
# ==========================================
plt.style.use("seaborn-v0_8-darkgrid")

# Chart 1: Score distribution (Final Score histogram)
plt.figure(figsize=(8, 5))
plt.hist(df["FinalScore"], bins=10, color="#1E3A8A", edgecolor="white")
plt.title("Distribution of Final Scores")
plt.xlabel("Final Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/1_score_distribution.png", dpi=150)
plt.close()

# Chart 2: Average score by course (bar chart)
plt.figure(figsize=(8, 5))
avg_by_course.plot(kind="bar", color="#14B8A6", edgecolor="black")
plt.title("Average Final Score by Course")
plt.xlabel("Course")
plt.ylabel("Average Final Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/2_average_score_by_course.png", dpi=150)
plt.close()

# Chart 3: Attendance vs Final Score (scatter plot)
plt.figure(figsize=(8, 5))
plt.scatter(df["Attendance"], df["FinalScore"], color="#1E3A8A", alpha=0.7, edgecolor="white")
plt.title(f"Attendance vs Final Score (correlation = {correlation:.2f})")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Score")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/3_attendance_vs_final_score.png", dpi=150)
plt.close()

print("\n" + "=" * 60)
print("CHARTS SAVED in the 'charts/' folder:")
print("=" * 60)
print("1_score_distribution.png")
print("2_average_score_by_course.png")
print("3_attendance_vs_final_score.png")

print("\nAnalysis complete.")
