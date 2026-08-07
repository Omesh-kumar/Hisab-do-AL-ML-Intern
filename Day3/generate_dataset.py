"""
HisabDo AI/ML Internship - Day 3 Task
Dataset Generator: Student Performance Data

Generates a CSV dataset of 35 students with some intentionally
missing/invalid values, so we can practice data cleaning.

Author: Omesh Kumar
"""

import random
import csv

random.seed(42)

first_names = [
    "Ali", "Sana", "Bilal", "Ayesha", "Zain", "Hira", "Usman", "Mahnoor",
    "Hamza", "Fatima", "Kashif", "Nida", "Farhan", "Iqra", "Saad", "Rabia",
    "Talha", "Zoya", "Danish", "Amna", "Adeel", "Sadia", "Waqas", "Mehak",
    "Faisal", "Noor", "Shahzad", "Anum", "Rizwan", "Laiba", "Omar", "Sara",
    "Junaid", "Hina", "Kamran"
]

last_names = [
    "Raza", "Khan", "Ahmed", "Tariq", "Malik", "Shah", "Ghani", "Iqbal",
    "Sheikh", "Noor", "Baig", "Farooq", "Butt", "Chaudhry", "Qureshi"
]

courses = ["AI/ML", "Data Science", "Web Development", "Cyber Security"]
genders = ["Male", "Female"]

rows = []
for i in range(1, 36):
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    age = random.randint(19, 25)
    gender = random.choice(genders)
    course = random.choice(courses)
    attendance = round(random.uniform(45, 100), 1)
    assignment = round(random.uniform(30, 100), 1)
    midterm = round(random.uniform(25, 100), 1)
    final = round(random.uniform(20, 100), 1)

    rows.append([i, name, age, gender, course, attendance,
                 assignment, midterm, final])

# ---- Intentionally introduce some messy/missing/invalid data ----
# (so Day 3 task can demonstrate data cleaning)
rows[3][5] = ""          # missing attendance
rows[7][6] = ""          # missing assignment score
rows[12][7] = -5         # invalid negative midterm score
rows[15][8] = ""         # missing final score
rows[20][2] = 150        # invalid/impossible age
rows[25][7] = 120        # invalid midterm score (>100)
rows[29][5] = ""         # missing attendance

header = ["StudentID", "Name", "Age", "Gender", "Course", "Attendance",
          "AssignmentScore", "MidtermScore", "FinalScore"]

with open("student_performance.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("student_performance.csv generated with", len(rows), "records.")
