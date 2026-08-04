"""
HisabDo AI/ML Internship - Day 2 Task
Student Data Analysis using Python & Pandas

Objective:
- Store basic student information (Name, Age, Course, Marks)
- Convert data into a Pandas DataFrame
- Perform basic filtering and calculations

Author: Omesh Kumar
"""

import pandas as pd

# -----------------------------------------
# STEP 1: Create the student dataset
# -----------------------------------------
# Using a dictionary of lists (easy to convert into a DataFrame)
students_data = {
    "Name": [
        "Ali Raza", "Sana Khan", "Bilal Ahmed", "Ayesha Tariq", "Zain Malik",
        "Hira Shah", "Usman Ghani", "Mahnoor Iqbal", "Hamza Sheikh", "Fatima Noor"
    ],
    "Age": [20, 21, 22, 20, 23, 21, 22, 20, 24, 21],
    "Course": [
        "AI/ML", "Data Science", "AI/ML", "Web Development", "AI/ML",
        "Data Science", "Cyber Security", "AI/ML", "Web Development", "Data Science"
    ],
    "Marks": [85, 72, 65, 90, 58, 77, 69, 95, 40, 81]
}

# -----------------------------------------
# STEP 2: Convert dictionary to Pandas DataFrame
# -----------------------------------------
df = pd.DataFrame(students_data)


def display_all_students(dataframe: pd.DataFrame) -> None:
    """1) Display all student records."""
    print("\n===== ALL STUDENTS =====")
    print(dataframe.to_string(index=False))


def display_high_scorers(dataframe: pd.DataFrame, threshold: int = 70) -> None:
    """2) Display students with marks above a given threshold (default 70)."""
    high_scorers = dataframe[dataframe["Marks"] > threshold]
    print(f"\n===== STUDENTS WITH MARKS ABOVE {threshold} =====")
    print(high_scorers.to_string(index=False))


def calculate_average_marks(dataframe: pd.DataFrame) -> float:
    """3) Calculate and return the average marks of all students."""
    average_marks = dataframe["Marks"].mean()
    print(f"\n===== AVERAGE MARKS =====\n{average_marks:.2f}")
    return average_marks


def find_highest_scorer(dataframe: pd.DataFrame) -> pd.Series:
    """4) Find and return the student record with the highest marks."""
    top_student = dataframe.loc[dataframe["Marks"].idxmax()]
    print("\n===== STUDENT WITH HIGHEST MARKS =====")
    print(top_student.to_string())
    return top_student


def find_lowest_scorer(dataframe: pd.DataFrame) -> pd.Series:
    """5) Find and return the student record with the lowest marks."""
    bottom_student = dataframe.loc[dataframe["Marks"].idxmin()]
    print("\n===== STUDENT WITH LOWEST MARKS =====")
    print(bottom_student.to_string())
    return bottom_student


def display_total_students(dataframe: pd.DataFrame) -> int:
    """6) Display the total number of students in the dataset."""
    total = len(dataframe)
    print(f"\n===== TOTAL NUMBER OF STUDENTS =====\n{total}")
    return total


def main():
    display_all_students(df)
    display_high_scorers(df, threshold=70)
    calculate_average_marks(df)
    find_highest_scorer(df)
    find_lowest_scorer(df)
    display_total_students(df)


if __name__ == "__main__":
    main()
