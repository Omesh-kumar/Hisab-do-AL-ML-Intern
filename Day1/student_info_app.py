"""
Day 1 Practical Task - HisabDo Internship Bootcamp (AI/ML Track)
Project: Student Info Collector

This project:
1. Takes basic information from the user.
2. Stores information using variables and lists.
3. Uses if/else conditions.
4. Uses a loop.
5. Uses at least one function.
6. Displays the final output.
"""

students = []  # list to store all student records


def get_grade(marks):
    """Takes marks and returns a grade using if/else conditions."""
    if marks >= 90:
        return "A+"
    elif marks >= 80:
        return "A"
    elif marks >= 70:
        return "B"
    elif marks >= 60:
        return "C"
    else:
        return "F"


def display_student(student):
    """Displays a single student's info in a clean format."""
    print("-" * 40)
    print(f"Name   : {student['name']}")
    print(f"Age    : {student['age']}")
    print(f"City   : {student['city']}")
    print(f"Marks  : {student['marks']}")
    print(f"Grade  : {student['grade']}")
    print("-" * 40)


def main():
    print("=" * 40)
    print(" HisabDo Bootcamp - Student Info App ")
    print("=" * 40)

    while True:
        # Taking basic information from the user
        name = input("Enter student name: ")
        age = input("Enter student age: ")
        city = input("Enter student city: ")
        marks = int(input("Enter student marks (0-100): "))

        # Using the function to calculate grade
        grade = get_grade(marks)

        # Storing information using a dictionary inside a list
        student = {
            "name": name,
            "age": age,
            "city": city,
            "marks": marks,
            "grade": grade
        }
        students.append(student)

        # if/else condition example
        if grade == "F":
            print(f"\n{name}, you need to work harder. Keep trying!\n")
        else:
            print(f"\nWell done {name}! You passed with grade {grade}.\n")

        # Loop control
        another = input("Add another student? (yes/no): ").strip().lower()
        if another != "yes":
            break

    # Final output using a loop
    print("\n\nFINAL REPORT - ALL STUDENTS")
    print("=" * 40)
    if len(students) == 0:
        print("No student records found.")
    else:
        for s in students:
            display_student(s)

    print(f"\nTotal students recorded: {len(students)}")


if __name__ == "__main__":
    main()
