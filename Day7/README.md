# Student Performance Prediction API
**HisabDo AI/ML Internship — Day 7 Task**

Deploys the Day 4–6 Student Performance model (tuned `RandomForestClassifier`,
GridSearchCV) as a REST API using **FastAPI**, so predictions can be made
outside the notebook.

---

## 📁 Project Structure
```
day7_api/
├── model/
│   ├── train_model.py                  # Trains & saves the model
│   ├── student_performance.csv         # Training dataset
│   ├── student_performance_model.pkl   # Trained RandomForestClassifier
│   ├── scaler.pkl                      # StandardScaler used at train time
│   └── feature_order.pkl               # Feature column order
├── api/
│   └── main.py                         # FastAPI app (/predict endpoint)
├── Student_Performance_API.postman_collection.json  # Import into Postman
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Install Dependencies

```bash
# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧠 (Optional) Retrain the Model

The trained model is already included in `model/`. If you want to regenerate
it (or swap in your own dataset):

```bash
cd model
python train_model.py
```

This saves `student_performance_model.pkl`, `scaler.pkl`, and
`feature_order.pkl` into the `model/` folder.

---

## 🚀 How to Run the API

```bash
cd api
uvicorn main:app --reload
```

The API will be available at: **http://127.0.0.1:8000**

Interactive Swagger docs: **http://127.0.0.1:8000/docs**

---

## 📡 Endpoint

### `POST /predict`

Predicts whether a student will **Pass** or **Fail** based on attendance and
scores.

#### Request Format
```json
{
  "attendance": 85,
  "assignment_score": 78,
  "midterm_score": 70,
  "final_score": 65
}
```

| Field              | Type  | Range  | Required |
|--------------------|-------|--------|----------|
| attendance         | float | 0–100  | ✅ |
| assignment_score   | float | 0–100  | ✅ |
| midterm_score      | float | 0–100  | ✅ |
| final_score        | float | 0–100  | ✅ |

#### Response Format
```json
{
  "prediction": "Pass",
  "confidence": 0.977,
  "probabilities": {
    "Fail": 0.023,
    "Pass": 0.977
  }
}
```

#### Invalid Input Handling
- Missing field → `422 Unprocessable Entity` with details of the missing field.
- Out-of-range value (e.g. `attendance: 150`) → `422 Unprocessable Entity`.
- Non-numeric / malformed body → `422` from FastAPI's built-in Pydantic validation.
- Any unexpected internal error during prediction → `400 Bad Request` with a
  descriptive message.

### `GET /health`
Simple health check — returns `{"status": "ok", "model_loaded": true}`.

### `GET /`
Root route with basic API info.

---

## 🧪 Testing with Postman

1. Open Postman → **Import** → select
   `Student_Performance_API.postman_collection.json` from this repo.
2. Make sure the API is running locally (`uvicorn main:app --reload`).
3. Run each request in the collection:
   - `Predict - Pass Case`
   - `Predict - Fail Case`
   - `Predict - Invalid Input (missing field)`
   - `Predict - Invalid Input (out of range)`
4. Take screenshots of each response and add them to a `screenshots/` folder
   in your repo for submission.

Example verified responses (from local testing):

**Valid request → Pass**
```json
{"prediction": "Pass", "confidence": 0.977, "probabilities": {"Fail": 0.023, "Pass": 0.977}}
```

**Valid request → Fail**
```json
{"prediction": "Fail", "confidence": 0.9783, "probabilities": {"Fail": 0.9783, "Pass": 0.0217}}
```

**Missing field**
```json
{"detail": [{"type": "missing", "loc": ["body", "final_score"], "msg": "Field required"}]}
```

**Out-of-range value**
```json
{"detail": [{"type": "less_than_equal", "loc": ["body", "attendance"], "msg": "Input should be less than or equal to 100"}]}
```

---

## 📊 Model Summary
- **Algorithm:** RandomForestClassifier (GridSearchCV tuned)
- **Features:** Attendance, Assignment Score, Midterm Score, Final Score
- **Preprocessing:** StandardScaler
- **Test Accuracy:** ~87%

---

## 🔗 Author
Omesh Kumar — [GitHub](https://github.com/Omesh-kumar) · [LinkedIn](https://linkedin.com/in/omesh-kumar-295971266)
