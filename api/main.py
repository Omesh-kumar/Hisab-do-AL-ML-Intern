"""
Day 7 - HisabDo AI/ML Internship
FastAPI service that serves the tuned RandomForestClassifier (Student
Performance Prediction model) built in model/train_model.py.

Run with:
    uvicorn main:app --reload

Docs (Swagger UI):
    http://127.0.0.1:8000/docs
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")

MODEL_PATH = os.path.join(MODEL_DIR, "student_performance_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_order.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_order = joblib.load(FEATURES_PATH)
except FileNotFoundError as e:
    raise RuntimeError(
        "Model files not found. Run model/train_model.py first to generate "
        "student_performance_model.pkl, scaler.pkl and feature_order.pkl."
    ) from e

LABEL_MAP = {0: "Fail", 1: "Pass"}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Student Performance Prediction API",
    description="HisabDo AI/ML Internship - Day 7 Task. Predicts whether a "
                 "student will Pass or Fail based on attendance and scores.",
    version="1.0.0",
)


class StudentData(BaseModel):
    attendance: float = Field(..., ge=0, le=100, description="Attendance percentage (0-100)")
    assignment_score: float = Field(..., ge=0, le=100, description="Assignment score (0-100)")
    midterm_score: float = Field(..., ge=0, le=100, description="Midterm score (0-100)")
    final_score: float = Field(..., ge=0, le=100, description="Final exam score (0-100)")

    @field_validator("attendance", "assignment_score", "midterm_score", "final_score")
    @classmethod
    def check_is_number(cls, v):
        if v is None or np.isnan(v):
            raise ValueError("Value must be a valid number")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "attendance": 85,
                "assignment_score": 78,
                "midterm_score": 70,
                "final_score": 65,
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict


@app.get("/")
def root():
    return {
        "message": "Student Performance Prediction API is running.",
        "endpoint": "/predict",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentData):
    try:
        # Build the feature vector in the exact order used during training
        input_dict = {
            "Attendance": student.attendance,
            "AssignmentScore": student.assignment_score,
            "MidtermScore": student.midterm_score,
            "FinalScore": student.final_score,
        }
        input_vector = np.array([[input_dict[f] for f in feature_order]])

        # Scale using the same scaler fit during training
        input_scaled = scaler.transform(input_vector)

        # Predict
        pred_class = model.predict(input_scaled)[0]
        pred_proba = model.predict_proba(input_scaled)[0]

        prediction_label = LABEL_MAP[int(pred_class)]
        confidence = float(np.max(pred_proba))

        return PredictionResponse(
            prediction=prediction_label,
            confidence=round(confidence, 4),
            probabilities={
                "Fail": round(float(pred_proba[0]), 4),
                "Pass": round(float(pred_proba[1]), 4),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input or prediction error: {str(e)}")
