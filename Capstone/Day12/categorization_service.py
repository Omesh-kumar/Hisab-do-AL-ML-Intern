"""
HisabDo Capstone - Day 12
AI Service / API Layer: Smart Expense Categorization

This wraps the Day 11 model in a real API service (FastAPI), so the
HisabDo mobile/web app could call it over HTTP exactly like a production
backend would.

Run locally with:
    uvicorn categorization_service:app --reload

Endpoint:
    POST /categorize
    Body: { "transaction_id": str, "description": str, "amount": float, "date": str, "vendor": str (optional) }
"""

from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hisabdo-ai-service")

app = FastAPI(title="HisabDo AI Categorization Service", version="1.0")

# -----------------------------
# 1. Structured input handling (via Pydantic schema -> automatic validation)
# -----------------------------
class TransactionRequest(BaseModel):
    transaction_id: str
    description: str
    amount: float
    date: str
    vendor: Optional[str] = None

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("description must not be empty")
        return v

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v):
        if v <= 0:
            raise ValueError("amount must be greater than 0")
        return v


class CategorizationResponse(BaseModel):
    transaction_id: str
    status: str                 # "success" | "error"
    predicted_category: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    latency_ms: float


# -----------------------------
# Load / train the AI model (in production, this would load a saved model file)
# -----------------------------
TRAINING_DATA = [
    ("milk bread eggs grocery store", "Grocery"), ("vegetables fruits sabzi wala", "Grocery"),
    ("monthly grocery shopping", "Grocery"), ("rice flour atta chawal", "Grocery"),
    ("chicken meat mutton shop", "Grocery"), ("uber ride to office", "Transport"),
    ("careem fare", "Transport"), ("petrol fuel bike", "Transport"),
    ("bus fare to university", "Transport"), ("rickshaw fare", "Transport"),
    ("electricity bill payment", "Utilities"), ("gas bill sui gas", "Utilities"),
    ("internet wifi bill", "Utilities"), ("mobile balance recharge", "Utilities"),
    ("water bill payment", "Utilities"), ("dinner at restaurant", "Food & Dining"),
    ("lunch with friends cafe", "Food & Dining"), ("biryani order online", "Food & Dining"),
    ("chai samosa nashta", "Food & Dining"), ("pizza delivery", "Food & Dining"),
    ("tuition fee payment", "Education"), ("books stationery purchase", "Education"),
    ("university semester fee", "Education"), ("online course subscription", "Education"),
    ("exam form fee", "Education"), ("doctor consultation fee", "Health"),
    ("medicine pharmacy purchase", "Health"), ("hospital bill", "Health"),
    ("dental checkup", "Health"), ("lab test blood report", "Health"),
    ("shirt jeans clothing shop", "Shopping"), ("shoes purchase", "Shopping"),
    ("mobile phone accessories", "Shopping"), ("online shopping order", "Shopping"),
    ("gift for friend birthday", "Shopping"), ("client payment received sales", "Income"),
    ("salary credited", "Income"), ("freelance project payment", "Income"),
    ("shop daily sales collection", "Income"), ("customer udhar payment received", "Income"),
]

model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))])
model.fit([t[0] for t in TRAINING_DATA], [t[1] for t in TRAINING_DATA])

CONFIDENCE_THRESHOLD = 0.15  # below this, treat prediction as unreliable (hallucination guard)


# -----------------------------
# 2-4. AI request processing, response validation, error handling
# -----------------------------
@app.post("/categorize", response_model=CategorizationResponse)
def categorize(txn: TransactionRequest) -> CategorizationResponse:
    start = time.time()
    try:
        prediction = model.predict([txn.description])[0]
        confidence = float(max(model.predict_proba([txn.description])[0]))
        latency_ms = round((time.time() - start) * 1000, 2)

        # Response validation / low-confidence guard (prevents a confident-looking
        # wrong answer being silently trusted -- our version of hallucination control)
        if confidence < CONFIDENCE_THRESHOLD:
            logger.info(f"Low confidence ({confidence}) for txn {txn.transaction_id}, flagging for manual review")
            return CategorizationResponse(
                transaction_id=txn.transaction_id,
                status="error",
                error="Low model confidence — category needs manual selection",
                latency_ms=latency_ms
            )

        return CategorizationResponse(
            transaction_id=txn.transaction_id,
            status="success",
            predicted_category=prediction,
            confidence=round(confidence, 3),
            latency_ms=latency_ms
        )
    except Exception as e:
        # Catch-all so the service NEVER crashes or returns an unhandled 500
        # with no explanation -- always a structured, safe response.
        logger.error(f"Categorization failed for {txn.transaction_id}: {e}")
        return CategorizationResponse(
            transaction_id=txn.transaction_id,
            status="error",
            error="Internal categorization error, please try again",
            latency_ms=round((time.time() - start) * 1000, 2)
        )


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}
