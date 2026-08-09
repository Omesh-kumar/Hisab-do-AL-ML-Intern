# Day 9 – AI Feature Specification: Smart Expense Categorization

**Intern:** Omesh Lakhani (CSC-23S-152) | **Track:** AI/ML — HisabDo Capstone

## 1. Selected Primary Feature
**Smart Expense Categorization** — chosen because it's the highest-frequency user action (every transaction), has clear measurable impact, and is achievable as a fast, reliable POC within one day.

## 2. Complete Workflow

| Stage | Detail |
|---|---|
| **Input** | Transaction description text (e.g. "Karachi Electric bill payment"), optionally amount |
| **Processing** | Text cleaned → vectorized (TF-IDF) → passed to trained classifier |
| **AI/ML Model** | Multinomial Naive Bayes / Logistic Regression (scikit-learn) trained on labeled transaction descriptions |
| **Output** | Predicted category + confidence score, returned as JSON |

## 3. Research: Suitable Models/APIs

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| TF-IDF + Naive Bayes/LogReg (scikit-learn) | Fast, free, runs offline, easy to explain | Needs labeled training data, limited on unseen vocabulary | **Chosen for POC** |
| Hugging Face DistilBERT (fine-tuned) | Better accuracy, understands context | Needs GPU/more data, heavier for POC | Future upgrade |
| Groq/OpenAI LLM (zero-shot prompt) | No training data needed, handles free text well | API cost, latency, less consistent for structured labels | Production alternative for edge cases |

**Decision:** Start with scikit-learn for POC (fast, interpretable, zero cost), upgrade to LLM-assisted fallback for descriptions the classifier is unsure about.

## 4. Integration Plan

- **Website:** Not applicable directly (marketing site), but can showcase feature in a demo/product tour section.
- **Web Application:** Categorization runs as an API call when user types a transaction description; category auto-fills in the form field with an "edit" option.
- **Mobile Application:** Same API called on transaction entry screen; also used right after OCR receipt scan to auto-categorize extracted items.

## 5. Chosen Technology — Short Explanation
For the POC, I used **scikit-learn's TF-IDF vectorizer + Multinomial Naive Bayes classifier** wrapped in a simple Python function (deployable as a FastAPI endpoint, consistent with my Day 1–7 FastAPI work). This was chosen because it trains instantly on a small labeled dataset, requires no external API/cost, and is easy to explain and extend. For production, this would be retrained periodically on real HisabDo transaction data, with a confidence threshold below which the system falls back to an LLM prompt for better accuracy on ambiguous text.
