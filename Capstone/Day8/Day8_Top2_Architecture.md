# Top 2 AI Feature Architecture — HisabDo

## 1. Smart Expense Categorization

```
User (enters transaction)
      |
      v
Application (Web/Mobile - transaction form)
      |
      v
AI Service (Categorization API - FastAPI)
      |
      v
Model (TF-IDF + Logistic Regression / small LLM)
      |
      v
Response (predicted category + confidence)
      |
      v
Application (auto-fills category, user confirms)
```

**Flow explanation:** When a user types a transaction description ("Careem to office"), the app sends the text to a lightweight FastAPI service. The service vectorizes the text and passes it to a trained classifier, which returns the most likely category with a confidence score. If confidence is high, the app auto-fills the category; if low, it shows top-3 suggestions for the user to pick.

## 2. AI Financial Assistant / Chatbot

```
User (asks: "How much did I spend on food this month?")
      |
      v
Application (chat widget - Web/Mobile/Website)
      |
      v
AI Service (Chat API - FastAPI + LangChain)
      |
      v
Retrieval Layer (fetch user's transactions from DB, filtered by query intent)
      |
      v
Model/API (Groq Llama 3.3 70B - generates natural language answer using retrieved data)
      |
      v
Response (text answer + optional chart data)
      |
      v
Application (displays answer + chart in chat UI)
```

**Flow explanation:** The user's question goes to a backend chat service. The service first determines intent (e.g., "spending query"), pulls the relevant transaction data from HisabDo's database (this is the "retrieval" step, not open-web search), then sends both the question and retrieved data to the LLM as context. The LLM generates a grounded, accurate answer instead of guessing, and the app renders it back to the user, optionally with a chart.
