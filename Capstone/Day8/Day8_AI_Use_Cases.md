# Day 8 – AI/ML Use Case Analysis for HisabDo

**Intern:** Omesh Lakhani (CSC-23S-152)
**Track:** AI/ML — HisabDo Capstone
**Date:** Day 8

## App Overview
HisabDo is a digital hisab-kitab (expense/income ledger) app used to record transactions, track spending, and manage personal/business finances via Website, Web App, and Mobile App. Core existing workflows: manual transaction entry, category tagging, balance viewing, basic reports.

## Identified AI/ML Use Cases

### 1. Smart Expense Categorization
1. **Problem:** Users manually pick a category for every transaction, which is slow and inconsistent.
2. **AI Solution:** Auto-classify transactions into categories (Food, Bills, Transport, etc.) based on description/merchant text.
3. **Input Data:** Transaction title, amount, merchant name, past user categorization history.
4. **Output:** Suggested category (with confidence score), auto-applied or one-tap confirm.
5. **Technology:** NLP text classification (TF-IDF/embeddings + classifier, or fine-tuned small LLM).
6. **Possible API/Model:** scikit-learn (Naive Bayes/Logistic Regression) for POC; Hugging Face DistilBERT or Groq/OpenAI LLM for production.
7. **Integration:** Web App, Mobile App (at transaction entry point).

### 2. Receipt/OCR Processing
1. **Problem:** Manually typing every receipt item wastes time and causes entry errors.
2. **AI Solution:** Scan/upload a receipt photo, auto-extract amount, date, vendor, and items.
3. **Input Data:** Receipt image (camera/gallery upload).
4. **Output:** Pre-filled transaction form (amount, date, category suggestion).
5. **Technology:** OCR + structured data extraction.
6. **Possible API/Model:** Google ML Kit / Tesseract OCR (POC), Google Cloud Vision API or AWS Textract (production).
7. **Integration:** Mobile App (camera), Web App (upload).

### 3. AI Financial Assistant / Chatbot
1. **Problem:** Users can't quickly ask "how much did I spend on food this month?" without digging through reports.
2. **AI Solution:** Conversational assistant that answers finance questions in natural language using the user's own transaction data.
3. **Input Data:** User query (text/voice) + user's transaction history.
4. **Output:** Natural-language answer, optionally a chart/summary.
5. **Technology:** RAG (Retrieval-Augmented Generation) over user's transaction DB + LLM.
6. **Possible API/Model:** Groq (Llama 3.3 70B) or OpenAI GPT-4o-mini, LangChain/LangGraph for orchestration.
7. **Integration:** Website (help widget), Web App, Mobile App.

### 4. Financial Insights & Spending Pattern Detection
1. **Problem:** Users don't notice spending trends or anomalies until it's too late.
2. **AI Solution:** Auto-generate weekly/monthly insights (e.g., "Food spending up 30% this month", unusual transaction alerts).
3. **Input Data:** Historical transactions, time-series spending per category.
4. **Output:** Insight cards/notifications with trend summaries.
5. **Technology:** Time-series analysis, anomaly detection (statistical + ML).
6. **Possible API/Model:** pandas + scikit-learn (Isolation Forest for anomalies), Prophet for trend forecasting.
7. **Integration:** Web App, Mobile App (dashboard/notifications).

### 5. Budget Recommendation Engine
1. **Problem:** Users don't know how to set realistic budgets per category.
2. **AI Solution:** Suggest personalized monthly budgets per category based on income and past spending behavior.
3. **Input Data:** Income, historical category-wise spending, savings goals.
4. **Output:** Recommended budget limits per category with rationale.
5. **Technology:** Regression/rule-based hybrid model on historical spend patterns.
6. **Possible API/Model:** scikit-learn Regression models, or simple statistical percentile-based rules for POC.
7. **Integration:** Web App, Mobile App.

### 6. Predictive Cash Flow / Balance Forecasting
1. **Problem:** Users don't know if they'll run short on money before month-end.
2. **AI Solution:** Predict end-of-month balance based on recurring income/expenses and current spending rate.
3. **Input Data:** Historical balance data, recurring transactions, current month's transactions so far.
4. **Output:** Forecasted balance graph + low-balance warning.
5. **Technology:** Time-series forecasting.
6. **Possible API/Model:** Facebook Prophet, ARIMA, or simple linear regression for POC.
7. **Integration:** Web App, Mobile App.

### 7. Smart Reminders for Bills/Recurring Payments
1. **Problem:** Users forget recurring bill due dates, leading to late fees.
2. **AI Solution:** Detect recurring transaction patterns automatically and remind users before the due date.
3. **Input Data:** Transaction history (recurring amount/vendor/date patterns).
4. **Output:** Push/SMS/email reminder notifications.
5. **Technology:** Pattern detection/clustering on transaction dates and amounts.
6. **Possible API/Model:** scikit-learn clustering (DBSCAN) on transaction intervals; Firebase Cloud Messaging for delivery.
7. **Integration:** Mobile App (push notification), Web App (in-app alert).

## Summary Table

| # | Use Case | Platform(s) | Priority |
|---|----------|-------------|----------|
| 1 | Smart Expense Categorization | Web, Mobile | High |
| 2 | Receipt/OCR Processing | Mobile, Web | High |
| 3 | AI Financial Assistant/Chatbot | Website, Web, Mobile | High |
| 4 | Financial Insights & Anomaly Detection | Web, Mobile | Medium |
| 5 | Budget Recommendation Engine | Web, Mobile | Medium |
| 6 | Predictive Balance Forecasting | Web, Mobile | Medium |
| 7 | Smart Bill Reminders | Mobile, Web | Low-Medium |

## Top 2 Selected Features (for Architecture + POC)
1. **Smart Expense Categorization** — highest daily-use impact, easiest to POC, direct UX improvement.
2. **AI Financial Assistant/Chatbot** — highest strategic value, showcases RAG + LLM skills, differentiator feature.
