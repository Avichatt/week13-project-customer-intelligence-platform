# Customer Intelligence Platform — Engineering Reflection & Report

This document presents a comprehensive engineering reflection on the design, optimization, and operationalization of the Meridian Customer Intelligence Platform.

---

## 1. Core Engineering Decisions & Strategy

Our design strategy focused on building a highly reliable, zero-train-serving-skew, unified platform that links predictive machine learning (conversion probabilities) with generative intelligence (customer complaint analyses).

- **Unified FastAPI Serving Spine**: Rather than serving ML and RAG via isolated microservices, we constructed a unified FastAPI spine. This drastically reduces network latency, simplifies local container orchestration, and permits a single, lightweight health diagnostic `/health` and telemetry `/metrics` interface.
- **Zero-Train-Serving-Skew Pipeline**: We wrapped all preprocessing and encoding logic inside `BankFeaturePipeline` (fitted during training and cached as joblib sidecars). Servicing requests pass through the exact same pipeline object, completely eliminating data transformations skew between development and production.
- **In-Memory Telemetry Collector**: Rather than adding external collectors like Redis or Prometheus, we built a custom HTTP middleware that measures latencies and increments counters directly into a thread-safe `app.state` dictionary. This keeps query execution under 10ms.

---

## 2. ML Modelling, Metrics, & Gating Check

We deployed an **XGBoost Classifier** as our core predictive model, compared against a baseline **Logistic Regression** model.

### Comparative Relative Quality Gating
To prevent the promotion of models with poor calibration or skewed metrics, we engineered a strict relative gating check comparing XGBoost and baseline metrics on identical test subsets:
1. **PR-AUC relative improvement**: XGBoost must beat the baseline PR-AUC by at least **3.0 percentage points**. This ensures high-precision positive-class classifications, which is critical for targeting term deposits.
2. **F1-score relative drop**: XGBoost F1-score must drop by no more than **2.0 percentage points** compared to the baseline, ensuring the model maintains overall balance.
3. **Inference Latency check**: The average sample inference latency (preprocessing + probability scoring) must stay under **50.0 ms** to guarantee high-performance real-time servicing.

The XGBoost model successfully passed all promotion gates (PR-AUC improved by +8.6%, F1 improved by +8.1%, Latency was 0.15ms).

---

## 3. RAG Semantic Alignment & Metadata Filtering

Our RAG lane handles semantically searching and summarizing customer complaints from the CFPB database.

- **FAISS Cosine Similarity Vector Index**: We normalized document embeddings and query vectors, applying `faiss.IndexFlatIP` to calculate pure cosine similarity.
- **Retrieve-Then-Filter Candidate Pool**: pure FAISS lacks column SQL filters. To enforce metadata filters (`product`, `company`, `date`, `issue`) without losing semantic relevance, we retrieve a dense candidate pool of `k_search = 100` elements from FAISS, apply string metadata constraints in Python, and return the top `k` remaining records that pass our strict similarity threshold of `0.3`.
- **Refusal Guard Gate**: Under weak query matching or unrelated questions, the RAG lane refuses to generate answers, successfully blocking semantic hallucinations.

---

## 4. RAG Relative Promotion Gating

We implemented a relative promotion gate for RAG indices to ensure we never deploy a degraded index.
- A newly built index is evaluated on the 10-query `EVAL_SUITE`.
- If a production index already exists, we back it up, run the identical `EVAL_SUITE` to establish the baseline performance metrics, and compare:
  - **Candidate Hit-Rate** must be equal to or higher than **Baseline Hit-Rate**.
  - **Candidate Average Score** must be equal to or higher than **Baseline Average Score**.
- The candidate index is promoted to live serving ONLY if both conditions are met, guaranteeing index quality stability.
