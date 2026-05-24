# Architectural Decision Log — Meridian Customer Intelligence Platform

This document logs critical architectural decisions, tradeoffs, and design rationales chosen during the development and hardening of the Meridian Customer Intelligence Platform.

---

## 1. Vector Database Search Metric
- **Decision**: Preferred `faiss.IndexFlatIP` (Inner Product) with unit-normalized vectors over standard Euclidean L2 distance (`faiss.IndexFlatL2`).
- **Rationale**: For text embedding representation, cosine similarity is widely proven to be more robust than Euclidean distance, as it evaluates the directional alignment (semantic meaning) rather than vector magnitude (length of narrative). By L2-normalizing both our document embeddings and query vectors prior to index injection and semantic retrieval, the Inner Product corresponds precisely to cosine similarity.
- **Alternatives Rejected**: Standard `IndexFlatL2` was discarded because narrative length variance introduces scaling distortions in Euclidean distance space.

---

## 2. Large Language Model Selection
- **Decision**: Standardized on Google Gemini `gemini-1.5-flash` for narrative question-answering, and `models/text-embedding-004` for semantic representation.
- **Rationale**: `gemini-1.5-flash` delivers extremely low latency (under 1.5 seconds response times), high concurrency capability, a massive context window, and exceptional reasoning for structured analytical summaries, matching our strict production budget. `text-embedding-004` generates dense 768-dimensional representations optimized for retrieval-augmented generation.
- **Alternatives Rejected**: Local models (e.g. LLaMA/Mistral) were rejected due to deployment latency overheads and resource requirements in our single-node runtime scope.

---

## 3. Observability and Performance Telemetry State
- **Decision**: In-memory thread-safe state dictionary (`app.state.metrics`) managed via unified FastAPI http middleware.
- **Rationale**: Surface real-time metrics (/metrics) including request latencies, endpoint counts, error rates, and classification distributions. Storing stats in-memory avoids adding database read/write bottlenecks during production traffic, keeping servicing latency under 15ms.
- **Alternatives Rejected**: External collectors (Prometheus/Redis) were deferred to stretch goals to keep the deployment package lightweight and zero-dependency.

---

## 4. Hybrid Metadata Filtering Scheme
- **Decision**: Retrieve-then-filter pattern (top-100 semantic recall pool refined in Python by exact metadata strings).
- **Rationale**: Pure FAISS lacks standard structural column SQL queries. Retrieving a dense pool of 100 semantic candidates from FAISS, then applying metadata constraints in memory, offers extremely fast query speeds (under 5ms) without requiring a multi-stage graph search or complex partition indexing.
