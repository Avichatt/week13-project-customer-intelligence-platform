# Production Hardening & Operational Guide

This document describes the operational design choices, concurrency patterns, security sanitization, and fallback options engineered to make the Meridian Customer Intelligence Platform enterprise-ready.

---

## 1. Concurrency & Serving Design
- **FastAPI Lifespan Startup Hook**: The predictive ML model package (`preprocessor` and `model` wrapper) and the FAISS semantic retriever index are loaded into memory exactly *once* during startup (`app.state`). Standard HTTP requests only read the initialized models from memory, eliminating disk I/O bottlenecks and ensuring that `/predict` and `/ask-complaints` run concurrently under high async load.
- **Async Execution Bounds**: Operational database reads and ML predictions run in thread-pools where blocking occurs, preventing CPU-bound inference operations from starving FastAPI’s event loop.

---

## 2. Security and Secret Sanitation
- **Git Hygiene**: Environment variables are strictly parsed from an uncommitted `.env` file (which is actively listed in `.gitignore`). Default mock configurations are implemented so the system runs smoothly out-of-the-box in local environments, preventing developers from committing sensitive API keys.
- **PII Redaction**: All raw consumer narratives from the CFPB database are processed using regex filters to gracefully scrub standard financial identifiers and placeholders ("XXXX" sequences) with `[REDACTED]` prior to embedding, ensuring compliance with data privacy standards.

---

## 3. Servicing Robustness & Error Routing
- **Similarity Threshold Guard**: For RAG queries, we enforce a strict similarity threshold gate (default `0.3`). If no retrieved document passes the threshold, the system immediately executes a fallback refusal response ("*I'm sorry, but I couldn't find any relevant complaints...*") instead of passing weak matches to the LLM, preventing semantic hallucination.
- **Input Sanitization**: Pydantic V2 models strictly validate input types (e.g. age bounds `18 <= age <= 100`, pdays constraint `0 <= pdays <= 999`), blocking invalid payloads at the gateway (HTTP 422) before they can cause downstream training or processing skew.

---

## 4. Disaster Recovery & Key Rotation
- **Mock Fallback Modes**: If the Gemini API key is missing or revoked, the embedding and generation stages fall back to synthetic mock representations and responses rather than crashing the API spine, keeping downstream services alive.
- **Model Hot-Reloading**: Prediction routes support checking model state reload triggers dynamically, supporting zero-downtime hot-swaps of `best_model.joblib` without requiring a service reboot.
