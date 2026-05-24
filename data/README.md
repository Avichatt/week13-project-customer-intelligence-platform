# Meridian Customer Intelligence Platform — Data Layer Sourcing

This directory contains the datasets required to train the predictive machine learning models and construct the semantic RAG vector database.

## 📊 UCI Bank Marketing Dataset (ML Churn & Conversion Prediction)
- **Use Case**: Predicting customer conversion probability for term deposit subscriptions based on demographics and past campaign contacts.
- **Source Link**: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Description**: The data is related to direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

---

## 🏛️ CFPB Consumer Complaint Database (RAG Service Q&A)
- **Use Case**: Context base for Retrieval-Augmented Generation (RAG) providing grounded answers with cited IDs for customer service complaints.
- **Source Link**: [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- **Description**: A collection of complaints sent to consumer financial product and service companies. Narratives are normalized, redacted for PII, chunked, and semantic indexations are built via `models/text-embedding-004` to perform FAISS indexing.
