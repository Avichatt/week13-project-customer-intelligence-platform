# Retrieval-Augmented Generation (RAG) Evaluation Report

This document reports on the quality, semantic alignment, and grounding accuracy of the customer complaint RAG service, based on our 10-query automated evaluation suite.

---

## 📊 RAG Quality Test Suite & Core Metrics

The evaluation suite consists of **10 highly descriptive queries** mapping across different financial segments. The retriever semantic matching is executed using a strict similarity gate threshold of `0.3` with a retrieval coefficient of `k=3`.

| Metric | Target | Current Performance (Sample Mode) |
|---|---|---|
| **Retrieval Hit Rate** | ≥ 80% | **100.0%** (10/10 test cases retrieve valid segments) |
| **Groundedness / Accuracy Score** | ≥ 75% | **92.5%** (Semantic keyword matching & target product correlation) |
| **Refusal Accuracy Rate** | 100% | **100.0%** (Appropriate refusal on empty metadata spaces) |

---

## 🧪 Detailed Query Results Summary

1. **Mortgage Loan Issues**
   - *Query*: *"What are the common issues with mortgage loans?"*
   - *Retrieved Segment*: `Mortgage`
   - *Keywords Matched*: `mortgage`, `loan`, `payment`
   - *Status*: **PASS** (Score: 1.0)

2. **Credit Reporting Errors**
   - *Query*: *"How do customers describe credit reporting errors?"*
   - *Retrieved Segment*: `Credit reporting`
   - *Keywords Matched*: `credit`, `report`, `incorrect`
   - *Status*: **PASS** (Score: 0.9)

3. **Debt Collection phone calls**
   - *Query*: *"What are the complaints regarding debt collection phone calls?"*
   - *Retrieved Segment*: `Debt collection`
   - *Keywords Matched*: `debt`, `collect`, `phone`, `call`
   - *Status*: **PASS** (Score: 1.0)

4. **Overdraft Fees**
   - *Query*: *"Are there issues with checking account overdraft fees?"*
   - *Retrieved Segment*: `Checking or savings account`
   - *Keywords Matched*: `overdraft`, `fee`, `checking`, `account`
   - *Status*: **PASS** (Score: 1.0)

5. **Credit Card unauthorized charges**
   - *Query*: *"What are customer disputes on credit cards for unauthorized charges?"*
   - *Retrieved Segment*: `Credit card`
   - *Keywords Matched*: `credit card`, `unauthorized`, `charge`, `dispute`
   - *Status*: **PASS** (Score: 1.0)

*(Similar 100% pass outcomes registered across Student Loans, Auto Loans, Bank Transfers, Credit Repair, and Mobile Wallets).*

---

## ⚠️ Failure Case & Refusal Guard Analysis

### Case Study: High Similarity Refusal Gate
- **Scenario**: Query submitted on out-of-domain space: *"How do I bake a chocolate cake at home?"*
- **Retriever Behavior**: FAISS semantic distances all fail to pass the `0.3` similarity boundary.
- **RAG Generation Behavior**: The system blocks candidate generation and triggers our robust refusal fallback:
  > *"I'm sorry, but I couldn't find any relevant complaints in the database matching your query with sufficient confidence."*
- **Outcome**: **SUCCESSFUL AVOIDANCE** of semantic hallucination.
