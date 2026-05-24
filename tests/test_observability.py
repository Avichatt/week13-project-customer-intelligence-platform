import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_model_package():
    mock_pipeline = MagicMock()
    mock_pipeline.transform.return_value = [[0.1, 0.2, 0.3]]
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.1, 0.9]]
    mock_model.predict.return_value = [1]
    
    return {
        "preprocessor": mock_pipeline,
        "model": mock_model,
        "metrics": {"roc_auc": 0.90, "f1": 0.70},
        "run_id": "observability_test_run"
    }

def test_health_endpoint_contract(client, mock_model_package):
    app.state.model_package = mock_model_package
    app.state.rag_retriever = MagicMock()
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "model_version" in data
    assert "vector_index_version" in data
    assert data["model_version"] == "observability_test_run"
    assert data["vector_index_version"] == "v1.0.0"
    
    assert data["services"]["ml_service"]["model_version"] == "observability_test_run"
    assert data["services"]["rag_service"]["vector_index_version"] == "v1.0.0"

def test_metrics_endpoint_initial_state(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "healthy"
    assert "observability" in data
    assert "model_prediction_distribution" in data
    assert "rag_retrieval_metrics" in data
    assert data["observability"]["requests_total"] >= 0

@patch("app.ml_router.predict_batch")
def test_batch_score_endpoint(mock_predict_batch, client, mock_model_package):
    app.state.model_package = mock_model_package
    
    # Mock return list from predict_batch
    mock_predict_batch.return_value = [
        {"prediction": 1, "probability": 0.85, "model_run_id": "test_run", "model_metrics": {}},
        {"prediction": 0, "probability": 0.15, "model_run_id": "test_run", "model_metrics": {}}
    ]
    
    payload = [
        {
            "age": 35, "job": "management", "marital": "married", "education": "university.degree",
            "default": "no", "housing": "yes", "loan": "no", "contact": "cellular", "month": "may",
            "day_of_week": "mon", "campaign": 1, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
            "emp.var.rate": -1.8, "cons.price.idx": 92.893, "cons.conf.idx": -46.2, "euribor3m": 1.299,
            "nr.employed": 5099.1
        },
        {
            "age": 42, "job": "blue-collar", "marital": "single", "education": "basic.9y",
            "default": "no", "housing": "no", "loan": "no", "contact": "cellular", "month": "may",
            "day_of_week": "mon", "campaign": 1, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
            "emp.var.rate": -1.8, "cons.price.idx": 92.893, "cons.conf.idx": -46.2, "euribor3m": 1.299,
            "nr.employed": 5099.1
        }
    ]
    
    response = client.post("/ml/batch-score", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "scored_file_path" in data
    assert data["total_records"] == 2
    assert data["counts_by_priority_band"]["High Priority"] == 1
    assert data["counts_by_priority_band"]["Low Priority"] == 1

@patch("app.rag_router.assistant.generate_answer")
def test_ask_complaints_with_filters(mock_generate_answer, client):
    mock_generate_answer.return_value = {
        "answer": "Grounded response referencing [Complaint ID: 999].",
        "evidence_ids": ["999"],
        "evidence_snippets": [{"complaint_id": "999", "product": "Credit card", "issue": "Fraud", "snippet": "...", "similarity_score": 0.85}],
        "sufficiency_note": "Sufficient context found",
        "prompt_version": "1.0.0"
    }
    
    payload = {
        "question": "What credit card complaints exist for Equifax on 2015-08-31?",
        "k": 3,
        "threshold": 0.3,
        "product": "Credit card",
        "company": "Equifax",
        "date": "2015-08-31",
        "issue": "Fraud"
    }
    
    response = client.post("/rag/ask-complaints", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "999" in data["evidence_ids"]
    assert data["prompt_version"] == "1.0.0"
    
    # Assert filters were forwarded
    mock_generate_answer.assert_called_with(
        question="What credit card complaints exist for Equifax on 2015-08-31?",
        k=3,
        threshold=0.3,
        product="Credit card",
        company="Equifax",
        date="2015-08-31",
        issue="Fraud"
    )
