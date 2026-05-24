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
    mock_model.predict_proba.return_value = [[0.1, 0.9]] # High probability (90%)
    mock_model.predict.return_value = [1]
    
    return {
        "preprocessor": mock_pipeline,
        "model": mock_model,
        "metrics": {"roc_auc": 0.90, "f1": 0.70},
        "run_id": "integration_test_run"
    }

@patch("app.integration_router.ask_complaints_intelligence")
def test_integration_endpoint_success(mock_ask_complaints, client, mock_model_package):
    # Load state
    app.state.model_package = mock_model_package
    
    # Mock RAG output
    mock_ask_complaints.return_value = MagicMock(
        answer="Active issue found with credit card unauthorized charge [Complaint ID: 12345].",
        evidence_ids=["12345"],
        evidence_snippets=[{"complaint_id": "12345", "product": "Credit card", "issue": "Unauthorized charges", "snippet": "..."}],
        sufficiency_note="Sufficient evidence"
    )
    
    payload = {
        "customer": {
            "age": 42,
            "job": "admin.",
            "marital": "married",
            "education": "high.school",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "month": "may",
            "day_of_week": "wed",
            "campaign": 1,
            "pdays": 999,
            "previous": 0,
            "poutcome": "nonexistent",
            "emp.var.rate": -1.8,
            "cons.price.idx": 92.893,
            "cons.conf.idx": -46.2,
            "euribor3m": 1.299,
            "nr.employed": 5099.1
        },
        "complaints_question": "Are there credit card disputes?"
    }
    
    response = client.post("/integration/customer-intel", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Assert ML fields
    assert data["campaign_conversion"]["subscribe_prediction"] is True
    assert data["campaign_conversion"]["probability"] == 0.9
    assert data["campaign_conversion"]["risk_band"] == "High Priority"
    
    # Assert RAG fields
    assert data["complaint_intelligence"] is not None
    assert data["complaint_intelligence"]["evidence_ids"] == ["12345"]
    
    # Assert Recommendation
    assert "PROCEED WITH SENSITIVITY" in data["unified_outreach_recommendation"]
