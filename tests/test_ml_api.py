import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_model_package():
    # Mock preprocessor
    mock_pipeline = MagicMock()
    mock_pipeline.transform.return_value = [[0.1, 0.2, 0.3]]
    
    # Mock classifier model
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.2, 0.8]] # 80% subscription probability
    mock_model.predict.return_value = [1]
    
    return {
        "preprocessor": mock_pipeline,
        "model": mock_model,
        "metrics": {"roc_auc": 0.85, "f1": 0.60},
        "run_id": "test_run_12345"
    }

def test_health_check_endpoint_no_model(client):
    app.state.model_package = None
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["services"]["ml_service"]["status"] == "not_loaded"

def test_health_check_endpoint_with_model(client, mock_model_package):
    app.state.model_package = mock_model_package
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["services"]["ml_service"]["status"] == "loaded"
    assert data["services"]["ml_service"]["model_version"] == "test_run_12345"

def test_predict_endpoint_success(client, mock_model_package):
    app.state.model_package = mock_model_package
    
    payload = {
        "age": 35,
        "job": "management",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.857,
        "nr.employed": 5191.0
    }
    
    response = client.post("/ml/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["subscribe_prediction"] is True
    assert data["probability"] == 0.8
    assert data["risk_band"] == "High Priority"
    assert data["model_version"] == "test_run_12345"

def test_predict_endpoint_missing_features(client, mock_model_package):
    app.state.model_package = mock_model_package
    # Missing required 'age' field
    payload = {
        "job": "management",
        "marital": "married"
    }
    response = client.post("/ml/predict", json=payload)
    assert response.status_code == 422 # Validation error
