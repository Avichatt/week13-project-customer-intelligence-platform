import pytest
from unittest.mock import patch, MagicMock
from src.rag.generate import GroundedComplaintAssistant

@pytest.fixture
def mock_retriever_docs():
    return [
        {
            "complaint_id": "12345",
            "product": "Credit card",
            "issue": "Unauthorized charges",
            "consumer_complaint_narrative": "Someone charged $500 on my card.",
            "company": "Chase",
            "company_response": "Closed with explanation"
        }
    ]

@patch("src.rag.generate.ComplaintRetriever")
@patch("google.generativeai.GenerativeModel")
def test_grounded_assistant_success(mock_gemini_model, mock_retriever_class, mock_retriever_docs):
    # Setup mock retriever instance
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.retrieve.return_value = mock_retriever_docs
    mock_retriever_class.return_value = mock_retriever_instance
    
    # Setup mock Gemini LLM response
    mock_response = MagicMock()
    mock_response.text = "Based on [Complaint ID: 12345], there was an unauthorized charge of $500."
    
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value = mock_response
    mock_gemini_model.return_value = mock_model_instance
    
    # Run assistant
    assistant = GroundedComplaintAssistant()
    res = assistant.generate_answer("What unauthorized charges occurred?", threshold=0.1)
    
    assert res["evidence_ids"] == ["12345"]
    assert "12345" in res["answer"]
    assert "Sufficient evidence found" in res["sufficiency_note"]

@patch("src.rag.generate.ComplaintRetriever")
def test_grounded_assistant_empty_refusal(mock_retriever_class):
    # Setup mock retriever instance to return empty list (nothing passes threshold)
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.retrieve.return_value = []
    mock_retriever_class.return_value = mock_retriever_instance
    
    assistant = GroundedComplaintAssistant()
    res = assistant.generate_answer("How to cancel credit card?", threshold=0.4)
    
    assert res["evidence_ids"] == []
    assert "couldn't find any relevant complaints" in res["answer"]
    assert "Refused answer" in res["sufficiency_note"]
