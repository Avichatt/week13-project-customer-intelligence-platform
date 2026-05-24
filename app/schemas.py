from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- ML prediction schemas ---
class CustomerPredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age of the client", example=35)
    job: str = Field(..., description="Type of job", example="management")
    marital: str = Field(..., description="Marital status", example="married")
    education: str = Field(..., description="Education level", example="university.degree")
    default: str = Field(..., description="Has credit in default?", example="no")
    housing: str = Field(..., description="Has housing loan?", example="yes")
    loan: str = Field(..., description="Has personal loan?", example="no")
    contact: str = Field(..., description="Contact communication type", example="cellular")
    month: str = Field(..., description="Last contact month", example="may")
    day_of_week: str = Field(..., description="Last contact day of week", example="mon")
    campaign: int = Field(..., ge=1, description="Number of contacts during this campaign", example=2)
    pdays: int = Field(..., ge=0, le=999, description="Number of days since last contact from a previous campaign (999 means never contacted)", example=999)
    previous: int = Field(..., ge=0, description="Number of contacts performed before this campaign", example=0)
    poutcome: str = Field(..., description="Outcome of the previous marketing campaign", example="nonexistent")
    emp_var_rate: float = Field(..., alias="emp.var.rate", description="Employment variation rate - quarterly indicator", example=-1.8)
    cons_price_idx: float = Field(..., alias="cons.price.idx", description="Consumer price index - monthly indicator", example=92.893)
    cons_conf_idx: float = Field(..., alias="cons.conf.idx", description="Consumer confidence index - monthly indicator", example=-46.2)
    euribor3m: float = Field(..., description="Euribor 3 month rate - daily indicator", example=1.299)
    nr_employed: float = Field(..., alias="nr.employed", description="Number of employees - quarterly indicator", example=5099.1)

    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    subscribe_prediction: bool = Field(..., description="Will the customer subscribe to a term deposit?")
    probability: float = Field(..., description="Confidence probability score [0.0, 1.0]")
    risk_band: str = Field(..., description="Outreach priority band based on probability (High, Medium, Low)")
    model_version: str = Field(..., description="Underlying tracking MLflow run or deployment ID")


# --- RAG Q&A schemas ---
class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Complaint intelligence query question", example="What are the main issues with credit cards?")
    k: int = Field(default=3, ge=1, le=10, description="Number of context records to retrieve")
    threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Similarity threshold for documents")

class EvidenceSnippet(BaseModel):
    complaint_id: str
    product: str
    issue: str
    snippet: str

class RAGQueryResponse(BaseModel):
    answer: str = Field(..., description="Grounded answer from LLM with citations")
    evidence_ids: List[str] = Field(..., description="Complaint IDs cited as evidence")
    evidence_snippets: List[EvidenceSnippet] = Field(..., description="Details of retrieved context snippets")
    sufficiency_note: str = Field(..., description="Description of retrieval success or refusal reasons")


# --- Unified Spine Integration schemas ---
class CustomerIntelligenceRequest(BaseModel):
    customer: CustomerPredictionRequest = Field(..., description="Demographics and financial features for campaign scoring")
    complaints_question: Optional[str] = Field(None, description="Optional question regarding complaint history or themes to resolve", example="Are there major complaints concerning banking services?")

class CustomerIntelligenceResponse(BaseModel):
    campaign_conversion: PredictionResponse
    complaint_intelligence: Optional[RAGQueryResponse] = Field(None, description="Complaint query intelligence response if requested")
    unified_outreach_recommendation: str = Field(..., description="Summary action recommendation blending ML and RAG observations")
