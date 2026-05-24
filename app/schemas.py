from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional

# --- ML prediction schemas ---
class CustomerPredictionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(..., ge=18, le=100, description="Age of the client", json_schema_extra={"example": 35})
    job: str = Field(..., description="Type of job", json_schema_extra={"example": "management"})
    marital: str = Field(..., description="Marital status", json_schema_extra={"example": "married"})
    education: str = Field(..., description="Education level", json_schema_extra={"example": "university.degree"})
    default: str = Field(..., description="Has credit in default?", json_schema_extra={"example": "no"})
    housing: str = Field(..., description="Has housing loan?", json_schema_extra={"example": "yes"})
    loan: str = Field(..., description="Has personal loan?", json_schema_extra={"example": "no"})
    contact: str = Field(..., description="Contact communication type", json_schema_extra={"example": "cellular"})
    month: str = Field(..., description="Last contact month", json_schema_extra={"example": "may"})
    day_of_week: str = Field(..., description="Last contact day of week", json_schema_extra={"example": "mon"})
    campaign: int = Field(..., ge=1, description="Number of contacts during this campaign", json_schema_extra={"example": 2})
    pdays: int = Field(..., ge=0, le=999, description="Number of days since last contact from a previous campaign (999 means never contacted)", json_schema_extra={"example": 999})
    previous: int = Field(..., ge=0, description="Number of contacts performed before this campaign", json_schema_extra={"example": 0})
    poutcome: str = Field(..., description="Outcome of the previous marketing campaign", json_schema_extra={"example": "nonexistent"})
    emp_var_rate: float = Field(..., alias="emp.var.rate", description="Employment variation rate - quarterly indicator", json_schema_extra={"example": -1.8})
    cons_price_idx: float = Field(..., alias="cons.price.idx", description="Consumer price index - monthly indicator", json_schema_extra={"example": 92.893})
    cons_conf_idx: float = Field(..., alias="cons.conf.idx", description="Consumer confidence index - monthly indicator", json_schema_extra={"example": -46.2})
    euribor3m: float = Field(..., description="Euribor 3 month rate - daily indicator", json_schema_extra={"example": 1.299})
    nr_employed: float = Field(..., alias="nr.employed", description="Number of employees - quarterly indicator", json_schema_extra={"example": 5099.1})


class PredictionResponse(BaseModel):
    subscribe_prediction: bool = Field(..., description="Will the customer subscribe to a term deposit?")
    probability: float = Field(..., description="Confidence probability score [0.0, 1.0]")
    risk_band: str = Field(..., description="Outreach priority band based on probability (High, Medium, Low)")
    model_version: str = Field(..., description="Underlying tracking MLflow run or deployment ID")


# --- RAG Q&A schemas ---
class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Complaint intelligence query question", json_schema_extra={"example": "What are the main issues with credit cards?"})
    k: int = Field(default=3, ge=1, le=10, description="Number of context records to retrieve")
    threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Similarity threshold for documents")
    product: Optional[str] = Field(default=None, description="Optional product metadata filter", json_schema_extra={"example": "Credit card"})
    company: Optional[str] = Field(default=None, description="Optional company metadata filter", json_schema_extra={"example": "Equifax"})
    date: Optional[str] = Field(default=None, description="Optional date metadata filter (YYYY-MM-DD or partial)", json_schema_extra={"example": "2015-08-31"})
    issue: Optional[str] = Field(default=None, description="Optional issue metadata filter", json_schema_extra={"example": "Incorrect information"})


class EvidenceSnippet(BaseModel):
    complaint_id: str
    product: str
    issue: str
    snippet: str
    similarity_score: Optional[float] = None


class RAGQueryResponse(BaseModel):
    answer: str = Field(..., description="Grounded answer from LLM with citations")
    evidence_ids: List[str] = Field(..., description="Complaint IDs cited as evidence")
    evidence_snippets: List[EvidenceSnippet] = Field(..., description="Details of retrieved context snippets")
    sufficiency_note: str = Field(..., description="Description of retrieval success or refusal reasons")
    prompt_version: str = Field(default="1.0.0", description="Version of the RAG prompt template used")


# --- Unified Spine Integration schemas ---
class CustomerIntelligenceRequest(BaseModel):
    customer: CustomerPredictionRequest = Field(..., description="Demographics and financial features for campaign scoring")
    complaints_question: Optional[str] = Field(None, description="Optional question regarding complaint history or themes to resolve", json_schema_extra={"example": "Are there major complaints concerning banking services?"})
    product: Optional[str] = Field(default=None, description="Optional product metadata filter for complaints query")
    issue: Optional[str] = Field(default=None, description="Optional issue metadata filter for complaints query")
    date: Optional[str] = Field(default=None, description="Optional date metadata filter for complaints query")


class CustomerIntelligenceResponse(BaseModel):
    campaign_conversion: PredictionResponse
    complaint_intelligence: Optional[RAGQueryResponse] = Field(None, description="Complaint query intelligence response if requested")
    unified_outreach_recommendation: str = Field(..., description="Summary action recommendation blending ML and RAG observations")
