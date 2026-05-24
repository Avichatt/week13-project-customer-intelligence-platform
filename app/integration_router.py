from fastapi import APIRouter, HTTPException, Request
from app.schemas import CustomerIntelligenceRequest, CustomerIntelligenceResponse
from app.ml_router import predict_campaign_conversion
from app.rag_router import ask_complaints_intelligence
from app.schemas import RAGQueryRequest

router = APIRouter(prefix="/integration", tags=["Spine Integration"])

@router.post("/customer-intel", response_model=CustomerIntelligenceResponse)
async def get_integrated_customer_intelligence(request: Request, payload: CustomerIntelligenceRequest):
    """
    Spine integration endpoint blending predictive scoring and grounded complaints intelligence
    to generate an actionable customer outreach and risk mitigation directive.
    """
    try:
        # 1. Run ML campaign conversion prediction
        ml_prediction = await predict_campaign_conversion(request, payload.customer)
        
        # 2. Run optional RAG complaints query
        rag_response = None
        if payload.complaints_question:
            # Construct a RAG query payload
            rag_payload = RAGQueryRequest(
                question=payload.complaints_question,
                k=3,
                threshold=0.3
            )
            rag_response = await ask_complaints_intelligence(rag_payload)
            
        # 3. Formulate unified outreach directive
        risk_band = ml_prediction.risk_band
        will_subscribe = ml_prediction.subscribe_prediction
        
        has_complaints = False
        if rag_response and len(rag_response.evidence_ids) > 0:
            has_complaints = "couldn't find any relevant" not in rag_response.answer.lower()
            
        recommendation = ""
        if will_subscribe and risk_band == "High Priority":
            if has_complaints:
                recommendation = (
                    "PROCEED WITH SENSITIVITY: Customer has high subscription probability, "
                    "but active service complaints exist in their segment. Route outreach through a senior agent "
                    "equipped to resolve complaints first before promoting the deposit product."
                )
            else:
                recommendation = (
                    "IMMEDIATE OUTREACH: High conversion probability with no recent complaint indicators. "
                    "Target with primary marketing campaign channels."
                )
        elif risk_band == "Medium Priority":
            if has_complaints:
                recommendation = (
                    "COMPLAINT WORKFLOW FIRST: Segment reports issues. Hold marketing outreach "
                    "until complaint resolution workflows clear."
                )
            else:
                recommendation = (
                    "NURTURE OUTREACH: Moderate conversion potential. Include in secondary email/SMS campaigns."
                )
        else:
            recommendation = (
                "MONITOR ONLY: Low subscription probability. Suppress from aggressive telemarketing campaigns "
                "to minimize outreach fatigue."
            )
            
        return CustomerIntelligenceResponse(
            campaign_conversion=ml_prediction,
            complaint_intelligence=rag_response,
            unified_outreach_recommendation=recommendation
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Spine integration endpoint processing failure: {str(e)}"
        )
