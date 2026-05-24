from fastapi import APIRouter, HTTPException
from app.schemas import RAGQueryRequest, RAGQueryResponse
from src.rag.generate import GroundedComplaintAssistant

router = APIRouter(prefix="/rag", tags=["RAG Service"])
assistant = GroundedComplaintAssistant()

@router.post("/ask-complaints", response_model=RAGQueryResponse)
async def ask_complaints_intelligence(payload: RAGQueryRequest):
    """
    RAG service answering complaint database queries using retrieved context.
    Enforces a strict similarity threshold gate and returns evidence citations.
    """
    try:
        response = assistant.generate_answer(
            question=payload.question,
            k=payload.k,
            threshold=payload.threshold
        )
        return RAGQueryResponse(**response)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG query generation failure: {str(e)}"
        )
