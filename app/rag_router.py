from fastapi import APIRouter, HTTPException, Request
from app.schemas import RAGQueryRequest, RAGQueryResponse
from src.rag.generate import GroundedComplaintAssistant

router = APIRouter(prefix="/rag", tags=["RAG Service"])
assistant = GroundedComplaintAssistant()

@router.post("/ask-complaints", response_model=RAGQueryResponse)
async def ask_complaints_intelligence(request: Request, payload: RAGQueryRequest):
    """
    RAG service answering complaint database queries using retrieved context.
    Enforces strict similarity threshold gates, applies optional metadata filters,
    and returns evidence citations with dynamic metric reporting.
    """
    try:
        response = assistant.generate_answer(
            question=payload.question,
            k=payload.k,
            threshold=payload.threshold,
            product=payload.product,
            company=payload.company,
            date=payload.date,
            issue=payload.issue
        )
        
        # Dynamically record RAG metrics inside the server state
        if hasattr(request.app.state, "metrics"):
            rag_stats = request.app.state.metrics["rag_retrieval_stats"]
            rag_stats["total_queries"] += 1
            
            ev_ids = response.get("evidence_ids", [])
            if len(ev_ids) == 0:
                rag_stats["total_refusals"] += 1
            else:
                rag_stats["total_evidence_ids_retrieved"] += len(ev_ids)
                
            snippets = response.get("evidence_snippets", [])
            for snip in snippets:
                score = snip.get("similarity_score")
                if score is not None:
                    rag_stats["similarity_scores_sum"] += score
                    rag_stats["similarity_scores_count"] += 1
                    
        return RAGQueryResponse(**response)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG query generation failure: {str(e)}"
        )
