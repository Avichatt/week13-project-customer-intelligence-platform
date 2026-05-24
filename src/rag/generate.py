import google.generativeai as genai
from src import config
from src.rag.retrieve import ComplaintRetriever

class GroundedComplaintAssistant:
    def __init__(self):
        self.retriever = ComplaintRetriever()
        
    def generate_answer(
        self, 
        question: str, 
        k: int = 4, 
        threshold: float = 0.3,
        product: str | None = None,
        company: str | None = None,
        date: str | None = None,
        issue: str | None = None
    ) -> dict:
        """
        Retrieves relevant complaints, verifies threshold, applies metadata filters, and constructs a grounded response.
        """
        try:
            self.retriever.load()
            retrieved_docs = self.retriever.retrieve(
                question, 
                k=k, 
                threshold=threshold,
                product=product,
                company=company,
                date=date,
                issue=issue
            )
        except Exception as e:
            print(f"Failed to load or retrieve context: {e}")
            retrieved_docs = []
            
        if not retrieved_docs:
            return {
                "answer": "I'm sorry, but I couldn't find any relevant complaints in the database matching your query with sufficient confidence.",
                "evidence_ids": [],
                "evidence_snippets": [],
                "sufficiency_note": "Refused answer: No complaints crossed the similarity threshold of " + str(threshold),
                "prompt_version": "1.0.0"
            }
            
        # Build prompt context
        context_blocks = []
        evidence_ids = []
        evidence_snippets = []
        
        for idx, doc in enumerate(retrieved_docs):
            doc_id = doc.get("complaint_id", f"UNK_{idx}")
            evidence_ids.append(doc_id)
            snippet = doc.get("consumer_complaint_narrative", "")[:300]
            evidence_snippets.append({
                "complaint_id": doc_id,
                "product": doc.get("product"),
                "issue": doc.get("issue"),
                "snippet": snippet,
                "similarity_score": doc.get("similarity_score")
            })
            
            context_blocks.append(
                f"Complaint ID: {doc_id}\n"
                f"Product: {doc.get('product')}\n"
                f"Issue: {doc.get('issue')}\n"
                f"Narrative: {doc.get('consumer_complaint_narrative')}\n"
                "----------------------------------------"
            )
            
        context_str = "\n".join(context_blocks)
        
        prompt = f"""
You are an expert Customer Support & Operations intelligence analyst at Meridian Financial.
Your goal is to answer the user's question about customer complaints using ONLY the provided complaints context.

Instructions:
1. Base your answer solely on the complaints provided below.
2. For each fact or claim, cite the relevant Complaint ID(s) directly in the text (e.g., "[Complaint ID: 123456]").
3. If the complaints provided do not contain enough information to answer the question, state that clearly and refuse to make up information.
4. Keep the answer professional, analytical, and structured.

Complaints Context:
{context_str}

User Question: {question}

Response:
"""
        
        # Call Gemini LLM
        if config.GEMINI_API_KEY == "mock_key_for_testing" or not config.GEMINI_API_KEY.startswith("AIzaSy"):
            print("Using mock LLM response for testing/offline mode...")
            evidence_ids_str = ", ".join(evidence_ids)
            answer = f"Based on review of complaints [Complaint ID: {evidence_ids_str}], there are active concerns noted regarding the product segment. Please address customer complaints directly."
            sufficiency_note = f"Sufficient evidence found (Offline Mock): {len(retrieved_docs)} matching complaints."
        else:
            genai.configure(api_key=config.GEMINI_API_KEY)
            try:
                model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
                response = model.generate_content(prompt)
                answer = response.text.strip()
                sufficiency_note = "Sufficient evidence found: " + str(len(retrieved_docs)) + " matching complaints cited."
            except Exception as e:
                print(f"Error calling Gemini LLM: {e}")
                answer = "Error generating grounded answer due to API failures."
                sufficiency_note = f"LLM error: {e}"
            
        return {
            "answer": answer,
            "evidence_ids": evidence_ids,
            "evidence_snippets": evidence_snippets,
            "sufficiency_note": sufficiency_note,
            "prompt_version": "1.0.0"
        }

if __name__ == "__main__":
    assistant = GroundedComplaintAssistant()
    # Test query
    response = assistant.generate_answer("Are there any issues with debt collection practices or calls?")
    print("Answer:")
    print(response["answer"])
    print("\nEvidence IDs:")
    print(response["evidence_ids"])
    print("\nSufficiency Note:")
    print(response["sufficiency_note"])
