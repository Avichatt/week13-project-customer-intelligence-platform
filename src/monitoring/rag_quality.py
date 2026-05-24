import json
from src.rag.generate import GroundedComplaintAssistant

# 10 Defined Evaluation Queries with expected classifications/keywords
EVAL_SUITE = [
    {
        "query": "What are the common issues with mortgage loans?",
        "expected_product": "Mortgage",
        "keywords": ["mortgage", "loan", "payment", "escrow"]
    },
    {
        "query": "How do customers describe credit reporting errors?",
        "expected_product": "Credit reporting",
        "keywords": ["credit", "report", "incorrect", "bureau", "information"]
    },
    {
        "query": "What are the complaints regarding debt collection phone calls?",
        "expected_product": "Debt collection",
        "keywords": ["debt", "collect", "phone", "call", "harass"]
    },
    {
        "query": "Are there issues with checking account overdraft fees?",
        "expected_product": "Checking or savings account",
        "keywords": ["overdraft", "fee", "checking", "account", "charge"]
    },
    {
        "query": "What are customer disputes on credit cards for unauthorized charges?",
        "expected_product": "Credit card",
        "keywords": ["credit card", "unauthorized", "charge", "dispute", "fraud"]
    },
    {
        "query": "What problems do students report with student loans?",
        "expected_product": "Student loan",
        "keywords": ["student", "loan", "payment", "servicer"]
    },
    {
        "query": "What are the complaints about auto loans or leases?",
        "expected_product": "Vehicle loan or lease",
        "keywords": ["auto", "car", "lease", "vehicle", "loan"]
    },
    {
        "query": "Are there complaints about bank transfer delays?",
        "expected_product": "Money transfer",
        "keywords": ["transfer", "wire", "delay", "send", "money"]
    },
    {
        "query": "What complaints involve credit repair services?",
        "expected_product": "Credit repair",
        "keywords": ["credit", "repair", "service", "fee", "promise"]
    },
    {
        "query": "What are complaints about prepaid cards or mobile wallets?",
        "expected_product": "Prepaid card",
        "keywords": ["prepaid", "card", "mobile", "wallet", "app"]
    }
]

def run_rag_evaluation(threshold=0.3):
    print("--- Running RAG Evaluation Suite (10 Test Cases) ---")
    assistant = GroundedComplaintAssistant()
    
    results = []
    total_score = 0
    refused_count = 0
    
    for idx, case in enumerate(EVAL_SUITE):
        query = case["query"]
        print(f"Test case {idx+1}: '{query}'")
        
        # Run assistant
        res = assistant.generate_answer(query, k=3, threshold=threshold)
        
        answer = res["answer"]
        evidence_ids = res["evidence_ids"]
        sufficiency_note = res["sufficiency_note"]
        
        # Check if answered or refused
        is_refused = "couldn't find any relevant" in answer or len(evidence_ids) == 0
        if is_refused:
            refused_count += 1
            passed = False
            score = 0
            feedback = "Refused due to insufficient context."
        else:
            # Heuristic score based on matching product or keyword presence
            # Does the answer contains at least some keywords?
            matched_keywords = [w for w in case["keywords"] if w in answer.lower()]
            keyword_score = len(matched_keywords) / len(case["keywords"])
            
            # Did the retrieval pull correct products?
            retrieved_products = [s["product"].lower() for s in res["evidence_snippets"]]
            expected_product_match = any(case["expected_product"].lower() in p for p in retrieved_products)
            
            score = 0.5 * (1.0 if expected_product_match else 0.0) + 0.5 * keyword_score
            passed = score >= 0.5
            feedback = f"Matched keywords: {matched_keywords}. Expected product matches: {expected_product_match}."
            
        total_score += score
        results.append({
            "test_case": idx + 1,
            "query": query,
            "passed": passed,
            "score": score,
            "is_refused": is_refused,
            "evidence_count": len(evidence_ids),
            "evidence_ids": evidence_ids,
            "feedback": feedback,
            "sufficiency_note": sufficiency_note
        })
        
    avg_score = total_score / len(EVAL_SUITE)
    success_rate = sum(1 for r in results if r["passed"]) / len(EVAL_SUITE)
    
    summary = {
        "average_score": avg_score,
        "success_rate": success_rate,
        "refused_rate": refused_count / len(EVAL_SUITE),
        "results": results
    }
    
    return summary

if __name__ == "__main__":
    import pprint
    summary = run_rag_evaluation()
    pprint.pprint(summary)
