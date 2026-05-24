import json
import faiss
import numpy as np
import google.generativeai as genai
from src import config

class ComplaintRetriever:
    def __init__(self):
        self.index_path = config.PROCESSED_DATA_DIR / "complaints_faiss.index"
        self.metadata_path = config.PROCESSED_DATA_DIR / "complaints_metadata.json"
        self.index = None
        self.metadata = None
        self.is_loaded = False
        
    def load(self):
        """Loads FAISS index and metadata sidecar."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("FAISS index or metadata sidecar not found. Run index.py first.")
            
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_path))
        
        print("Loading metadata...")
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)
            
        self.is_loaded = True
        return self

    def retrieve(
        self, 
        query: str, 
        k: int = 5, 
        threshold: float = 0.3,
        product: str | None = None,
        company: str | None = None,
        date: str | None = None,
        issue: str | None = None
    ) -> list:
        """
        Embed query, perform hybrid FAISS search, apply optional metadata filters,
        and return the top k matching documents that pass the similarity threshold.
        """
        if not self.is_loaded:
            self.load()
            
        # Check if key is dummy or mock for offline testing/CI
        if config.GEMINI_API_KEY == "mock_key_for_testing" or not config.GEMINI_API_KEY.startswith("AIzaSy"):
            print("Using synthetic query embedding for testing/offline mode...")
            query_vector = np.random.randn(1, 768).astype(np.float32)
        else:
            # Get query embedding
            genai.configure(api_key=config.GEMINI_API_KEY)
            try:
                result = genai.embed_content(
                    model=config.EMBEDDING_MODEL_NAME,
                    contents=query,
                    task_type="retrieval_query"
                )
                query_vector = np.array([result['embedding']], dtype=np.float32)
            except Exception as e:
                print(f"Error embedding query: {e}")
                return []
            
        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Search index with a larger candidate pool to accommodate subsequent metadata filters
        k_search = max(100, k * 5)
        scores, indices = self.index.search(query_vector, k_search)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
                
            # Filter by similarity threshold
            if score < threshold:
                continue
                
            doc = self.metadata[idx].copy()
            
            # Apply metadata filters
            if product and product.lower() not in doc.get("product", "").lower():
                continue
            if company and company.lower() not in doc.get("company", "").lower():
                continue
            if date and date not in doc.get("date_received", ""):
                continue
            if issue and issue.lower() not in doc.get("issue", "").lower():
                continue
                
            doc["similarity_score"] = float(score)
            results.append(doc)
            
            # Stop if we have accumulated enough matches
            if len(results) >= k:
                break
                
        return results

if __name__ == "__main__":
    retriever = ComplaintRetriever()
    try:
        retriever.load()
        matches = retriever.retrieve("credit card billing issues", k=3)
        print(f"Found {len(matches)} results:")
        for idx, doc in enumerate(matches):
            print(f"\n[{idx+1}] Score: {doc['similarity_score']:.4f} | ID: {doc['complaint_id']}")
            print(f"Product: {doc['product']} | Issue: {doc['issue']}")
            print(f"Narrative snippet: {doc['consumer_complaint_narrative'][:150]}...")
    except FileNotFoundError as e:
        print(e)
