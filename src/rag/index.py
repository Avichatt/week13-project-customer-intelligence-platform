import time
import json
import faiss
import numpy as np
import pandas as pd
import google.generativeai as genai
from src import config
from src.data.features import clean_complaint_text

def get_gemini_embeddings(texts, batch_size=50):
    """
    Get embeddings from Gemini API. Supports batching to respect rate limits.
    """
    # Check if key is dummy or mock for offline testing/CI
    if config.GEMINI_API_KEY == "mock_key_for_testing" or not config.GEMINI_API_KEY.startswith("AIzaSy"):
        print("Using synthetic mock embeddings for testing/offline mode...")
        # text-embedding-004 has 768 dimensions
        mock_embs = np.random.randn(len(texts), 768).astype(np.float32)
        # Normalize to unit length (equivalent to cosine scaling)
        row_norms = np.linalg.norm(mock_embs, axis=1, keepdims=True)
        return mock_embs / (row_norms + 1e-10)

    genai.configure(api_key=config.GEMINI_API_KEY)
    embeddings = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Embedding batch {i // batch_size + 1} ({len(batch)} items)...")
        
        try:
            result = genai.embed_content(
                model=config.EMBEDDING_MODEL_NAME,
                contents=batch,
                task_type="retrieval_document"
            )
            # Extracted list of float lists
            batch_emb = result.get('embedding', [])
            embeddings.extend(batch_emb)
        except Exception as e:
            print(f"Error calling Gemini Embedding API: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            # Retry once
            try:
                result = genai.embed_content(
                    model=config.EMBEDDING_MODEL_NAME,
                    contents=batch,
                    task_type="retrieval_document"
                )
                embeddings.extend(result.get('embedding', []))
            except Exception as retry_err:
                print(f"Failed again: {retry_err}")
                raise retry_err
                
        # Optional sleep to avoid hit rate limits
        time.sleep(0.5)
        
    return np.array(embeddings, dtype=np.float32)

def build_and_save_index(csv_path, limit=1000):
    """
    Load complaints, preprocess, embed, build FAISS index, and save index & metadata.
    """
    print(f"Loading complaints from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Drop rows without narratives
    df = df.dropna(subset=["consumer_complaint_narrative"])
    
    # Limit number of records for indexing to save embedding calls and time
    if len(df) > limit:
        print(f"Limiting to first {limit} records for indexing...")
        df = df.head(limit)
        
    # Preprocess texts
    print("Preprocessing narrative texts...")
    df["cleaned_narrative"] = df["consumer_complaint_narrative"].apply(clean_complaint_text)
    
    # Let's create an enriched context string for each complaint to help retrieval
    # Includes metadata in the text block to make matches better
    df["context"] = df.apply(
        lambda row: f"Product: {row['product']}\nIssue: {row['issue']}\nNarrative: {row['cleaned_narrative']}", 
        axis=1
    )
    
    texts = df["context"].tolist()
    
    # Get embeddings
    print(f"Requesting embeddings for {len(texts)} complaints...")
    embeddings = get_gemini_embeddings(texts)
    
    dimension = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape}. Dimension: {dimension}")
    
    # Build FAISS index
    # We use IndexFlatIP (Inner Product) for cosine similarity assuming normalized vectors
    # Let's normalize embeddings first for cosine similarity
    faiss.normalize_L2(embeddings)
    
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save FAISS Index
    index_path = config.PROCESSED_DATA_DIR / "complaints_faiss.index"
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}")
    
    # Save Metadata sidecar
    # Map index offset to raw record details
    metadata = []
    for idx, row in df.iterrows():
        metadata.append({
            "complaint_id": str(row.get("complaint_id")),
            "date_received": str(row.get("date_received")),
            "product": str(row.get("product")),
            "issue": str(row.get("issue")),
            "consumer_complaint_narrative": str(row.get("consumer_complaint_narrative")),
            "company": str(row.get("company")),
            "company_response": str(row.get("company_response")),
            "state": str(row.get("state"))
        })
        
    metadata_path = config.PROCESSED_DATA_DIR / "complaints_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata sidecar to {metadata_path}")
    
    print("RAG Index building completed.")
    return len(metadata)

if __name__ == "__main__":
    import sys
    use_sample = "--sample" in sys.argv
    csv_file = (
        config.SAMPLE_DATA_DIR / "cfpb_complaints_sample.csv"
        if use_sample else
        config.RAW_DATA_DIR / "cfpb_complaints.csv"
    )
    
    if not csv_file.exists():
        print(f"File {csv_file} does not exist. Run download.py first.")
        sys.exit(1)
        
    # If sample mode, embed a smaller subset (e.g. 50 items) for speed
    limit = 50 if use_sample else 1000
    build_and_save_index(csv_file, limit=limit)
