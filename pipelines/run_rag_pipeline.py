import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src import config
from src.data.download import download_cfpb_complaints
from src.rag.index import build_and_save_index

def run_pipeline(sample_mode=False):
    print("==================================================")
    print("STARTING CUSTOMER INTEL PLATFORM - RAG LANE PIPELINE")
    print("==================================================")
    
    # Define limit based on sample mode or full mode
    # For full mode, let's download 2000 records to keep it fast but production size
    limit = 50 if sample_mode else 1000
    download_limit = 100 if sample_mode else 2000
    
    # 1. Download complaints
    download_cfpb_complaints(limit=download_limit)
    
    # 2. Build Index
    csv_file = (
        config.SAMPLE_DATA_DIR / "cfpb_complaints_sample.csv"
        if sample_mode else
        config.RAW_DATA_DIR / "cfpb_complaints.csv"
    )
    
    print(f"Building FAISS vector index using embeddings from Gemini...")
    indexed_count = build_and_save_index(csv_file, limit=limit)
    
    print(f"RAG Pipeline completed. Indexed {indexed_count} complaints.")
    print("==================================================")
    return True

if __name__ == "__main__":
    sample_mode = "--sample" in sys.argv
    success = run_pipeline(sample_mode=sample_mode)
    sys.exit(0 if success else 1)
