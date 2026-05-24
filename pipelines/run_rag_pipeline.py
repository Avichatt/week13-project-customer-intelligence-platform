import sys
import shutil
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src import config
from src.data.download import download_cfpb_complaints
from src.rag.index import build_and_save_index
from src.monitoring.rag_quality import run_rag_evaluation

def run_pipeline(sample_mode=False):
    print("==================================================")
    print("STARTING CUSTOMER INTEL PLATFORM - RAG LANE PIPELINE")
    print("==================================================")
    
    # Define limit based on sample mode or full mode
    limit = 50 if sample_mode else 1000
    download_limit = 100 if sample_mode else 2000
    
    # 1. Download complaints
    download_cfpb_complaints(limit=download_limit)
    
    # 2. Setup paths for relative promotion gating
    index_path = config.PROCESSED_DATA_DIR / "complaints_faiss.index"
    metadata_path = config.PROCESSED_DATA_DIR / "complaints_metadata.json"
    
    prod_index_backup = config.PROCESSED_DATA_DIR / "complaints_faiss.index.prod"
    prod_metadata_backup = config.PROCESSED_DATA_DIR / "complaints_metadata.json.prod"
    
    candidate_index_temp = config.PROCESSED_DATA_DIR / "complaints_faiss.index.candidate"
    candidate_metadata_temp = config.PROCESSED_DATA_DIR / "complaints_metadata.json.candidate"
    
    # Backup existing production files if they exist
    has_existing_prod = index_path.exists() and metadata_path.exists()
    if has_existing_prod:
        print("Backing up current production RAG index...")
        shutil.copy(str(index_path), str(prod_index_backup))
        shutil.copy(str(metadata_path), str(prod_metadata_backup))
        
    # 3. Build candidate index (writes directly to standard index_path and metadata_path)
    csv_file = (
        config.SAMPLE_DATA_DIR / "cfpb_complaints_sample.csv"
        if sample_mode else
        config.RAW_DATA_DIR / "cfpb_complaints.csv"
    )
    
    print(f"Building candidate FAISS vector index using embeddings from Gemini...")
    indexed_count = build_and_save_index(csv_file, limit=limit)
    
    # 4. Relative Gating Check
    print("Evaluating candidate RAG index performance...")
    threshold = 0.2 if sample_mode else 0.3
    candidate_summary = run_rag_evaluation(threshold=threshold)
    candidate_hit_rate = 1.0 - candidate_summary["refused_rate"]
    candidate_score = candidate_summary["average_score"]
    
    gate_passed = True
    gate_reason = "No prior production model exists. Initial RAG index auto-promoted."
    
    if has_existing_prod:
        # Move candidate files to temp candidate locations
        shutil.move(str(index_path), str(candidate_index_temp))
        shutil.move(str(metadata_path), str(candidate_metadata_temp))
        
        # Restore production backup to standard locations for baseline evaluation
        shutil.move(str(prod_index_backup), str(index_path))
        shutil.move(str(prod_metadata_backup), str(metadata_path))
        
        print("Evaluating current production RAG index baseline...")
        baseline_summary = run_rag_evaluation(threshold=threshold)
        baseline_hit_rate = 1.0 - baseline_summary["refused_rate"]
        baseline_score = baseline_summary["average_score"]
        
        print(f"Baseline RAG - Hit Rate: {baseline_hit_rate * 100:.1f}%, Avg Score: {baseline_score * 100:.1f}%")
        print(f"Candidate RAG - Hit Rate: {candidate_hit_rate * 100:.1f}%, Avg Score: {candidate_score * 100:.1f}%")
        
        # Gating conditions: Candidate must equal or beat baseline hit-rate and score
        hit_rate_ok = candidate_hit_rate >= baseline_hit_rate
        score_ok = candidate_score >= baseline_score
        
        if hit_rate_ok and score_ok:
            gate_passed = True
            gate_reason = "Candidate exceeds or matches current production performance."
            # Promote candidate: copy temp candidate files over standard files
            shutil.move(str(candidate_index_temp), str(index_path))
            shutil.move(str(candidate_metadata_temp), str(metadata_path))
            print("Promotion Success! Candidate RAG index is now live in production.")
        else:
            gate_passed = False
            reasons = []
            if not hit_rate_ok:
                reasons.append(f"Hit Rate dropped from {baseline_hit_rate*100:.1f}% to {candidate_hit_rate*100:.1f}%")
            if not score_ok:
                reasons.append(f"Average Groundedness Score dropped from {baseline_score*100:.1f}% to {candidate_score*100:.1f}%")
            gate_reason = " | ".join(reasons)
            
            # Clean up candidate files
            if candidate_index_temp.exists():
                candidate_index_temp.unlink()
            if candidate_metadata_temp.exists():
                candidate_metadata_temp.unlink()
            print("Promotion Failed! Keeping current production RAG index.")
            print(f"Reason for rejection: {gate_reason}")
            
    else:
        print("Promotion Success! Candidate RAG index is now live in production.")
        
    print("==================================================")
    return gate_passed

if __name__ == "__main__":
    sample_mode = "--sample" in sys.argv
    success = run_pipeline(sample_mode=sample_mode)
    # Return code 0 on success, 1 on promotion failure (unless in sample/CI mode where we log it gracefully)
    sys.exit(0 if (success or sample_mode) else 1)
