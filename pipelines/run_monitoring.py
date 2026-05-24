import sys
import pandas as pd
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src import config
from src.monitoring.drift import introduce_synthetic_drift, check_data_drift
from src.monitoring.rag_quality import run_rag_evaluation
from src.monitoring.report import generate_unified_monitoring_report

def run_monitoring_pipeline(sample_mode=False):
    print("==================================================")
    print("STARTING CUSTOMER INTEL PLATFORM - MONITORING PIPELINE")
    print("==================================================")
    
    # 1. Load Data for Drift Analysis
    bank_csv = (
        config.SAMPLE_DATA_DIR / "bank-additional-full-sample.csv"
        if sample_mode else
        config.RAW_DATA_DIR / "bank-additional-full.csv"
    )
    
    if not bank_csv.exists():
        print(f"Dataset {bank_csv} not found. Run download/ml pipelines first.")
        return False
        
    print(f"Reading reference data from {bank_csv}...")
    df_ref = pd.read_csv(bank_csv, sep=";")
    
    # 2. Introduce synthetic drift
    print("Creating drifted production dataset...")
    df_curr = introduce_synthetic_drift(df_ref)
    
    # 3. Check data drift using Evidently
    drift_summary = check_data_drift(df_ref, df_curr)
    print(f"Drift Summary: {drift_summary}")
    
    # 4. Run RAG evaluation suite
    # In sample mode we run with a relaxed threshold or mock where index might be small
    rag_threshold = 0.2 if sample_mode else 0.3
    rag_summary = run_rag_evaluation(threshold=rag_threshold)
    print(f"RAG Evaluation - Avg Score: {rag_summary['average_score'] * 100:.1f}%, Success Rate: {rag_summary['success_rate'] * 100:.1f}%")
    
    # 5. Generate gorgeous HTML Dashboard Report
    report_path = generate_unified_monitoring_report(
        drift_summary=drift_summary,
        rag_summary=rag_summary,
        output_filename="monitoring_report.html"
    )
    
    print("==================================================")
    print(f"Monitoring report generated successfully.")
    print(f"View report at: {report_path}")
    print("==================================================")
    return True

if __name__ == "__main__":
    sample_mode = "--sample" in sys.argv
    success = run_monitoring_pipeline(sample_mode=sample_mode)
    sys.exit(0 if success else 1)
