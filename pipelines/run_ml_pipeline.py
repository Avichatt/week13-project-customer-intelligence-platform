import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src import config
from src.data.download import download_bank_marketing, make_bank_marketing_sample
from src.data.validate import validate_datasets
from src.ml.train import run_training_pipeline

def run_pipeline(sample_mode=False):
    print("==================================================")
    print("STARTING CUSTOMER INTEL PLATFORM - ML LANE PIPELINE")
    print("==================================================")
    
    # 1. Download
    download_bank_marketing()
    make_bank_marketing_sample()
    
    # Check that complaints sample is also available for validation step
    complaints_sample = config.SAMPLE_DATA_DIR / "cfpb_complaints_sample.csv"
    if not complaints_sample.exists():
        from src.data.download import download_cfpb_complaints
        download_cfpb_complaints(limit=100)
        
    # 2. Validate
    bank_csv = (
        config.SAMPLE_DATA_DIR / "bank-additional-full-sample.csv"
        if sample_mode else
        config.RAW_DATA_DIR / "bank-additional-full.csv"
    )
    
    print(f"Running data validation checks on {bank_csv}...")
    report = validate_datasets(bank_csv, complaints_sample, sep=";")
    
    if not report["all_passed"]:
        print("WARNING: Data validation checks did not fully pass. Details:")
        print(f"Bank Schema Passed: {report['bank_schema_passed']}")
        print(f"Complaints Schema Passed: {report['complaints_schema_passed']}")
        print(f"Business Rules Passed: {report['business_rules']}")
    else:
        print("Data validation checks passed successfully.")
        
    # 3. Train + Evaluate + Gate Check
    print("Triggering model training and gate evaluations...")
    passed = run_training_pipeline(bank_csv, sample_mode=sample_mode)
    
    if passed:
        print("ML Pipeline completed. Best model is deployed.")
    else:
        print("ML Pipeline completed but model did not pass the deployment gate.")
        
    print("==================================================")
    return passed

if __name__ == "__main__":
    sample_mode = "--sample" in sys.argv
    success = run_pipeline(sample_mode=sample_mode)
    sys.exit(0 if success else 1)
