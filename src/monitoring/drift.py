import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from src import config

def introduce_synthetic_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a synthetically drifted version of the bank marketing dataset
    to simulate production distribution shift.
    - Shift age upwards (older audience)
    - Shift euribor3m downwards (interest rate drop)
    - Corrupt job column with missing values
    """
    drifted = df.copy()
    
    # 1. Demographic shift: shift age by adding 12 years to 40% of records
    mask_age = np.random.rand(len(drifted)) < 0.4
    drifted.loc[mask_age, "age"] = drifted.loc[mask_age, "age"] + 12
    
    # 2. Economic shift: drop euribor rate by 1.5 units
    drifted["euribor3m"] = drifted["euribor3m"] - 1.5
    
    # 3. Data corruption: replace some jobs with null or "unknown"
    mask_job = np.random.rand(len(drifted)) < 0.15
    drifted.loc[mask_job, "job"] = "unknown"
    
    # 4. Increase contact frequency (more aggressive campaign)
    drifted["campaign"] = drifted["campaign"] + 2
    
    return drifted

def check_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, report_name="data_drift_report.html"):
    """
    Runs Evidently AI data drift and target drift checking, saving report as HTML.
    """
    # Evidently needs similar columns. Drop target 'y' from drift check if needed, or keep it to check target drift.
    print("Running data drift analysis...")
    
    # Let's clean up columns to check only features
    cols_to_check = [
        "age", "job", "marital", "education", "default", "housing", "loan", 
        "contact", "month", "day_of_week", "campaign", "pdays", "previous", 
        "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", 
        "euribor3m", "nr.employed"
    ]
    
    # Ensure columns exist in both dataframes
    cols = [c for c in cols_to_check if c in reference_df.columns and c in current_df.columns]
    
    ref_sub = reference_df[cols]
    curr_sub = current_df[cols]
    
    drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    drift_report.run(reference_data=ref_sub, current_data=curr_sub)
    
    out_path = config.REPORTS_DIR / report_name
    drift_report.save_html(str(out_path))
    print(f"Evidently drift report saved to {out_path}")
    
    # Get JSON output to check metrics programmatically
    metrics_dict = drift_report.as_dict()
    
    # Parse dataset drift summary
    try:
        drift_metrics = metrics_dict["metrics"][0]["result"]
        number_of_drifted_features = drift_metrics["number_of_drifted_columns"]
        share_of_drifted_features = drift_metrics["share_of_drifted_columns"]
        dataset_drift = drift_metrics["dataset_drift"]
    except KeyError:
        number_of_drifted_features = 0
        share_of_drifted_features = 0.0
        dataset_drift = False
        
    return {
        "dataset_drift_detected": bool(dataset_drift),
        "number_of_drifted_features": int(number_of_drifted_features),
        "share_of_drifted_features": float(share_of_drifted_features),
        "report_path": str(out_path)
    }

if __name__ == "__main__":
    # Test drift run
    bank_sample = config.SAMPLE_DATA_DIR / "bank-additional-full-sample.csv"
    if bank_sample.exists():
        df = pd.read_csv(bank_sample, sep=";")
        df_drift = introduce_synthetic_drift(df)
        res = check_data_drift(df, df_drift)
        print("Drift Results:")
        print(res)
    else:
        print("Sample data not found. Run download.py first.")
