import pandera as pa
import pandas as pd
from datetime import datetime

# Job list from dataset documentation
JOBS = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", 
        "self-employed", "services", "student", "technician", "unemployed", "unknown"]
MARITAL = ["divorced", "married", "single", "unknown"]
EDUCATION = ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", 
             "professional.course", "university.degree", "unknown"]
BINARY_YES_NO = ["yes", "no", "unknown"]
CONTACT_TYPES = ["cellular", "telephone"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DAYS_OF_WEEK = ["mon", "tue", "wed", "thu", "fri"]
POUTCOME = ["failure", "nonexistent", "success"]

# Pandera Schema for Bank Marketing Data (Raw semi-colon delimited)
bank_marketing_schema = pa.DataFrameSchema({
    "age": pa.Column(int, checks=pa.Check.in_range(18, 100)),
    "job": pa.Column(str, checks=pa.Check.isin(JOBS)),
    "marital": pa.Column(str, checks=pa.Check.isin(MARITAL)),
    "education": pa.Column(str, checks=pa.Check.isin(EDUCATION)),
    "default": pa.Column(str, checks=pa.Check.isin(BINARY_YES_NO)),
    "housing": pa.Column(str, checks=pa.Check.isin(BINARY_YES_NO)),
    "loan": pa.Column(str, checks=pa.Check.isin(BINARY_YES_NO)),
    "contact": pa.Column(str, checks=pa.Check.isin(CONTACT_TYPES)),
    "month": pa.Column(str, checks=pa.Check.isin(MONTHS)),
    "day_of_week": pa.Column(str, checks=pa.Check.isin(DAYS_OF_WEEK)),
    "duration": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0)),
    "campaign": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(1)),
    "pdays": pa.Column(int, checks=pa.Check.isin(list(range(0, 1000)))), # 0 to 999
    "previous": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0)),
    "poutcome": pa.Column(str, checks=pa.Check.isin(POUTCOME)),
    "emp.var.rate": pa.Column(float, required=False),
    "cons.price.idx": pa.Column(float, required=False),
    "cons.conf.idx": pa.Column(float, required=False),
    "euribor3m": pa.Column(float, required=False),
    "nr.employed": pa.Column(float, required=False),
    "y": pa.Column(str, checks=pa.Check.isin(["yes", "no"]))
})

# Pandera Schema for CFPB Complaints Data
cfpb_complaints_schema = pa.DataFrameSchema({
    "complaint_id": pa.Column(pa.Int, required=False),
    "date_received": pa.Column(str, required=True),
    "product": pa.Column(str, required=True),
    "issue": pa.Column(str, required=True),
    "consumer_complaint_narrative": pa.Column(str, checks=pa.Check.str_length(min_value=20)),
    "company": pa.Column(str, required=True),
    "company_response": pa.Column(str, required=False, nullable=True)
})

def validate_date_string(val):
    try:
        # Standard formats: YYYY-MM-DD
        datetime.strptime(val[:10], "%Y-%m-%d")
        return True
    except ValueError:
        return False

def run_business_rule_validations(bank_df, complaints_df):
    """
    Run 5+ specific business rule validations.
    Returns dict of results.
    """
    results = {}
    
    # Rule 1: Age check
    invalid_age = bank_df[(bank_df["age"] < 18) | (bank_df["age"] > 100)]
    results["age_rule_passed"] = len(invalid_age) == 0
    
    # Rule 2: Campaign contacts check
    invalid_campaign = bank_df[bank_df["campaign"] < 1]
    results["campaign_rule_passed"] = len(invalid_campaign) == 0
    
    # Rule 3: Duration check
    invalid_duration = bank_df[bank_df["duration"] < 0]
    results["duration_rule_passed"] = len(invalid_duration) == 0
    
    # Rule 4: Target y values check
    invalid_y = bank_df[~bank_df["y"].isin(["yes", "no"])]
    results["target_rule_passed"] = len(invalid_y) == 0
    
    # Rule 5: pdays range check
    invalid_pdays = bank_df[(bank_df["pdays"] < 0) | (bank_df["pdays"] > 999)]
    results["pdays_rule_passed"] = len(invalid_pdays) == 0
    
    # Rule 6: Complaint narrative length check
    invalid_narrative = complaints_df[complaints_df["consumer_complaint_narrative"].str.len() < 20]
    results["narrative_length_passed"] = len(invalid_narrative) == 0
    
    # Rule 7: Complaint date parse check
    invalid_dates = complaints_df[~complaints_df["date_received"].apply(validate_date_string)]
    results["complaint_date_passed"] = len(invalid_dates) == 0
    
    return results

def validate_datasets(bank_csv_path, complaints_csv_path, sep=";"):
    """
    Validates both datasets and returns schemas and business rules validation results.
    """
    print("Validating datasets...")
    
    # Load
    bank_df = pd.read_csv(bank_csv_path, sep=sep)
    complaints_df = pd.read_csv(complaints_csv_path)
    
    # Pandera validation
    bank_passed = True
    bank_errors = ""
    try:
        bank_marketing_schema.validate(bank_df, lazy=True)
    except Exception as e:
        bank_passed = False
        bank_errors = str(e)
        
    complaints_passed = True
    complaints_errors = ""
    try:
        cfpb_complaints_schema.validate(complaints_df, lazy=True)
    except Exception as e:
        complaints_passed = False
        complaints_errors = str(e)
        
    # Business rules validation
    business_results = run_business_rule_validations(bank_df, complaints_df)
    
    validation_report = {
        "bank_schema_passed": bank_passed,
        "bank_schema_errors": bank_errors,
        "complaints_schema_passed": complaints_passed,
        "complaints_schema_errors": complaints_errors,
        "business_rules": business_results,
        "all_passed": bank_passed and complaints_passed and all(business_results.values())
    }
    
    return validation_report

if __name__ == "__main__":
    from src import config
    # Test validation on samples if they exist
    bank_sample = config.SAMPLE_DATA_DIR / "bank-additional-full-sample.csv"
    complaints_sample = config.SAMPLE_DATA_DIR / "cfpb_complaints_sample.csv"
    
    if bank_sample.exists() and complaints_sample.exists():
        report = validate_datasets(bank_sample, complaints_sample, sep=";")
        print("Validation report:")
        import pprint
        pprint.pprint(report)
    else:
        print("Samples not found. Download them first.")
