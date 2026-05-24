import pytest
import pandas as pd
from src.data.validate import bank_marketing_schema, cfpb_complaints_schema, run_business_rule_validations

def test_bank_schema_valid(sample_bank_data):
    # Should pass without exceptions
    bank_marketing_schema.validate(sample_bank_data)

def test_bank_schema_invalid(sample_bank_data):
    # Corrupt age
    invalid_df = sample_bank_data.copy()
    invalid_df.loc[0, "age"] = 150 # Out of bounds
    
    with pytest.raises(Exception):
        bank_marketing_schema.validate(invalid_df)

def test_complaints_schema_valid(sample_complaints_data):
    # Should pass without exceptions
    cfpb_complaints_schema.validate(sample_complaints_data)

def test_complaints_schema_invalid(sample_complaints_data):
    # Corrupt narrative length to be too short
    invalid_df = sample_complaints_data.copy()
    invalid_df.loc[0, "consumer_complaint_narrative"] = "Short"
    
    with pytest.raises(Exception):
        cfpb_complaints_schema.validate(invalid_df)

def test_business_rules(sample_bank_data, sample_complaints_data):
    res = run_business_rule_validations(sample_bank_data, sample_complaints_data)
    assert all(res.values()) is True
    
    # Corrupt values to fail rules
    corrupt_bank = sample_bank_data.copy()
    corrupt_bank.loc[0, "age"] = 105
    res_corrupt = run_business_rule_validations(corrupt_bank, sample_complaints_data)
    assert res_corrupt["age_rule_passed"] is False
