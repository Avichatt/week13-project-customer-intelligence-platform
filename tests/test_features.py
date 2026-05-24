import pytest
import pandas as pd
from src.data.features import clean_complaint_text, BankFeaturePipeline

def test_clean_complaint_text():
    raw_text = "I received a call from XXXX on 10/12/2023 regarding my credit card.  They were   harassing me."
    expected = "i received a call from on regarding my credit card. they were harassing me."
    
    cleaned = clean_complaint_text(raw_text)
    assert "xxxx" not in cleaned
    assert "XXXX" not in cleaned
    assert "  " not in cleaned
    assert cleaned.islower()

def test_bank_feature_pipeline_fit_transform(sample_bank_data):
    pipeline = BankFeaturePipeline()
    
    # Drop target column for pipeline input
    X = sample_bank_data.drop(columns=["y"])
    
    # Fit and transform
    X_processed = pipeline.fit_transform(X)
    
    assert X_processed.shape[0] == 3
    assert X_processed.shape[1] > 0
    assert "duration" not in pipeline._engineer_features(X).columns
    
    # Verify save & load works
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "pipeline.joblib")
        pipeline.save(filepath)
        
        loaded = BankFeaturePipeline.load(filepath)
        assert loaded.is_fitted is True
        X_trans = loaded.transform(X)
        assert X_trans.shape == X_processed.shape
