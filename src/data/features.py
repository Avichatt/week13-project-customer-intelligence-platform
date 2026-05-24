import re
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src import config

def clean_complaint_text(text: str) -> str:
    """
    Clean CFPB consumer complaints narrative text.
    Strips 'XXXX' PII masks, lowercases, and cleans whitespaces.
    """
    if not isinstance(text, str):
        return ""
    
    # Strip XXXX PII masks
    cleaned = re.sub(r'X{2,}', '', text)
    cleaned = re.sub(r'x{2,}', '', cleaned)
    
    # Strip dates or zip code patterns like xx/xx/xxxx
    cleaned = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', cleaned)
    
    # Lowercase
    cleaned = cleaned.lower()
    
    # Normalize whitespaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

class BankFeaturePipeline:
    def __init__(self):
        self.numerical_cols = [
            "age", "campaign", "previous", "emp.var.rate", 
            "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"
        ]
        self.categorical_cols = [
            "job", "marital", "education", "default", "housing", 
            "loan", "contact", "month", "day_of_week", "poutcome"
        ]
        self.preprocessor = None
        self.is_fitted = False
        
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers interaction and derived features.
        Modifies and returns a copy of the dataframe.
        """
        df = df.copy()
        
        # 1. Drop duration (known data leakage source)
        if "duration" in df.columns:
            df = df.drop(columns=["duration"])
            
        # 2. Engineer pdays: 999 means never contacted previously.
        # Create binary indicator "was_previously_contacted"
        df["was_previously_contacted"] = (df["pdays"] != 999).astype(int)
        
        # Binned pdays (0: never contacted, 1: 0-7 days, 2: 8-15 days, 3: 16+ days)
        df["pdays_bin"] = pd.cut(
            df["pdays"], 
            bins=[-1, 7, 15, 998, 1000], 
            labels=["recent", "moderate", "distant", "never"]
        ).astype(str)
        
        # 3. Create campaign contact intensity bins
        df["campaign_intensity"] = pd.cut(
            df["campaign"],
            bins=[0, 2, 5, 10, 100],
            labels=["low", "medium", "high", "extreme"]
        ).astype(str)
        
        # 4. Economic indicator ratio: relationship between consumer price index and euribor interest rate
        df["price_interest_ratio"] = df["cons.price.idx"] / (df["euribor3m"] + 0.1)
        
        return df

    def fit(self, X: pd.DataFrame):
        """
        Fits the preprocessing scaling and encoding pipeline on engineered training features.
        """
        X_eng = self._engineer_features(X)
        
        # Update columns lists for fitted preprocessing
        num_cols = self.numerical_cols + ["was_previously_contacted", "price_interest_ratio"]
        cat_cols = self.categorical_cols + ["pdays_bin", "campaign_intensity"]
        
        # We need a ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            ]
        )
        
        self.preprocessor.fit(X_eng)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms input dataframe using the fitted preprocessing pipeline.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted yet. Call fit first.")
            
        X_eng = self._engineer_features(X)
        return self.preprocessor.transform(X_eng)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)
        
    def save(self, filepath):
        """Save pipeline artifact."""
        joblib.dump(self, filepath)
        print(f"Saved BankFeaturePipeline to {filepath}")
        
    @staticmethod
    def load(filepath):
        """Load pipeline artifact."""
        pipeline = joblib.load(filepath)
        print(f"Loaded BankFeaturePipeline from {filepath}")
        return pipeline
