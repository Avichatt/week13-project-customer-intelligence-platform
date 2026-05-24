import pytest
import pandas as pd
from pathlib import Path
from src import config

@pytest.fixture
def sample_bank_data():
    """Returns a mock bank marketing dataframe matching the schema."""
    data = {
        "age": [30, 45, 60],
        "job": ["management", "blue-collar", "technician"],
        "marital": ["single", "married", "divorced"],
        "education": ["university.degree", "basic.9y", "professional.course"],
        "default": ["no", "no", "no"],
        "housing": ["yes", "no", "yes"],
        "loan": ["no", "no", "no"],
        "contact": ["cellular", "telephone", "cellular"],
        "month": ["may", "jun", "jul"],
        "day_of_week": ["mon", "tue", "wed"],
        "duration": [200, 150, 400],
        "campaign": [1, 2, 1],
        "pdays": [999, 999, 6],
        "previous": [0, 0, 1],
        "poutcome": ["nonexistent", "nonexistent", "success"],
        "emp.var.rate": [1.1, -1.8, -0.1],
        "cons.price.idx": [93.994, 92.893, 93.918],
        "cons.conf.idx": [-36.4, -46.2, -42.7],
        "euribor3m": [4.857, 1.299, 4.963],
        "nr.employed": [5191.0, 5099.1, 5228.1],
        "y": ["no", "no", "yes"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_complaints_data():
    """Returns a mock complaints dataframe matching the schema."""
    data = {
        "complaint_id": [10001, 10002, 10003],
        "date_received": ["2024-01-10", "2024-02-15", "2024-03-20"],
        "product": ["Credit card", "Mortgage", "Debt collection"],
        "issue": ["Unauthorized charges", "Payment processing delay", "Harassing calls"],
        "consumer_complaint_narrative": [
            "I saw unauthorized charges on my credit card that I did not make. The company refused to reverse them.",
            "My mortgage payment processing was delayed by the servicer resulting in a late fee which is unfair.",
            "Debt collectors are calling me multiple times a day at work despite me asking them to stop harassing me."
        ],
        "company": ["Acme Cards", "Best Loans", "Collect Corp"],
        "company_response": ["Closed with monetary relief", "Closed with explanation", "Closed with explanation"]
    }
    return pd.DataFrame(data)
