"""
model.py – Loads the trained pipeline and exposes predict_churn()
"""

import pickle
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "model.pkl"

# Load once at import time
with open(MODEL_PATH, "rb") as f:
    _pipeline = pickle.load(f)


def predict_churn(features: dict) -> dict:
    """
    Accept a flat dict of raw feature values (Geography/Gender as strings)
    and return:
        { "prediction": 0 or 1,
          "churn_probability": float,
          "result": "Churn" | "No Churn" }
    """
    EXPECTED = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]

    # Build DataFrame in the exact column order the pipeline was trained on
    df = pd.DataFrame([{k: features[k] for k in EXPECTED}])

    prediction   = int(_pipeline.predict(df)[0])
    probability  = float(_pipeline.predict_proba(df)[0][1])

    return {
        "prediction":        prediction,
        "churn_probability": round(probability, 4),
        "result":            "Churn" if prediction == 1 else "No Churn",
    }
