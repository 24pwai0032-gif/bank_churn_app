"""
main.py – FastAPI backend for Bank Churn Prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from app.model import predict_churn

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="Bank Churn Prediction API",
    description="Predicts whether a bank customer will churn (leave the bank).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    CreditScore:     int   = Field(..., ge=300, le=900,  example=650)
    Geography:       Literal["France", "Germany", "Spain"] = Field(..., example="France")
    Gender:          Literal["Male", "Female"]              = Field(..., example="Male")
    Age:             int   = Field(..., ge=18,  le=100,  example=40)
    Tenure:          int   = Field(..., ge=0,   le=10,   example=5)
    Balance:         float = Field(..., ge=0,            example=50000.0)
    NumOfProducts:   int   = Field(..., ge=1,   le=4,    example=2)
    HasCrCard:       int   = Field(..., ge=0,   le=1,    example=1)
    IsActiveMember:  int   = Field(..., ge=0,   le=1,    example=0)
    EstimatedSalary: float = Field(..., ge=0,            example=75000.0)


class PredictionResponse(BaseModel):
    prediction:        int
    churn_probability: float
    result:            str


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "message":     "🏦 Bank Churn Prediction API is running!",
        "docs":        "/docs",
        "predict_url": "/predict  [POST]",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Accepts customer features as JSON and returns:
    - **prediction**: 0 (No Churn) or 1 (Churn)
    - **churn_probability**: probability of churn (0–1)
    - **result**: human-readable label
    """
    try:
        result = predict_churn(customer.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
