from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("artifacts/model.joblib")

app = FastAPI(
    title="Company Bankruptcy Risk Modeling API",
    description="Predicts company bankruptcy probability using Logistic Regression.",
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    roa_c_before_interest_and_depreciation_before_interest: float = Field(
        ..., description="ROA(C) before interest and depreciation before interest"
    )
    operating_gross_margin: float = Field(
        ..., description="Operating Gross Margin"
    )
    current_ratio: float = Field(
        ..., description="Current Ratio"
    )
    debt_ratio_percent: float = Field(
        ..., description="Debt ratio %"
    )
    net_worth_assets: float = Field(
        ..., description="Net worth/Assets"
    )

class PredictionResponse(BaseModel):
    prediction: int
    bankruptcy_probability: float
    non_bankruptcy_probability: float
    status: str

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model.py first."
        )
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    model_load_error = None
except Exception as e:
    model = None
    model_load_error = str(e)

@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "Company Bankruptcy Risk Modeling API is running.",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
def health() -> Dict[str, str]:
    if model is None:
        return {
            "status": "error",
            "message": model_load_error,
        }
    return {
        "status": "ok",
        "message": "Model loaded successfully.",
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model unavailable: {model_load_error}",
        )

    try:
        input_df = pd.DataFrame([{
            "ROA(C) before interest and depreciation before interest":
                request.roa_c_before_interest_and_depreciation_before_interest,
            "Operating Gross Margin":
                request.operating_gross_margin,
            "Current Ratio":
                request.current_ratio,
            "Debt ratio %":
                request.debt_ratio_percent,
            "Net worth/Assets":
                request.net_worth_assets,
        }])

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        return PredictionResponse(
            prediction=prediction,
            bankruptcy_probability=float(probabilities[1]),
            non_bankruptcy_probability=float(probabilities[0]),
            status="success",
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}",
        )