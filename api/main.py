"""
api/main.py
-----------
FastAPI service that serves the trained churn model.

Run locally (from the project root, after `python src/train_model.py`):
    uvicorn api.main:app --reload

Interactive docs (Swagger UI): http://localhost:8000/docs
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from inference import DEFAULT_THRESHOLD, load_bundle, predict  # noqa: E402

DEFAULT_MODEL_PATH = ROOT / "models" / "churn_model.pkl"

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]

EXAMPLE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 191.0,
}


class Customer(BaseModel):
    """One customer profile, using the same columns as the Telco dataset."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_CUSTOMER})

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=500)
    TotalCharges: float | None = Field(
        default=None, ge=0,
        description="Optional. Defaults to MonthlyCharges x tenure when omitted.",
    )


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: Literal["High Risk", "Low Risk"]
    threshold: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup. If it is missing, the API still starts."""
    path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    app.state.bundle = load_bundle(path) if path.exists() else None
    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts the probability that a telecom customer will churn.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    """Returns 200 when the model is loaded, 503 otherwise."""
    loaded = request.app.state.bundle is not None
    return JSONResponse(
        status_code=200 if loaded else 503,
        content={"status": "ok" if loaded else "model not loaded", "model_loaded": loaded},
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(
    customer: Customer,
    request: Request,
    threshold: float = Query(
        DEFAULT_THRESHOLD, ge=0.0, le=1.0,
        description="Probability at or above which a customer is labelled High Risk",
    ),
):
    """Predict churn probability and risk label for one customer."""
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train it or set MODEL_PATH.")

    row = customer.model_dump()
    if row["TotalCharges"] is None:
        row["TotalCharges"] = row["MonthlyCharges"] * row["tenure"]

    result = predict(pd.DataFrame([row]), bundle, threshold)
    return PredictionResponse(
        churn_probability=float(result["ChurnProbability"].iloc[0]),
        prediction=result["ChurnPrediction"].iloc[0],
        threshold=threshold,
    )
