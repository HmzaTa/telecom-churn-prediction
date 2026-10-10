"""
src/inference.py
----------------
Lightweight prediction code used by the API and the tests.

It reuses the exact same preprocessing functions as training
(feature_engineering.py and data_preprocessing.py), but does not import any
plotting or training code, so the API image stays small.
"""

import contextlib
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_preprocessing import encode_features
from feature_engineering import engineer_features

# Columns that were standardised with StandardScaler during training
SCALE_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "AverageChargePerMonth"]
DEFAULT_THRESHOLD = 0.5
REQUIRED_KEYS = {"model", "scaler", "feature_names"}


def load_bundle(path) -> dict:
    """Load the saved model bundle (model + scaler + feature names)."""
    bundle = joblib.load(Path(path))
    missing = REQUIRED_KEYS - set(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")
    return bundle


def build_features(customers: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """
    Turn raw customer rows into the exact feature matrix the model was trained on.

    Steps mirror training: fix TotalCharges, engineer features, encode categories,
    then align the columns to the training feature list (a category that is not
    present in the input becomes a column of zeros, never a missing column).
    """
    df = customers.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # The training helpers print progress messages; keep API logs clean.
    with contextlib.redirect_stdout(io.StringIO()):
        df = engineer_features(df)
        df, _ = encode_features(df)

    return df.reindex(columns=feature_names, fill_value=0).fillna(0).astype(float)


def predict(customers: pd.DataFrame, bundle: dict,
            threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """
    Predict churn for raw customer rows.

    Returns a DataFrame with ChurnProbability and ChurnPrediction
    ("High Risk" when probability >= threshold, otherwise "Low Risk").
    """
    X = build_features(customers, bundle["feature_names"])

    scale_cols = [c for c in SCALE_COLS if c in X.columns]
    X[scale_cols] = bundle["scaler"].transform(X[scale_cols])

    probs = bundle["model"].predict_proba(X)[:, 1]

    result = pd.DataFrame(index=customers.index)
    result["ChurnProbability"] = probs.round(4)
    result["ChurnPrediction"] = np.where(probs >= threshold, "High Risk", "Low Risk")
    return result
