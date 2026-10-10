import numpy as np
import pandas as pd
import pytest
from customers import HIGH_RISK, LOW_RISK

from inference import build_features, predict


def test_features_match_training_columns(model_bundle):
    names = model_bundle["feature_names"]

    X = build_features(pd.DataFrame([HIGH_RISK]), names)

    assert list(X.columns) == names
    assert X.shape == (1, len(names))
    assert not X.isna().any().any()


def test_unseen_category_becomes_all_zero_one_hot(model_bundle):
    names = model_bundle["feature_names"]
    odd_customer = {**HIGH_RISK, "PaymentMethod": "Crypto"}

    X = build_features(pd.DataFrame([odd_customer]), names)

    payment_cols = [c for c in names if c.startswith("PaymentMethod_")]
    assert list(X.columns) == names
    assert X[payment_cols].sum(axis=1).iloc[0] == 0


def test_probabilities_are_valid_and_risk_ordering_makes_sense(model_bundle):
    result = predict(pd.DataFrame([HIGH_RISK, LOW_RISK]), model_bundle)

    probs = result["ChurnProbability"]
    assert probs.between(0, 1).all()
    assert probs.iloc[0] > probs.iloc[1]


def test_threshold_controls_the_label(model_bundle):
    customer = pd.DataFrame([HIGH_RISK])

    assert predict(customer, model_bundle, threshold=0.0)["ChurnPrediction"].iloc[0] == "High Risk"
    assert predict(customer, model_bundle, threshold=1.0)["ChurnPrediction"].iloc[0] == "Low Risk"


def test_matches_original_predict_churn(model_bundle):
    """The new inference code must give the same answers as the original CLI code."""
    pytest.importorskip("seaborn")  # train_model imports the plotting stack
    from train_model import predict_churn

    customers = pd.DataFrame([HIGH_RISK, LOW_RISK])

    original = predict_churn(customers, model_bundle)
    new = predict(customers, model_bundle)

    assert np.allclose(original["ChurnProbability"], new["ChurnProbability"], atol=1e-4)
