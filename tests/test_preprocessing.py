import numpy as np
import pandas as pd
import pytest
from customers import HIGH_RISK, LOW_RISK

from data_preprocessing import clean_data, encode_features, scale_features, split_data
from feature_engineering import engineer_features


def test_clean_data_fixes_total_charges_drops_id_and_duplicates():
    raw = pd.DataFrame({
        "customerID": ["A", "B", "C"],
        "tenure": [0, 5, 5],
        "TotalCharges": [" ", "100.5", "100.5"],  # blank string, like the real dataset
        "Churn": ["No", "Yes", "Yes"],
    })

    cleaned = clean_data(raw)

    assert "customerID" not in cleaned.columns
    assert cleaned["TotalCharges"].notna().all()
    assert cleaned.loc[cleaned["tenure"] == 0, "TotalCharges"].iloc[0] == 0
    assert set(cleaned["Churn"]) <= {0, 1}
    assert len(cleaned) == 2  # B and C are identical once the ID is dropped


def test_engineer_features_values():
    df = engineer_features(pd.DataFrame([HIGH_RISK, LOW_RISK]))

    assert list(df["NumAddonServices"]) == [2, 4]
    assert list(df["HasProtection"]) == [0, 1]
    assert df["ChargePerService"].iloc[0] == pytest.approx(95.5 / 3, abs=0.01)


def test_average_charge_uses_monthly_rate_for_new_customers():
    new_customer = {**HIGH_RISK, "tenure": 0, "TotalCharges": 0.0, "MonthlyCharges": 50.0}

    df = engineer_features(pd.DataFrame([new_customer]))

    assert df["AverageChargePerMonth"].iloc[0] == 50.0


def test_tenure_group_boundaries():
    rows = [{**HIGH_RISK, "tenure": t} for t in (12, 13, 36, 37)]

    df = engineer_features(pd.DataFrame(rows))

    assert list(df["TenureGroup_encoded"]) == [0, 1, 1, 2]


def test_encode_features_leaves_only_numeric_columns():
    df = engineer_features(pd.DataFrame([HIGH_RISK]))

    encoded, _ = encode_features(df)

    assert encoded.select_dtypes(include="object").empty
    assert encoded["gender"].iloc[0] in (0, 1)
    assert encoded["Contract_Month-to-month"].iloc[0] == 1


def test_split_is_stratified_and_reproducible():
    df = pd.DataFrame({"x": range(1000), "Churn": [1] * 250 + [0] * 750})

    X_train, X_test, y_train, y_test = split_data(df)
    X_train2, _, _, _ = split_data(df)

    assert len(X_train) == 800 and len(X_test) == 200
    assert y_train.mean() == pytest.approx(0.25, abs=0.01)
    assert y_test.mean() == pytest.approx(0.25, abs=0.01)
    assert list(X_train.index) == list(X_train2.index)


def test_scaler_is_fit_on_training_data_only():
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame({"tenure": rng.normal(10, 2, 100)})
    X_test = pd.DataFrame({"tenure": rng.normal(30, 2, 50)})  # deliberately far from train

    X_train_s, X_test_s, scaler = scale_features(X_train, X_test, cols_to_scale=["tenure"])

    assert X_train_s["tenure"].mean() == pytest.approx(0, abs=1e-9)
    assert scaler.mean_[0] == pytest.approx(X_train["tenure"].mean())
    assert X_test_s["tenure"].mean() > 5  # not re-centred, so no information leaked from test
